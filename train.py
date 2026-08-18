"""Train joint denoise + 2x SR models on KLA paired npy data.

Examples:
  python train.py --model nafnet --width 64 --blocks 32 --epochs 300
  python train.py --model swinir --embed_dim 120 --epochs 300
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PairDataset, split_ids


def build_model(cfg):
    if cfg["model"] == "nafnet":
        from models.nafnet import NAFNetSR
        return NAFNetSR(width=cfg["width"], num_blocks=cfg["blocks"], upscale=2)
    elif cfg["model"] == "swinir":
        from models.swinir import SwinIR
        return SwinIR(
            upscale=2, in_chans=1, img_size=128, window_size=8,
            img_range=1.0, depths=[6] * cfg["swin_layers"],
            embed_dim=cfg["embed_dim"], num_heads=[6] * cfg["swin_layers"],
            mlp_ratio=2, upsampler="pixelshuffledirect", resi_connection="1conv",
        )
    raise ValueError(cfg["model"])


def charbonnier(pred, target, eps=1e-6):
    return torch.sqrt((pred - target) ** 2 + eps).mean()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_sum, n = 0.0, 0
    for lr, gt in loader:
        lr, gt = lr.to(device), gt.to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            out = model(lr)
        out = out.float().clamp(0, 1)
        mse = ((out - gt) ** 2).mean(dim=(1, 2, 3))
        psnr_sum += (10 * torch.log10(1.0 / mse)).sum().item()
        n += lr.size(0)
    model.train()
    return psnr_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["nafnet", "swinir"], default="nafnet")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=32)
    p.add_argument("--embed_dim", type=int, default=120)
    p.add_argument("--swin_layers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lpips_weight", type=float, default=0.0)
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--init_from", default=None, help="warm-start weights (e.g. for LPIPS fine-tune)")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    cfg = vars(args)
    run = args.run_name or f"{args.model}_{args.width if args.model=='nafnet' else args.embed_dim}"
    out_dir = Path(args.out_dir) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids, val_ids = split_ids(args.data_root)
    print(f"train {len(train_ids)} / val {len(val_ids)}", flush=True)
    train_ds = PairDataset(args.data_root, train_ids, augment=True)
    val_ds = PairDataset(args.data_root, val_ids, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=32, num_workers=2, pin_memory=True)

    model = build_model(cfg).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"{run}: {n_params/1e6:.2f}M params", flush=True)

    if args.compile:
        model = torch.compile(model)

    def raw(m):
        return getattr(m, "_orig_mod", m)

    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        raw(model).load_state_dict(ck["model"])
        print(f"warm-started from {args.init_from}", flush=True)

    lpips_fn = None
    if args.lpips_weight > 0:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        for q in lpips_fn.parameters():
            q.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.9))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    total_steps = args.epochs * len(train_dl)
    warmup = min(500, total_steps // 20)

    def lr_at(step):
        if step < warmup:
            return step / max(warmup, 1)
        t = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    start_epoch, best_psnr = 0, 0.0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw(model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_psnr = ck["epoch"] + 1, ck["best_psnr"]
        print(f"resumed at epoch {start_epoch}, best {best_psnr:.2f}", flush=True)

    log = open(out_dir / "log.txt", "a")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss_sum = 0.0
        for lr_img, gt in train_dl:
            lr_img, gt = lr_img.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                out = model(lr_img)
                loss = charbonnier(out, gt)
                if lpips_fn is not None:
                    o3 = (out.clamp(0, 1) * 2 - 1).repeat(1, 3, 1, 1)
                    g3 = (gt * 2 - 1).repeat(1, 3, 1, 1)
                    loss = loss + args.lpips_weight * lpips_fn(o3, g3).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item()

        msg = f"epoch {epoch} loss {loss_sum/len(train_dl):.5f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s"
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            psnr = validate(model, val_dl, device)
            msg += f" | val PSNR {psnr:.3f}"
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({"model": raw(model).state_dict(), "cfg": cfg, "epoch": epoch,
                            "val_psnr": psnr}, out_dir / "best.pth")
                msg += " *best*"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        torch.save({"model": raw(model).state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "cfg": cfg, "epoch": epoch, "best_psnr": best_psnr},
                   out_dir / "last.pth")

    print(f"done. best val PSNR {best_psnr:.3f}", flush=True)


if __name__ == "__main__":
    main()
