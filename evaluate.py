"""Standalone evaluation script (KLA benchmark deliverable).

Restores every .npy image in --input_dir (128x128 noisy LR -> 256x256 clean),
writes outputs to --output_dir, and reports mean per-image inference time.
If --gt_dir is given, also reports PSNR / SSIM / LPIPS against ground truth.

Runs as-is, no editing required:
  python evaluate.py --input_dir path/to/NoisyLR --output_dir restored [--gt_dir path/to/GT]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from train import build_model


def load_model(checkpoint, device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck["cfg"]


@torch.no_grad()
def restore(model, lr_np, device, tta=False):
    x = torch.from_numpy(lr_np.astype(np.float32))[None, None].to(device)
    if not tta:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
        return out.float().clamp(0, 1)[0, 0].cpu().numpy()
    # x8 self-ensemble: 4 rotations x optional horizontal flip
    outs = []
    for flip in (False, True):
        xf = torch.flip(x, dims=[-1]) if flip else x
        for k in range(4):
            xr = torch.rot90(xf, k, dims=[-2, -1])
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                o = model(xr)
            o = torch.rot90(o.float(), -k, dims=[-2, -1])
            if flip:
                o = torch.flip(o, dims=[-1])
            outs.append(o)
    return torch.stack(outs).mean(0).clamp(0, 1)[0, 0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="directory of degraded .npy images")
    p.add_argument("--output_dir", default="restored", help="where restored .npy images are written")
    p.add_argument("--gt_dir", default=None, help="optional ground-truth dir for metrics")
    p.add_argument("--checkpoint", default="weights/best.pth")
    p.add_argument("--tta", action="store_true", help="x8 self-ensemble (higher quality, ~8x slower)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    print(f"model: {cfg['model']} | device: {device}")

    files = sorted(Path(args.input_dir).glob("*.npy"))
    assert files, f"no .npy files found in {args.input_dir}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # warmup so timing excludes CUDA init / cudnn autotune
    restore(model, np.load(files[0]), device, args.tta)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times, outputs = [], {}
    for f in files:
        lr = np.load(f)
        t0 = time.perf_counter()
        out = restore(model, lr, device, args.tta)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        np.save(out_dir / f.name, out.astype(np.float32))
        outputs[f.stem] = out

    print(f"restored {len(files)} images -> {out_dir}")
    print(f"mean inference time: {np.mean(times)*1000:.2f} ms/image "
          f"(median {np.median(times)*1000:.2f} ms)")

    if args.gt_dir:
        from skimage.metrics import structural_similarity as ssim_fn
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        except Exception as e:
            lpips_fn = None
            print(f"(LPIPS unavailable: {e})")

        psnrs, ssims, lps = [], [], []
        for f in files:
            gt_path = Path(args.gt_dir) / f.name
            if not gt_path.exists():
                continue
            gt = np.load(gt_path).astype(np.float32)
            out = outputs[f.stem]
            mse = np.mean((out - gt) ** 2)
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
            ssims.append(ssim_fn(gt, out, data_range=1.0))
            if lpips_fn is not None:
                with torch.no_grad():
                    o = torch.from_numpy(out)[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
                    g = torch.from_numpy(gt)[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
                    lps.append(lpips_fn(o, g).item())
        print(f"PSNR:  {np.mean(psnrs):.3f} dB")
        print(f"SSIM:  {np.mean(ssims):.4f}")
        if lps:
            print(f"LPIPS: {np.mean(lps):.4f}")


if __name__ == "__main__":
    main()
