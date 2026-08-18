"""Quick EDA: value ranges and stats of GT vs NoisyLR pairs."""
import numpy as np
from pathlib import Path

root = Path(__file__).parent / "data"
gt_dir = root / "train" / "GT"
lr_dir = root / "train" / "NoisyLR"
test_dir = root / "NoisyLR"

ids = sorted(p.stem for p in gt_dir.glob("*.npy"))
print(f"train pairs: {len(ids)}, test inputs: {len(list(test_dir.glob('*.npy')))}")

rng = np.random.default_rng(0)
sample = rng.choice(ids, 200, replace=False)

gt_min, gt_max, lr_min, lr_max = [], [], [], []
gt_means, lr_means = [], []
for i in sample:
    gt = np.load(gt_dir / f"{i}.npy")
    lr = np.load(lr_dir / f"{i}.npy")
    gt_min.append(gt.min()); gt_max.append(gt.max()); gt_means.append(gt.mean())
    lr_min.append(lr.min()); lr_max.append(lr.max()); lr_means.append(lr.mean())

print(f"GT  shape {gt.shape} dtype {gt.dtype}")
print(f"GT  min [{np.min(gt_min):.4f}, {np.max(gt_min):.4f}]  max [{np.min(gt_max):.4f}, {np.max(gt_max):.4f}]  mean {np.mean(gt_means):.4f}")
print(f"LR  shape {lr.shape} dtype {lr.dtype}")
print(f"LR  min [{np.min(lr_min):.4f}, {np.max(lr_min):.4f}]  max [{np.min(lr_max):.4f}, {np.max(lr_max):.4f}]  mean {np.mean(lr_means):.4f}")

# Baseline: bicubic 2x upsample of noisy LR vs GT (PSNR/SSIM)
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

psnrs, ssims = [], []
for i in sample[:50]:
    gt = np.load(gt_dir / f"{i}.npy").astype(np.float32)
    lr = np.load(lr_dir / f"{i}.npy").astype(np.float32)
    up = F.interpolate(torch.from_numpy(lr)[None, None], scale_factor=2, mode="bicubic", align_corners=False)[0, 0].numpy()
    up = np.clip(up, 0, 1)
    mse = np.mean((up - gt) ** 2)
    psnrs.append(10 * np.log10(1.0 / mse))
    ssims.append(ssim(gt, up, data_range=1.0))
print(f"Bicubic baseline (50 imgs): PSNR {np.mean(psnrs):.2f} dB, SSIM {np.mean(ssims):.4f}")
