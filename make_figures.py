"""Generate before/after comparison figures for the slides.

Usage: python make_figures.py --restored_dir restored --n 4 --out_dir docs/figures
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--restored_dir", default="restored")
    p.add_argument("--out_dir", default="docs/figures")
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args()

    root = Path(args.data_root)
    restored = Path(args.restored_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ids = sorted(p.stem for p in restored.glob("*.npy"))[:: max(1, len(list(restored.glob('*.npy'))) // args.n)][: args.n]
    for i in ids:
        lr = np.load(root / "NoisyLR" / f"{i}.npy")
        up = F.interpolate(torch.from_numpy(lr)[None, None], scale_factor=2, mode="bicubic",
                           align_corners=False)[0, 0].numpy().clip(0, 1)
        rest = np.load(restored / f"{i}.npy")
        gt_path = root / "train" / "GT" / f"{i}.npy"
        cols = [("Degraded input (bicubic x2)", up), ("Restored", rest)]
        if gt_path.exists():
            cols.append(("Ground truth", np.load(gt_path)))

        fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4.2))
        for ax, (title, img) in zip(axes, cols):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(f"Image {i}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"compare_{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out / f'compare_{i}.png'}")


if __name__ == "__main__":
    main()
