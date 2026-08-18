"""Paired npy dataset for joint denoising + 2x SR.

NoisyLR: 128x128 float32 (noise overshoots [0,1]); GT: 256x256 float32 in [0,1].
Entire dataset (~1 GB) is preloaded into RAM.
"""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


def split_ids(data_root):
    """Val = the 400 ids that appear in the released test set; train = the rest."""
    data_root = Path(data_root)
    all_ids = sorted(p.stem for p in (data_root / "train" / "GT").glob("*.npy"))
    test_dir = data_root / "NoisyLR"
    if test_dir.exists():
        val_ids = sorted(p.stem for p in test_dir.glob("*.npy"))
    else:
        val_ids = all_ids[:: len(all_ids) // 400][:400]
    val_set = set(val_ids)
    train_ids = [i for i in all_ids if i not in val_set]
    return train_ids, val_ids


class PairDataset(Dataset):
    def __init__(self, data_root, ids, augment=False, lr_patch=None):
        root = Path(data_root) / "train"
        self.lr = [np.load(root / "NoisyLR" / f"{i}.npy") for i in ids]
        self.gt = [np.load(root / "GT" / f"{i}.npy") for i in ids]
        self.ids = list(ids)
        self.augment = augment
        self.lr_patch = lr_patch

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        lr, gt = self.lr[idx], self.gt[idx]
        if self.lr_patch:
            p = self.lr_patch
            y = np.random.randint(lr.shape[0] - p + 1)
            x = np.random.randint(lr.shape[1] - p + 1)
            lr = lr[y:y + p, x:x + p]
            gt = gt[2 * y:2 * (y + p), 2 * x:2 * (x + p)]
        if self.augment:
            k = np.random.randint(4)
            if k:
                lr, gt = np.rot90(lr, k), np.rot90(gt, k)
            if np.random.rand() < 0.5:
                lr, gt = np.fliplr(lr), np.fliplr(gt)
            if np.random.rand() < 0.5:
                lr, gt = np.flipud(lr), np.flipud(gt)
        lr = torch.from_numpy(np.ascontiguousarray(lr))[None]
        gt = torch.from_numpy(np.ascontiguousarray(gt))[None]
        return lr, gt
