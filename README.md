# AI-Based Restoration of Degraded Semiconductor Inspection Images

Solution for the SEMICON India Hackathon 2026 KLA challenge: joint **denoising
(speckle + Gaussian) and 2× super-resolution** of semiconductor inspection
images (128×128 noisy `.npy` → 256×256 clean `.npy`, single channel, float32).

## Approach

Two architectures were trained and compared on a held-out 400-image validation
split; the best checkpoint (by PSNR/SSIM/LPIPS) is shipped in `weights/`:

- **NAFNet-SR** — NAFNet body (Chen et al., ECCV 2022) running at LR
  resolution, PixelShuffle ×2 head, global bicubic residual. Small and fast.
- **SwinIR** — official SwinIR (Liang et al., ICCVW 2021) with
  `pixelshuffledirect` upsampler, adapted to 1 channel.

Training: Charbonnier loss (+ optional LPIPS fine-tune), AdamW, cosine LR with
warmup, AMP, flip/rot90 augmentation.

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.10 with a CUDA-capable PyTorch build recommended (CPU works but is
slow).

## Evaluation (KLA benchmark entry point)

Runs as-is, no editing required. Restores every `.npy` in `--input_dir`,
writes results to `--output_dir`, and prints mean per-image inference time.
With `--gt_dir` it also reports PSNR / SSIM / LPIPS.

```bash
python evaluate.py --input_dir path/to/NoisyLR --output_dir restored
python evaluate.py --input_dir path/to/NoisyLR --output_dir restored --gt_dir path/to/GT
```

Optional flags: `--checkpoint path/to.pth` (default `weights/best.pth`),
`--tta` (×8 self-ensemble; higher quality, ~8× slower).

## Training

Data layout expected under `data/`:

```
data/train/GT/*.npy        # 256x256 clean
data/train/NoisyLR/*.npy   # 128x128 degraded
data/NoisyLR/*.npy         # test inputs (used as val split ids)
```

```bash
python train.py --model nafnet --width 64 --blocks 32 --epochs 300 --batch 16
python train.py --model swinir --embed_dim 120 --epochs 300 --batch 16
```

## Results (400-image validation set)

| Model | PSNR (dB) | SSIM | LPIPS | ms/img (GPU) |
|---|---|---|---|---|
| Bicubic ×2 (baseline) | 23.00 | 0.555 | 0.431 | — |
| NAFNet-SR | TBD | TBD | TBD | TBD |
| SwinIR | TBD | TBD | TBD | TBD |

## Repository contents

- `evaluate.py` — standalone evaluation/restoration script (benchmark entry point)
- `train.py` — training loop
- `dataset.py` — paired npy data loader
- `models/` — NAFNet-SR and SwinIR architectures
- `weights/` — trained checkpoint used by `evaluate.py`
- `restored/` — restored outputs for the released test set
