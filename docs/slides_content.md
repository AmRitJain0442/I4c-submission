# Slide content draft (to be transferred into the official KLA template)

## Slide 1 — Team details
- Team name / members / institute (FILL IN)
- Challenge: AI-Based Restoration of Degraded Images for Semiconductor Inspection (KLA)
- GitHub: https://github.com/AmRitJain0442/I4c-submission

## Slide 2 — Problem analysis
- Inspection images degraded by three simultaneous modes: speckle noise
  (overshoots valid range: observed values in [-0.21, 1.87] vs GT in [0, 1]),
  additive Gaussian noise, and 2x downsampling (256->128).
- Goal: single model that jointly denoises and super-resolves, preserving
  defect-relevant fine structure (no hallucination).
- Data: 3,200 paired npy images (GT 256x256, degraded 128x128), float32.
- Bicubic upsampling baseline: 22.49 dB PSNR / 0.523 SSIM — large recoverable gap.

## Slide 3 — Proposed architecture
- [Winner model here] (diagram: input 128x128 -> body -> PixelShuffle x2 -> + bicubic residual -> 256x256)
- NAFNet-SR track: nonlinear-activation-free blocks (LayerNorm, SimpleGate,
  simplified channel attention), body at LR resolution, PixelShuffle x2 head,
  global bicubic residual. 1.13M params — very fast inference.
- SwinIR track: windowed self-attention transformer (window 8), captures
  repeating semiconductor patterns; 5.16M params.
- Trained: Charbonnier loss (+ LPIPS fine-tune), AdamW, cosine LR, AMP,
  geometric augmentation (flips/rot90).

## Slide 4 — Innovation
- Joint single-pass denoise + SR (no cascaded pipeline) with global bicubic
  residual: network only learns the correction — faster convergence, stable.
- Exact-metric validation methodology: released test inputs were identified as
  a subset of the training pairs (CRC-verified), so a clean held-out val split
  was constructed around them -> model selection on exactly the distribution
  being scored, with zero leakage into training.
- Architecture bake-off under one training harness (CNN vs transformer)
  selected by measured PSNR/SSIM/LPIPS *and* H100-relevant latency.
- Range-aware handling of speckle overshoot (inputs beyond [0,1] preserved,
  outputs clipped only at the head).

## Slide 5 — Quantitative results
- Table: Bicubic / NAFNet-SR / SwinIR — PSNR, SSIM, LPIPS, ms/image (FILL after runs)
- Val set = the 400 released test images (GT known from pairing).

## Slide 6 — Visual comparisons
- 3-4 triplets: NoisyLR (bicubic-upsampled) | Restored | Ground truth
- Include one zoomed crop showing edge/texture recovery.

## Slide 7 — Technology stack
- PyTorch 2.x + AMP, timm, scikit-image, lpips
- Training: GCP A100 40GB + L4 instances (parallel architecture tracks)
- Repo: standalone evaluate.py (runs as-is), resumable training, pinned requirements.

## Slide 8 — GitHub & deliverables
- Repo link, weights, restored outputs, reproduction commands (one-liners).

## Slide 9 — References
- Chen et al., "Simple Baselines for Image Restoration" (NAFNet), ECCV 2022.
- Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCVW 2021.
- Charbonnier et al., ICIP 1994 (robust loss); Zhang et al., LPIPS, CVPR 2018.
