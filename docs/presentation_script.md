# Presentation Script — Team gitcommitall

Target: ~6–7 minutes total. Times per slide are a guide. Words in *italics* are
delivery notes, not things to say. Numbers below are your real measured results —
you can say them with confidence.

---

## Slide 1 — Team Details (~20 s)

> "Hi, we're team **gitcommitall** from BITS Pilani — I'm Amrit, and with me are
> Shubham and Saiswaroop. We took on KLA's image restoration challenge, and I'd
> like to walk you through how we solved it and what results we got."

*Keep it short. Judges don't need bios — get to the problem.*

---

## Slide 2 — Problem Statement (~45 s)

> "In semiconductor fabs, inspection images are what defect detection depends
> on — and they're degraded in three ways at once. First, **speckle noise** —
> and it's severe: pixel values in the data actually overshoot the valid range,
> we measured values from −0.2 up to 1.9 on images that should live between 0
> and 1. Second, **Gaussian noise** that blurs edges. Third, the images are
> **downsampled 2×** — from 256×256 to 128×128 — so fine detail is simply gone.
>
> The task: one AI model that takes a noisy 128×128 image and returns a clean
> 256×256 one — judged on SSIM, PSNR, LPIPS, inference speed on an H100, and
> how well it generalizes to structures it has never seen."

*The −0.2 to 1.9 measurement shows you actually explored the data — say it.*

---

## Slide 3 — Idea Description (~40 s)

> "Our idea is **one network, one pass** — no separate denoising and upscaling
> stages, because cascading them lets errors compound. The model learns the
> joint inverse of the whole degradation directly from KLA's 3,200 paired
> images.
>
> The key design choice: the network doesn't rebuild the image from scratch.
> We add its output to a simple bicubic upsample of the input — a **global
> residual** — so all of its capacity goes into removing noise and recovering
> detail. That makes training more stable and, importantly for inspection, it
> doesn't hallucinate structure that isn't there.
>
> And rather than trusting paper benchmarks, we trained **two architectures**
> — a CNN and a transformer — under the exact same harness, and submitted the
> one that measured better."

---

## Slide 4 — Proposed Solution (~45 s)

> "Track A was **NAFNet** — a very efficient CNN, 1.1 million parameters.
> Track B was **SwinIR** — a windowed-attention transformer, about 5 million
> parameters. Window attention suits this data well because semiconductor
> structures are repetitive — the same patterns recur across the image.
>
> Both trained on random 64×64 crops with flip and rotation augmentation,
> Charbonnier loss, cosine learning-rate schedule, mixed precision — in
> parallel on GCP, an A100 and an L4, roughly two hours each. We validated
> every five epochs on a strict 400-image held-out split and kept the best
> checkpoint.
>
> Then a final step that mattered a lot: a short **perceptual fine-tune** with
> LPIPS added to the loss — I'll show you what that did on the results slide."

---

## Slide 5 — Innovation and Uniqueness (~40 s)

> "Three things set our solution apart.
>
> One — the **global bicubic residual**: the network learns only the
> correction, never invents geometry. For defect inspection that's a safety
> property, not just a training trick.
>
> Two — we're **range-aware**: the speckle overshoot beyond [0,1] is real
> signal about the noise, so we feed it to the network unclipped and only clip
> at the output.
>
> Three — **honest measurement**: a CNN-versus-transformer bake-off under one
> identical harness, model selection on a strictly held-out validation set,
> and every number we report is reproducible from the repo with one command."

---

## Slide 6 — Impact and Benefits (~30 s)

> "What does this buy a fab? Restored, inspection-grade images from noisy,
> low-resolution captures in a single forward pass — about 85 milliseconds on
> a laptop GPU, which translates to just a few milliseconds on an H100, so it
> fits inline inspection throughput.
>
> Concretely: **+4.8 dB PSNR** over bicubic, SSIM up from 0.55 to 0.76, and
> LPIPS — the perceptual metric — down from 0.43 to **0.15**, a 66% reduction.
> Better images mean more reliable defect detection and fewer false rejects."

---

## Slide 7 — Results (~60 s — spend time here)

> "Here are the full numbers on our 400-image held-out validation set.
>
> Bicubic upsampling — the do-nothing baseline — sits at 23 dB. Plain NAFNet
> reaches 27.6; plain SwinIR wins quality at **28.1 dB**.
>
> Then the LPIPS fine-tune: it trades a third of a dB of PSNR for a massive
> perceptual gain — LPIPS drops from 0.27 to **0.146**, nearly half. Since the
> challenge scores all three metrics, the fine-tuned SwinIR is our submission
> — the bold row.
>
> *point at the training curve* — both models converged smoothly and cleared
> the baseline within five epochs.
>
> *point at the image triplet* — left is the degraded input, middle is our
> restoration, right is ground truth. The speckle is gone, edges are sharp,
> and the fine structures match the ground truth — no invented detail.
>
> One more thing we tested and rejected: test-time ensembling gave only 0.03
> dB for 8× the latency — so we ship the fast single pass."

---

## Slide 8 — Technology & Feasibility (~30 s)

> "The stack is deliberately standard: PyTorch 2 with mixed precision and
> torch.compile, trained on GCP. Total training cost is about two GPU-hours
> per model — so retraining as fab processes drift is cheap and practical.
>
> And we built the deliverable the way KLA's benchmark team needs it: a
> standalone `evaluate.py` that takes a folder of images and produces restored
> outputs, metrics, and per-image latency — we verified it runs from a fresh
> clone of the repo with zero edits."

---

## Slide 9 — GitHub & Video (~15 s)

> "Everything is public in this repository — training code, the trained
> weights, all 400 restored test outputs, and the evaluation script. One
> command reproduces every number in this deck."

---

## Slide 10 — References (~10 s)

> "Our work builds on NAFNet, SwinIR, and the LPIPS perceptual metric —
> referenced here. Thank you — happy to take questions."

---

# Likely Q&A — prepare these

**Q: Why SwinIR over NAFNet if NAFNet is 2× faster?**
> "All three quality metrics favor SwinIR, and at ~85 ms on a laptop GPU it's
> already far inside any realistic latency budget — on an H100 it's a few
> milliseconds. If a deployment were latency-critical, our NAFNet fine-tune is
> a drop-in swap; the repo ships both."

**Q: How do you know it generalizes out-of-distribution?**
> "The released test inputs are genuinely new scenes — we verified they don't
> match any training image. Our restorations on them are visually clean, and
> the model's design helps: the bicubic residual means unfamiliar content
> degrades gracefully toward a blurry-but-faithful image rather than
> hallucinating."

**Q: Why does PSNR drop after the LPIPS fine-tune — isn't that worse?**
> "PSNR rewards averaged-out, slightly blurry predictions. LPIPS measures
> perceptual similarity — texture and edge fidelity — which is closer to what
> a defect-detection system needs. Trading 0.3 dB for a 46% LPIPS improvement
> is a good trade under the challenge's own scoring."

**Q: Could you have used a diffusion model?**
> "We considered it. Diffusion models hallucinate plausible detail — that's
> the opposite of what inspection needs — and they're orders of magnitude
> slower at inference. A deterministic restoration network is the right tool."

**Q: What would you do with more time?**
> "Degradation-randomized augmentation for stronger OOD robustness, a wider
> architecture sweep, and knowledge distillation from SwinIR into the small
> NAFNet to get transformer quality at CNN speed."

---

*General delivery notes:*
- *Slide 7 is your money slide — slow down there.*
- *Say "we measured" / "we verified" often; it's your differentiator.*
- *If short on time, compress slides 1, 9, 10 — never slide 7.*
