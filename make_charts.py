"""Generate slide charts: validation PSNR curves + metrics comparison panels.

Inputs:  docs/logs/nafnet.log, docs/logs/swinir.log (training logs),
         docs/results.json  {"models": {name: {psnr, ssim, lpips, ms}}, "baseline": {...}}
Outputs: docs/figures/training_curves.png, docs/figures/metrics_panels.png
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
BASELINE_PSNR = 22.49

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def parse_log(path):
    epochs, psnrs = [], []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"epoch (\d+).*val PSNR ([\d.]+)", line)
        if m:
            epochs.append(int(m.group(1)))
            psnrs.append(float(m.group(2)))
    return epochs, psnrs


def training_curves(out):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    series = [("NAFNet-SR", "docs/logs/nafnet.log", BLUE),
              ("SwinIR", "docs/logs/swinir.log", ORANGE)]
    for name, log, color in series:
        if not Path(log).exists():
            continue
        e, p = parse_log(log)
        ax.plot(e, p, color=color, linewidth=2)
        ax.annotate(f"{name}  {p[-1]:.2f} dB", (e[-1], p[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=10.5, fontweight="bold", va="center")
    ax.axhline(BASELINE_PSNR, color=GRAY, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.annotate(f"bicubic baseline  {BASELINE_PSNR:.2f} dB", (0, BASELINE_PSNR),
                xytext=(4, 5), textcoords="offset points", color=MUTED, fontsize=9.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_title("Validation PSNR during training (400 held-out images)",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.grid(axis="y", color="#eceae5", linewidth=0.8)
    ax.margins(x=0.02)
    right = ax.get_xlim()[1]
    ax.set_xlim(ax.get_xlim()[0], right * 1.28)  # room for end labels
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def metrics_panels(out):
    res = json.loads(Path("docs/results.json").read_text())
    rows = [("Bicubic ×2", res["baseline"], GRAY),
            ("NAFNet-SR", res["models"]["nafnet"], BLUE),
            ("SwinIR", res["models"]["swinir"], ORANGE)]
    panels = [("PSNR (dB) — higher is better", "psnr", "{:.2f}"),
              ("SSIM — higher is better", "ssim", "{:.3f}"),
              ("LPIPS — lower is better", "lpips", "{:.3f}")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    for ax, (title, key, fmt) in zip(axes, panels):
        names = [r[0] for r in rows]
        vals = [r[1].get(key) for r in rows]
        colors = [r[2] for r in rows]
        keep = [i for i, v in enumerate(vals) if v is not None]
        names = [names[i] for i in keep]
        vals = [vals[i] for i in keep]
        colors = [colors[i] for i in keep]
        bars = ax.bar(names, vals, color=colors, width=0.62, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", color=INK, fontsize=10, fontweight="bold")
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)
        ax.grid(axis="y", color="#eceae5", linewidth=0.8, zorder=0)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.margins(y=0.18)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    training_curves("docs/figures/training_curves.png")
    if Path("docs/results.json").exists():
        metrics_panels("docs/figures/metrics_panels.png")
    else:
        print("docs/results.json missing — skipped metrics panels")
