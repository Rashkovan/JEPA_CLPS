# step6_pipeline.py
# Maps to: LeWM paper (arXiv 2603.19312) — full evaluation pipeline
# Runs all steps in sequence, assembles a unified summary figure, and prints
# a summary table comparing model variants.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import step1_data
import step2_train
import step3_probe
import step4_surprise
import step5_ablations

FIG_DIR  = "figures"
DATA_DIR = "data"


def assemble_summary(labels_paths):
    """
    Stitch saved PNGs into a single 2×2 summary figure.
    """
    images, titles = [], []
    for title, path in labels_paths:
        if os.path.exists(path):
            images.append(plt.imread(path))
            titles.append(title)

    if not images:
        print("No figures found to assemble.")
        return

    n     = len(images)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 8))
    axes = np.array(axes).flatten()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("LeWM-mini: Full Pipeline Summary", fontsize=16,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    out = f"{FIG_DIR}/summary.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved summary figure → {out}")


def print_table(mse_s, r_s, mse_n, r_n, surprise_ratio):
    header = f"{'Model':<28} {'Probe MSE':>12} {'Probe r':>10} {'Surprise ratio':>16}"
    sep    = "-" * len(header)
    print("\n" + "=" * len(header))
    print("SUMMARY TABLE")
    print("=" * len(header))
    print(header)
    print(sep)
    print(f"{'With SIGReg (λ=0.1)':<28} {mse_s:>12.4f} {r_s:>10.4f} {surprise_ratio:>16.2f}")
    print(f"{'Without SIGReg (λ=0)':<28} {mse_n:>12.4f} {r_n:>10.4f} {'N/A':>16}")
    print("=" * len(header))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("LeWM-mini Pipeline")
    print("=" * 60)

    print("\n[Step 1] Generating data …")
    step1_data.main()

    print("\n[Step 2] Training models …")
    step2_train.main()

    print("\n[Step 3] Linear probing …")
    mse_s, r_s, mse_n, r_n = step3_probe.main()

    print("\n[Step 4] Surprise detection …")
    surprise_ratio = step4_surprise.main()

    print("\n[Step 5] Ablations …")
    step5_ablations.main()

    print_table(mse_s, r_s, mse_n, r_n, surprise_ratio)

    assemble_summary([
        ("Training curves  (Step 2)",       f"{FIG_DIR}/training_curves.png"),
        ("Linear probing  (Step 3)",         f"{FIG_DIR}/probe_scatter.png"),
        ("VoE surprise  (Step 4)",           f"{FIG_DIR}/surprise.png"),
        ("Ablations  (Step 5)",              f"{FIG_DIR}/ablations.png"),
    ])

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
