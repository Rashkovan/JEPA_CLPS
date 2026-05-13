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

    # Only figures that actually exist on disk are included.
    # This makes the function tolerant of partial pipeline runs — if an earlier
    # step failed or was skipped, the summary is still assembled from whatever
    # is available rather than crashing.

    images, titles = [], []
    for title, path in labels_paths:
        if os.path.exists(path):
            images.append(plt.imread(path))
            titles.append(title)

    if not images:
        print("No figures found to assemble.")
        return

    # The grid dimensions are computed dynamically from the number
    # of available figures rather than being hard-coded as 2×2. nrows is derived
    # via ceiling division so an odd number of panels still fills correctly.

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
    # Any unused subplot cells (when n is odd) are explicitly hidden.
    # Without this, matplotlib leaves empty axes with visible borders in the
    # output figure.

        ax.axis("off")

    fig.suptitle("LeWM-mini: Full Pipeline Summary", fontsize=16,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    out = f"{FIG_DIR}/summary.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved summary figure → {out}")


def print_table(mse_s, r_s, mse_n, r_n, surprise_ratio):

    # The summary table is printed to stdout rather than saved to a
    # file. It exists as a human-readable sanity check at the end of the run,
    # presenting the three key metrics side-by-side so the reader can immediately
    # see whether SIGReg improved probing and whether the VoE spike is meaningful.
    # Surprise ratio is shown only for the SIGReg model because step4 evaluates
    # only that checkpoint; "N/A" makes the asymmetry explicit rather than silent.
    
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

    # Steps are executed in strict sequential order by calling each
    # module's main() directly. There is no parallelism or dependency graph — each
    # step writes its outputs (data files, checkpoints, figures) to disk and the
    # next step reads them. This makes the pipeline easy to debug: any step can be
    # re-run in isolation by calling its module directly without touching the others.
    
    print("\n[Step 1] Generating data …")
    step1_data.main()

    print("\n[Step 2] Training models …")
    step2_train.main()

    # Return values from steps 3 and 4 are captured directly rather
    # than reloading them from the .npz files written to disk. This avoids a
    # redundant file read and keeps the summary table consistent with the values
    # that were just computed, even if a previous run left stale metrics on disk.
    
    print("\n[Step 3] Linear probing …")
    mse_s, r_s, mse_n, r_n = step3_probe.main()

    print("\n[Step 4] Surprise detection …")
    surprise_ratio = step4_surprise.main()

    print("\n[Step 5] Ablations …")
    step5_ablations.main()

    print_table(mse_s, r_s, mse_n, r_n, surprise_ratio)

    # The figure paths passed to assemble_summary are listed in the
    # same order as the pipeline steps, so the assembled grid reads left-to-right,
    # top-to-bottom in execution order. This makes the summary figure a visual
    # narrative of the full paper rather than an arbitrary collection of panels.
    
    assemble_summary([
        ("Training curves  (Step 2)",       f"{FIG_DIR}/training_curves.png"),
        ("Linear probing  (Step 3)",         f"{FIG_DIR}/probe_scatter.png"),
        ("VoE surprise  (Step 4)",           f"{FIG_DIR}/surprise.png"),
        ("Ablations  (Step 5)",              f"{FIG_DIR}/ablations.png"),
    ])

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
