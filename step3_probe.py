# step3_probe.py
# Maps to: LeWM paper (arXiv 2603.19312)
#   Section 4.2 — Linear probing evaluation of learned representations
#   Table 2     — Probing MSE and correlation for position

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from step2_train import (
    Encoder, EMBED_DIM, DATA_DIR, CKPT_DIR, FIG_DIR,
)

PROBE_ALPHA = 1.0
BATCH_SIZE  = 256


def load_encoder(path, embed_dim=EMBED_DIM):
    enc = Encoder(embed_dim)
    enc.load_state_dict(torch.load(path, map_location="cpu"))
    enc.eval()
    return enc


def encode_dataset(encoder, obs):
    """
    Encode all frames.

    Args:
        obs : (N, T, 1, H, W)
    Returns:
        latents : (N*T, embed_dim)
    """
    N, T = obs.shape[:2]
    flat = obs.reshape(-1, *obs.shape[2:])   # (N*T, 1, H, W)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(flat), BATCH_SIZE):
            batch = torch.tensor(flat[i : i + BATCH_SIZE], dtype=torch.float32)
            chunks.append(encoder(batch).numpy())
    return np.concatenate(chunks, axis=0)   # (N*T, d)


def probe(encoder, obs, states, tag):
    """
    Train a Ridge regression from latent vectors to (x, y) positions.

    Returns:
        y_test  : (n_test, 2)
        y_pred  : (n_test, 2)
        mse     : float
        r_mean  : float  mean Pearson r across x and y
    """
    latents = encode_dataset(encoder, obs)  # (N*T, d)
    labels  = states.reshape(-1, 2)         # (N*T, 2)

    n_train = int(0.8 * len(latents))
    X_tr, X_te = latents[:n_train], latents[n_train:]
    y_tr, y_te = labels[:n_train],  labels[n_train:]

    reg = Ridge(alpha=PROBE_ALPHA)
    reg.fit(X_tr, y_tr)
    y_pr = reg.predict(X_te)

    mse = float(np.mean((y_pr - y_te) ** 2))
    r_x = float(pearsonr(y_te[:, 0], y_pr[:, 0])[0])
    r_y = float(pearsonr(y_te[:, 1], y_pr[:, 1])[0])
    r_m = (r_x + r_y) / 2.0

    print(f"  [{tag:18s}]  MSE={mse:.4f}  r_x={r_x:.4f}  r_y={r_y:.4f}  r_mean={r_m:.4f}")
    return y_te, y_pr, mse, r_m


def main():
    obs    = np.load(f"{DATA_DIR}/obs_train.npy")
    states = np.load(f"{DATA_DIR}/states_train.npy")

    enc_sig   = load_encoder(f"{CKPT_DIR}/encoder_sigreg.pt")
    enc_nosig = load_encoder(f"{CKPT_DIR}/encoder_nosigreg.pt")

    print("Linear probing results:")
    y_te_s, y_pr_s, mse_s, r_s = probe(enc_sig,   obs, states, "With SIGReg")
    y_te_n, y_pr_n, mse_n, r_n = probe(enc_nosig, obs, states, "Without SIGReg")

    # Scatter plots: (SIGReg / No-SIGReg) × (x / y)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    combos = [
        (y_te_s, y_pr_s, "With SIGReg"),
        (y_te_n, y_pr_n, "Without SIGReg"),
    ]
    for col, (y_te, y_pr, label) in enumerate(combos):
        for row, coord in enumerate(["x", "y"]):
            ax = axes[row, col]
            ax.scatter(y_te[:, row], y_pr[:, row], s=4, alpha=0.3)
            lo = min(y_te[:, row].min(), y_pr[:, row].min())
            hi = max(y_te[:, row].max(), y_pr[:, row].max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_xlabel(f"True {coord}")
            ax.set_ylabel(f"Predicted {coord}")
            ax.set_title(f"{label} — {coord} position")

    plt.tight_layout()
    path = f"{FIG_DIR}/probe_scatter.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved scatter plot → {path}")

    np.savez(
        f"{DATA_DIR}/probe_metrics.npz",
        mse_sig=mse_s, r_sig=r_s,
        mse_nosig=mse_n, r_nosig=r_n,
    )

    return mse_s, r_s, mse_n, r_n


if __name__ == "__main__":
    main()
