# step5_ablations.py
# Maps to: LeWM paper (arXiv 2603.19312)
#   Section 4.3 — Ablation studies
#   Figure 15   — Probing R² vs SIGReg strength λ
#   Figure 16   — Probing R² vs embedding dimension
#
# Uses a reduced dataset (200 trajectories) and fewer epochs (15) to complete
# the full sweep in < 10 minutes on a modern laptop CPU.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from step2_train import (
    Encoder, Predictor, TransitionDataset, compute_sigreg,
    _make_proj_vectors,
    BATCH_SIZE, LR, DATA_DIR, FIG_DIR, SEED,
    N_QUAD, T_MIN, T_MAX,
)

ABLATION_EPOCHS = 15
ABLATION_N_TRAJ = 200

LAMBDA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
DIM_VALUES    = [4, 8, 16, 32, 64]


def _set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def train_and_probe(lam, embed_dim, obs, actions, states):
    """
    Train one model configuration and return probing R².
    """
    _set_seed()

    dataset = TransitionDataset(obs, actions)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    encoder   = Encoder(embed_dim)
    predictor = Predictor(embed_dim)
    params    = list(encoder.parameters()) + list(predictor.parameters())
    opt       = torch.optim.Adam(params, lr=LR)

    proj_vectors = _make_proj_vectors(embed_dim)
    t_grid       = torch.linspace(T_MIN, T_MAX, N_QUAD)

    encoder.train()
    predictor.train()

    for _ in range(ABLATION_EPOCHS):
        for obs_t, act_t, obs_t1 in loader:
            opt.zero_grad()
            z_t  = encoder(obs_t)
            z_t1 = encoder(obs_t1)
            z_pr = predictor(z_t, act_t)
            loss = nn.functional.mse_loss(z_pr, z_t1)
            if lam > 0:
                loss = loss + lam * compute_sigreg(
                    torch.cat([z_t, z_t1], dim=0), proj_vectors, t_grid
                )
            loss.backward()
            opt.step()

    # Linear probe on full obs/states
    encoder.eval()
    N, T = obs.shape[:2]
    flat   = obs.reshape(-1, *obs.shape[2:])
    labels = states.reshape(-1, 2)

    chunks = []
    with torch.no_grad():
        for i in range(0, len(flat), 256):
            batch = torch.tensor(flat[i : i + 256], dtype=torch.float32)
            chunks.append(encoder(batch).numpy())
    latents = np.concatenate(chunks, axis=0)

    n_tr = int(0.8 * len(latents))
    reg  = Ridge(alpha=1.0)
    reg.fit(latents[:n_tr], labels[:n_tr])
    y_pr = reg.predict(latents[n_tr:])
    return float(r2_score(labels[n_tr:], y_pr))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    obs_all    = np.load(f"{DATA_DIR}/obs_train.npy")[:ABLATION_N_TRAJ]
    act_all    = np.load(f"{DATA_DIR}/actions_train.npy")[:ABLATION_N_TRAJ]
    states_all = np.load(f"{DATA_DIR}/states_train.npy")[:ABLATION_N_TRAJ]

    # --- Lambda sweep (Figure 15) ---
    print(f"Lambda sweep over {LAMBDA_VALUES} …")
    lambda_r2 = []
    for lam in LAMBDA_VALUES:
        r2 = train_and_probe(lam, 32, obs_all, act_all, states_all)
        lambda_r2.append(r2)
        print(f"  λ={lam:<5}  R²={r2:.4f}")

    # --- Embedding-dim sweep (Figure 16) ---
    print(f"Embedding-dim sweep over {DIM_VALUES} …")
    dim_r2 = []
    for dim in DIM_VALUES:
        r2 = train_and_probe(0.1, dim, obs_all, act_all, states_all)
        dim_r2.append(r2)
        print(f"  dim={dim:<4}  R²={r2:.4f}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(LAMBDA_VALUES, lambda_r2, "o-")
    axes[0].set_xscale("symlog", linthresh=0.005)
    axes[0].set_xlabel("λ (SIGReg strength)")
    axes[0].set_ylabel("Probing R²")
    axes[0].set_title("Effect of λ on probing performance  (Fig. 15)")

    axes[1].plot(DIM_VALUES, dim_r2, "o-")
    axes[1].set_xlabel("Embedding dimension")
    axes[1].set_ylabel("Probing R²")
    axes[1].set_title("Effect of embedding dim  (Fig. 16)")

    plt.tight_layout()
    path = f"{FIG_DIR}/ablations.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved ablation plots → {path}")

    np.savez(
        f"{DATA_DIR}/ablation_metrics.npz",
        lambda_values=LAMBDA_VALUES,
        lambda_r2=lambda_r2,
        dim_values=DIM_VALUES,
        dim_r2=dim_r2,
    )

    return lambda_r2, dim_r2


if __name__ == "__main__":
    main()
