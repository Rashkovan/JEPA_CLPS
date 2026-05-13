# step4_surprise.py
# Maps to: LeWM paper (arXiv 2603.19312)
#   Section 5.3 — Violation-of-Expectation (VoE) experiments
#   Figure 10   — Prediction surprise over time for normal vs teleportation
#                 trajectories (spike visible at the teleportation step)

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step2_train import (
    Encoder, Predictor, EMBED_DIM, DATA_DIR, CKPT_DIR, FIG_DIR,
)

TELEPORT_STEP = 25   # 0-indexed frame where the dot jumps


def compute_surprise(encoder, predictor, obs, actions):
    """
    Run the predictor autoregressively and record per-step MSE.

    Args:
        obs     : (N, T, 1, H, W)
        actions : (N, T, 2)
    Returns:
        surprises : (N, T-1)  MSE(z_pred_{t+1}, z_{t+1}) for each step
    """
    N, T = obs.shape[:2]
    encoder.eval()
    predictor.eval()
    surprises = np.zeros((N, T - 1), dtype=np.float32)

    with torch.no_grad():
        for n in range(N):
            obs_n = torch.tensor(obs[n],     dtype=torch.float32)  # (T, 1, H, W)
            act_n = torch.tensor(actions[n], dtype=torch.float32)  # (T, 2)
            z_all = encoder(obs_n)                                  # (T, d)
            for t in range(T - 1):
                z_pred = predictor(z_all[t : t + 1], act_n[t : t + 1])  # (1, d)
                diff   = z_pred - z_all[t + 1 : t + 2]
                surprises[n, t] = float(torch.mean(diff ** 2))

    return surprises


def main():
    enc  = Encoder(EMBED_DIM)
    pred = Predictor(EMBED_DIM)
    enc.load_state_dict( torch.load(f"{CKPT_DIR}/encoder_sigreg.pt",   map_location="cpu"))
    pred.load_state_dict(torch.load(f"{CKPT_DIR}/predictor_sigreg.pt", map_location="cpu"))

    obs_norm = np.load(f"{DATA_DIR}/obs_test_normal.npy")
    act_norm = np.load(f"{DATA_DIR}/actions_test_normal.npy")
    obs_tele = np.load(f"{DATA_DIR}/obs_test_tele.npy")
    act_tele = np.load(f"{DATA_DIR}/actions_test_tele.npy")

    print("Computing surprise on normal trajectories …")
    surp_norm = compute_surprise(enc, pred, obs_norm, act_norm)   # (20, 49)
    print("Computing surprise on teleportation trajectories …")
    surp_tele = compute_surprise(enc, pred, obs_tele, act_tele)   # (20, 49)

    T_steps = surp_norm.shape[1]
    steps   = np.arange(T_steps)

    mu_n, sd_n = surp_norm.mean(0), surp_norm.std(0)
    mu_t, sd_t = surp_tele.mean(0), surp_tele.std(0)

    # Surprise ratio at the teleportation step (index TELEPORT_STEP - 1)
    spike_idx     = TELEPORT_STEP - 1          # prediction from step 24 → 25
    surprise_ratio = float(mu_t[spike_idx] / (mu_n[spike_idx] + 1e-8))
    print(f"Surprise ratio at step {TELEPORT_STEP}: {surprise_ratio:.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, mu_n, "b-", label="Normal")
    ax.fill_between(steps, mu_n - sd_n, mu_n + sd_n, alpha=0.25, color="b")
    ax.plot(steps, mu_t, "r-", label="Teleportation")
    ax.fill_between(steps, mu_t - sd_t, mu_t + sd_t, alpha=0.25, color="r")
    ax.axvline(x=spike_idx, color="k", linestyle="--", alpha=0.7,
               label=f"Teleportation at step {TELEPORT_STEP}")
    ax.set_xlabel("Time step  (t → t+1 prediction)")
    ax.set_ylabel("Prediction surprise  (MSE in embedding space)")
    ax.set_title("Violation-of-Expectation Surprise Signal  (mirrors Fig. 10)")
    ax.legend()
    plt.tight_layout()
    path = f"{FIG_DIR}/surprise.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved surprise plot → {path}")

    np.savez(
        f"{DATA_DIR}/surprise_metrics.npz",
        mean_normal=mu_n, std_normal=sd_n,
        mean_tele=mu_t,   std_tele=sd_t,
        surprise_ratio=surprise_ratio,
    )

    return float(surprise_ratio)


if __name__ == "__main__":
    main()
