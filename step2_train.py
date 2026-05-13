# step2_train.py
# Maps to: LeWM paper (arXiv 2603.19312)
#   Section 3   — Model Architecture (Encoder, Predictor)
#   Section 3.2 — SIGReg objective
#   Appendix A  — SIGReg implementation (Epps-Pulley via characteristic function,
#                 M=64 random projections, 17-point quadrature in [0.2, 4.0])
#   Section 4   — Training protocol (Adam, lr=1e-3, MSE prediction loss)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
EMBED_DIM = 32
BATCH_SIZE = 64
N_EPOCHS = 50
LR = 1e-3
LAMBDA_SIGREG = 0.1

# SIGReg hyperparameters (Appendix A)
M_PROJ = 64       # random projection directions
N_QUAD = 17       # quadrature points
T_MIN = 0.2       # quadrature grid start
T_MAX = 4.0       # quadrature grid end

DATA_DIR = "data"
CKPT_DIR = "checkpoints"
FIG_DIR = "figures"


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """3-layer CNN followed by a linear projection to embed_dim."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()

    # Three strided convolutions progressively halve spatial
        # resolution (32→16→8→4), doubling channels each time (1→16→32→64).
        # Strided convolutions are used instead of pooling to keep the
        # operation learnable. 
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 32→16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16→8
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8→4
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 4 * 4, embed_dim)

    def forward(self, x):
        # x: (B, 1, 32, 32)
        h = self.cnn(x).flatten(1)
        return self.fc(h)


class Predictor(nn.Module):
    """2-layer MLP: (embedding, action) → next embedding."""
    
    def __init__(self, embed_dim=EMBED_DIM, action_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
        # The predictor concatenates the current embedding and
        # action before passing them through the MLP. This makes the transition
        # model explicitly action-conditioned rather than treating action as a
        # separate modulation signal.
            nn.Linear(embed_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TransitionDataset(Dataset):
    """Pairs of (obs_t, action_t, obs_{t+1}) from trajectory data."""

    # Trajectories are sliced into overlapping (obs_t, obs_{t+1})
        # pairs by indexing [:,:-1] and [:,1:] then flattening the first two
        # dimensions. This converts N trajectories of length T into N*(T-1)
        # independent transition samples, maximising data efficiency.
    
    def __init__(self, obs, actions):
        # obs: (N, T, 1, H, W)  actions: (N, T, 2)
        rest = obs.shape[2:]
        self.obs_t  = torch.tensor(obs[:, :-1].reshape(-1, *rest),       dtype=torch.float32)
        self.obs_t1 = torch.tensor(obs[:, 1:].reshape(-1, *rest),        dtype=torch.float32)
        self.acts   = torch.tensor(actions[:, :-1].reshape(-1, 2),       dtype=torch.float32)

    def __len__(self):
        return len(self.obs_t)

    def __getitem__(self, idx):
        return self.obs_t[idx], self.acts[idx], self.obs_t1[idx]


# ---------------------------------------------------------------------------
# SIGReg (Appendix A)
# ---------------------------------------------------------------------------

def compute_sigreg(embeddings, proj_vectors, t_grid):
    """
    Epps-Pulley Gaussianity test aggregated over random projections.

    For each projection direction v_m, the 1-D Epps-Pulley statistic is:
        T_m = mean_k |ECF(t_k) - phi_N(t_k)|^2
    where ECF is the empirical characteristic function of standardised
    projections and phi_N(t) = exp(-t^2/2) is the N(0,1) CF.
    SIGReg = mean_m T_m.

    Args:
        embeddings   : (B, d)  current batch of latent vectors
        proj_vectors : (M, d)  fixed random unit vectors
        t_grid       : (K,)    quadrature evaluation points
    Returns:
        scalar loss (minimise → push embeddings toward Gaussianity)
    """
    # Project: (B, M)
    proj = embeddings @ proj_vectors.T

    # Standardise each projection to zero mean, unit variance
    mu  = proj.mean(dim=0, keepdim=True)               # (1, M)
    std = proj.std(dim=0, keepdim=True).clamp(min=1e-6) # (1, M)
    z   = (proj - mu) / std                             # (B, M)

    # Empirical CF evaluated at each t_k: (B, M, K) → average over B
    tz       = z.unsqueeze(2) * t_grid.view(1, 1, -1)  # (B, M, K)
    ecf_real = torch.cos(tz).mean(dim=0)                # (M, K)
    ecf_imag = torch.sin(tz).mean(dim=0)                # (M, K)

    # Target CF: phi_N(t) = exp(-t^2/2), real-valued
    phi = torch.exp(-0.5 * t_grid ** 2)                 # (K,)

    # The target CF phi_N(t) = exp(-t^2/2) is the CF of N(0,1).
    # The loss is |ECF - phi_N|^2, summing squared deviations of both real
    # and imaginary parts. Minimising this drives the latent distribution
    # toward Gaussianity without requiring an explicit density estimate. 
    
    # |ECF - phi_N|^2 averaged over projections and quadrature points
    diff_sq = (ecf_real - phi.unsqueeze(0)) ** 2 + ecf_imag ** 2  # (M, K)
    return diff_sq.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_proj_vectors(embed_dim, seed=SEED):
    
    # Projection vectors are fixed at initialisation (not learned)
    # and normalised to unit length. Using a separate seeded generator keeps
    # them reproducible and independent of the main training RNG state.

    gen = torch.Generator()
    gen.manual_seed(seed)
    raw = torch.randn(M_PROJ, embed_dim, generator=gen)
    return raw / raw.norm(dim=1, keepdim=True)


def train_model(lam, embed_dim=EMBED_DIM, n_epochs=N_EPOCHS,
                data_dir=DATA_DIR, verbose=True):
    """
    Train encoder + predictor with loss = MSE + lam * SIGReg.

    Args:
        lam        : SIGReg regularisation strength
        embed_dim  : latent dimension
        n_epochs   : training epochs
    Returns:
        encoder, predictor, losses (list of per-epoch average loss)
    """
    set_seed()

    obs     = np.load(f"{data_dir}/obs_train.npy")
    actions = np.load(f"{data_dir}/actions_train.npy")

    dataset = TransitionDataset(obs, actions)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0)

    encoder   = Encoder(embed_dim)
    predictor = Predictor(embed_dim)
    params    = list(encoder.parameters()) + list(predictor.parameters())
    opt       = optim.Adam(params, lr=LR)

    proj_vectors = _make_proj_vectors(embed_dim)
    t_grid       = torch.linspace(T_MIN, T_MAX, N_QUAD)

    losses = []
    for epoch in range(n_epochs):
        total, n_batches = 0.0, 0
        for obs_t, act_t, obs_t1 in loader:
            opt.zero_grad()

            z_t  = encoder(obs_t)
            z_t1 = encoder(obs_t1)
            z_pr = predictor(z_t, act_t)

            # The primary loss is MSE between the predicted next
            # embedding z_pr and the encoded true next observation z_t1.
            # This is a latent-space prediction loss — the model never decodes
            # back to pixels, avoiding the cost and blur of reconstruction.

            loss = nn.functional.mse_loss(z_pr, z_t1)
           
            #SIGReg is added only when lam > 0, making it
            # trivial to run an ablation (lam=0) with identical code.
            # Both z_t and z_t1 from the current batch are pooled together
            # before computing the statistic, doubling the effective sample
            # size for the Gaussianity test at no extra forward-pass cost.
        
            if lam > 0:
                all_emb = torch.cat([z_t, z_t1], dim=0)
                loss = loss + lam * compute_sigreg(all_emb, proj_vectors, t_grid)

            loss.backward()
            opt.step()

            total    += loss.item()
            n_batches += 1

        avg = total / n_batches
        losses.append(avg)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1:3d}/{n_epochs}  loss={avg:.5f}")

    return encoder, predictor, losses


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Two full training runs are executed back-to-back — one with
    # SIGReg (lam=0.1) and one without (lam=0) — and both checkpoints are saved.
    # This produces the paired models needed for the ablation study in Section 4.

    print("Training WITH SIGReg (λ=0.1) …")
    enc_s, pred_s, losses_s = train_model(LAMBDA_SIGREG)
    torch.save(enc_s.state_dict(),  f"{CKPT_DIR}/encoder_sigreg.pt")
    torch.save(pred_s.state_dict(), f"{CKPT_DIR}/predictor_sigreg.pt")

    print("Training WITHOUT SIGReg (λ=0) …")
    enc_n, pred_n, losses_n = train_model(0.0)
    torch.save(enc_n.state_dict(),  f"{CKPT_DIR}/encoder_nosigreg.pt")
    torch.save(pred_n.state_dict(), f"{CKPT_DIR}/predictor_nosigreg.pt")

    # Save loss curves: 
    # gives an immediate visual check that SIGReg does not
    # destabilise training and lets the reader eyeball convergence speed.
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses_s, label="With SIGReg (λ=0.1)")
    ax.plot(losses_n, label="Without SIGReg (λ=0)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    plt.tight_layout()
    path = f"{FIG_DIR}/training_curves.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved training curves → {path}")


if __name__ == "__main__":
    main()
