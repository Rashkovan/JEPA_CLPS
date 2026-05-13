# step1_data.py
# Maps to: LeWM paper (arXiv 2603.19312)
#   Section 4 — Experimental Setup (synthetic environment description)
#   Appendix B — Environment Details (dot-in-box physics, rendering)

import os
import numpy as np

SEED = 42
N_TRAJ = 500
T_LEN = 50
IMG_SIZE = 32
DOT_RADIUS = 3
N_TEST = 20
TELEPORT_STEP = 25
DATA_DIR = "data"


def render_frame(x, y, size=IMG_SIZE, dot_r=DOT_RADIUS):
    """Render a grayscale frame with a filled dot of radius dot_r at (x, y)."""
    img = np.zeros((size, size), dtype=np.float32)
    xi, yi = int(round(x)), int(round(y))

    # Rasterizes a filled circle using a bounding-box scan.
    # Every pixel within dot_r distance of (xi, yi) is set to 1.0 (white).
    # Pixels outside the image bounds are silently skipped.

    for dx in range(-dot_r, dot_r + 1):
        for dy in range(-dot_r, dot_r + 1):
            if dx * dx + dy * dy <= dot_r * dot_r:
                px, py = xi + dx, yi + dy
                if 0 <= px < size and 0 <= py < size:
                    img[py, px] = 1.0
    return img


def step_env(x, y, vx, vy, action, size=IMG_SIZE, dot_r=DOT_RADIUS):
    """Apply action with momentum, bounce off walls."""
  
    # Action clipping: raw actions are clamped to [-1, 1]
    # before being blended into velocity, preventing runaway acceleration.

    dx, dy = np.clip(action[0], -1.0, 1.0), np.clip(action[1], -1.0, 1.0)

    # Momentum / exponential smoothing.
    # New velocity = 80% old + 20% action input, making motion feel inertial
    # rather than instantaneous. Controlled by the 0.8 / 0.2 coefficients.
    
    vx = 0.8 * vx + 0.2 * dx
    vy = 0.8 * vy + 0.2 * dy

    # Wall margin keeps the dot's centre at least dot_r pixels
    # from each edge, so the dot never renders partially outside the frame.

    margin = float(dot_r)
    lo, hi = margin, float(size) - 1.0 - margin

    x_new = x + vx
    y_new = y + vy

    # Elastic (mirror) wall bouncing.
    # When the dot would exceed a boundary, its position is clamped and the
    # corresponding velocity component is reversed, simulating a perfect bounce.

    if x_new < lo:
        x_new = lo
        vx = abs(vx)
    elif x_new > hi:
        x_new = hi
        vx = -abs(vx)

    if y_new < lo:
        y_new = lo
        vy = abs(vy)
    elif y_new > hi:
        y_new = hi
        vy = -abs(vy)

    return x_new, y_new, vx, vy


def generate_trajectory(rng, teleport_step=None, T=T_LEN):
    """
    Generate one trajectory of length T.
    If teleport_step is set, the dot jumps to a random position at that step.

    Returns:
        obs     — (T, 1, IMG_SIZE, IMG_SIZE)  float32
        actions — (T, 2)                       float32
        states  — (T, 2)  (x, y) ground truth
    """
    margin = float(DOT_RADIUS)
    lo, hi = margin, float(IMG_SIZE) - 1.0 - margin

    # Random initialisation within the valid margin area,
    # ensuring the dot starts fully visible and with a non-zero random velocity.
    
    x = rng.uniform(lo, hi)
    y = rng.uniform(lo, hi)
    vx = rng.uniform(-1.0, 1.0)
    vy = rng.uniform(-1.0, 1.0)

    obs_list, act_list, state_list = [], [], []

    for t in range(T):

        # Teleportation injection.
        # At the designated step the dot is moved to a random new position
        # without resetting velocity, creating an abrupt, unpredictable jump.
        # This is used exclusively in the out-of-distribution test set to probe
        # how well a world model handles sudden discontinuities.
        
        if teleport_step is not None and t == teleport_step:
            x = rng.uniform(lo, hi)
            y = rng.uniform(lo, hi)

        frame = render_frame(x, y)
        obs_list.append(frame[np.newaxis])   # (1, H, W)
        state_list.append([x, y])

        # Action is sampled uniformly at random each step,
        # producing an unbiased, policy-agnostic training distribution.

        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        act_list.append(action)

        x, y, vx, vy = step_env(x, y, vx, vy, action)

    obs = np.stack(obs_list).astype(np.float32)      # (T, 1, H, W)
    actions = np.stack(act_list).astype(np.float32)  # (T, 2)
    states = np.array(state_list, dtype=np.float32)  # (T, 2)
    return obs, actions, states


def _collect(rng, n, teleport_step=None):
    all_obs, all_act, all_states = [], [], []
    for _ in range(n):
        obs, act, states = generate_trajectory(rng, teleport_step=teleport_step)
        all_obs.append(obs)
        all_act.append(act)
        all_states.append(states)
    return (
        np.stack(all_obs),     # (n, T, 1, H, W)
        np.stack(all_act),     # (n, T, 2)
        np.stack(all_states),  # (n, T, 2)
    )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Fixed global seed ensures the entire dataset (train + both
    # test splits) is fully reproducible from a single RNG state.

    rng = np.random.default_rng(SEED)

    # Three distinct dataset splits are written to disk as .npy files:
    #   1. Training set      — 500 normal trajectories for model training.
    #   2. Normal test set   — 20 held-out trajectories, same distribution as train.
    #   3. Teleportation set — 20 trajectories with a forced jump at step 25,
    #                          used to evaluate robustness to discontinuities.
    
    print("Generating training trajectories...")
    obs_tr, act_tr, states_tr = _collect(rng, N_TRAJ)
    np.save(f"{DATA_DIR}/obs_train.npy", obs_tr)
    np.save(f"{DATA_DIR}/actions_train.npy", act_tr)
    np.save(f"{DATA_DIR}/states_train.npy", states_tr)
    print(f"  obs_train     : {obs_tr.shape}")
    print(f"  actions_train : {act_tr.shape}")

    print("Generating normal test trajectories...")
    obs_tn, act_tn, states_tn = _collect(rng, N_TEST)
    np.save(f"{DATA_DIR}/obs_test_normal.npy", obs_tn)
    np.save(f"{DATA_DIR}/actions_test_normal.npy", act_tn)
    np.save(f"{DATA_DIR}/states_test_normal.npy", states_tn)
    print(f"  obs_test_normal : {obs_tn.shape}")

    print("Generating teleportation test trajectories...")
    obs_tt, act_tt, states_tt = _collect(rng, N_TEST, teleport_step=TELEPORT_STEP)
    np.save(f"{DATA_DIR}/obs_test_tele.npy", obs_tt)
    np.save(f"{DATA_DIR}/actions_test_tele.npy", act_tt)
    np.save(f"{DATA_DIR}/states_test_tele.npy", states_tt)
    print(f"  obs_test_tele : {obs_tt.shape}")

    print("Data generation complete.")


if __name__ == "__main__":
    main()
