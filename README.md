# JEPA_CLPS
Final project for CLPS950, playing with JEPA models. 

A self-contained, CPU-only Python implementation of the key ideas from the
**LeWorldModel (LeWM)** paper:

> *"LeWM: Language-Enhanced World Model"* — arXiv 2603.19312

The project runs on a synthetic **dot-in-a-box** environment (32×32 greyscale
images, single dot bouncing around) and exercises each pillar of the paper:
representation learning, SIGReg anti-collapse regularisation, linear probing,
and violation-of-expectation surprise detection.

## The architecture of the files: 

step1_data.py file is the data generation block/first step of the pipeline: 
Creates a tiny simulated world: a white dot bouncing around inside a 32×32 pixel box. The dot moves with a bit of momentum, and bounces off the walls. We recorded 500 "episodes" (called trajectories) of 50 frames each — saving both the images and the true (x, y) position of the dot at every frame. Also made two sets of test episodes: 20 normal ones, and 20 where the dot secretly teleports to a random spot at frame 25. Those teleportation episodes are used later to test whether the model can detect something unexpected happening.                                                                            


step2_train.py is the training block, step2  
Trains two neural networks together: an Encoder (a small CNN that compresses each image into a compact vector called a latent) and a Predictor (a small MLP that takes the current latent + the action taken and predicts what the next latent should be).
Trained two versions: one with a regularizer called SIGReg (λ=0.1) and one without (λ=0). SIGReg is the key idea from the paper — it penalises the encoder if all its latent vectors start looking the same, forcing it to keep representing the world meaningfully rather than collapsing to a constant.                                             

step3_probe.py is the linear probing of above steps to test whether the latents actually encode useful information. Trained a dead-simple linear regression directly on top of the frozen encoder. It just asks "can I predict the dot's (x, y) position from the latent vector alone?" If the encoder has learned something real about the world, the line should fit well. If the encoder collapsed, you can't predict anything.                                          


step4_surprise.py asks "how wrong was the predictor's guess?" after running each trained model. In normal trajectories the prediction error should stay low. In teleportation trajectories, it should spike exactly at frame 25 because the dot jumped somewhere the model had no way to anticipate. This mirrors Figure 10 of the paper.                                    

step5_ablations.py is the "playing with the node" step that re-trains the model many times, varying one thing at a time: (a) the strength of SIGReg (λ = 0, 0.01, 0.05, 0.1, 0.2, 0.5) and (b) the size of the latent vector (4, 8, 16, 32, 64 dimensions). For each we measured probing R² to see which settings actually matter.                                                                                                                                                   

step6_pipeline.py runs all of the above in order and stitches all the output plots into a single summary figure.

in Run1Figures folder, you can find the outputs of the model I ran. 

## AI Usage
This project made substantial use of Claude Code (Anthropic), an AI coding agent, to implement the full LeWM-mini codebase. The AI was used as a primary implementation partner: the agent wrote all Python source files from scratch, verified them with smoke tests. The debugging for issues that came up was done collaboratively. My tasks were primarily the lit review and absorption of how the approach works. I gave AI detailed specification of what to build and which paper sections to map each feature to 

For me, using agentic coding wasn't new at all. I use it all the time in my labs and for person project. However, what was the most significant win for me is that AI coding allowed me to replicate something from scratch that even a year ago required an advance degree and long time spent on the project.  At the same time, this project allowed me to see why huge AI companies still hire teams of software engineers: It takes effort and time to understand what you are trying to create and to map part of your imagination with tools for its implementation. Finally, the big AI labs are still ahead of an individual coder not only because of research capabilities, and not even because of compute -- data is the biggest missing variable that large companies have effort, people, and tools to collect on an unimaginable scale. I really hope that world-building efforts will soon show cool products and wins as they are clearly building out the scope of their datasets. 

## Environment overview

| Component | Detail |
|---|---|
| State | (x, y) dot position |
| Action | (dx, dy) clipped to [−1, 1] |
| Dynamics | Euler step with 0.8 momentum; elastic wall bounce |
| Observation | 32×32 float32 greyscale image, dot radius = 3 px |
| Training set | 500 trajectories × 50 steps |
| Test sets | 20 normal + 20 teleportation (dot jumps at step 25) |

## Architecture of the Model

```
Observation (1×32×32)
       │
   Encoder
   ├─ Conv2d 1→16,  stride 2  (32→16)
   ├─ Conv2d 16→32, stride 2  (16→8)
   ├─ Conv2d 32→64, stride 2  (8→4)
   └─ Linear 1024 → d
       │
   Latent z_t  (d-dim)
       │ + action a_t (2-dim)
   Predictor (2-layer MLP, hidden 128)
       │
   Predicted z_{t+1}
```

**Loss** = MSE(z̃_{t+1}, z_{t+1})  +  λ · SIGReg(embeddings)

**SIGReg** (Appendix A): for each of M=64 random unit-vector projections,
standardise the projected values, evaluate the empirical characteristic function
on a 17-point grid in [0.2, 4.0], and measure the mean squared deviation from
the N(0,1) characteristic function (Epps-Pulley statistic). Aggregate by
averaging over projections. This forces the latent space toward a Gaussian
geometry and prevents representation collapse.

## Run

### Full pipeline (recommended)

```bash
python step6_pipeline.py
```

This runs all five steps in order, prints a summary table to stdout, and saves
`figures/summary.png` containing all result plots.

## Outputs

All random seeds are fixed to `42`. Running `python step6_pipeline.py` from a
clean repository will produce bit-identical outputs across runs on the same
hardware.

| File | Description |
|---|---|
| `data/obs_train.npy` | Training observations (500, 50, 1, 32, 32) |
| `data/actions_train.npy` | Training actions (500, 50, 2) |
| `data/states_train.npy` | Ground-truth (x, y) states (500, 50, 2) |
| `data/obs_test_normal.npy` | Normal test observations (20, 50, 1, 32, 32) |
| `data/obs_test_tele.npy` | Teleportation test observations (20, 50, 1, 32, 32) |
| `checkpoints/encoder_sigreg.pt` | Encoder trained with SIGReg (λ=0.1) |
| `checkpoints/encoder_nosigreg.pt` | Encoder trained without SIGReg (λ=0) |
| `figures/training_curves.png` | Loss curves for both training runs |
| `figures/probe_scatter.png` | Predicted vs. actual (x, y) scatter plots |
| `figures/surprise.png` | VoE surprise over time (Fig. 10 replica) |
| `figures/ablations.png` | λ and dim sweep (Figs. 15–16 replicas) |
| `figures/summary.png` | Unified 2×2 summary of all results |

## Expected results

With default settings (d=32, λ=0.1, 50 epochs, 500 trajectories):

- **With SIGReg**: probing R² ≈ 0.85–0.95; clear spike in surprise at step 25
  for teleportation trajectories.
- **Without SIGReg**: probing R² near zero (representation collapse); flat
  surprise curve.
- **Lambda sweep**: R² rises sharply from λ=0, peaks around λ=0.1.
- **Dim sweep**: R² improves with dimension up to ~32 then plateaus.

The ablation sweep uses 200 trajectories and 15 epochs per model to keep total
wall-clock time under 10 minutes on a modern laptop CPU.
