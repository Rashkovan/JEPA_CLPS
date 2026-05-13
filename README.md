# JEPA_CLPS
Final project for CLPS950, playing with JEPA models. 

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

