# E9: kickstarting produces the project's first screen clears

One-line summary: BC warm-start plus an annealed BC auxiliary loss in the
PPO objective (kickstarting) produced the first training clears at 0.5M
steps, with the in-training clear rate accelerating 6 to 20 percent between
0.5M and 0.74M steps; the loop E8 lacked is closed.

## Why E8 failed and E9 works

E8 (BC init only, ledger row ppo_bc_10m_s1_20260611_075540): the cloned
climbing policy died to barrels under sticky actions before any clear paid
out. Climbing is zero-reward until the lump bonus, so on-policy gradients
contained nothing distinguishing a good climb from a bad one, while farming
gradients (+1 per jump) were dense. The prior washed out and the run ended
with train_clears=0, eval mean 4,075, clear_rate 0.00, the same farming
basin as every other arm.

E9 adds two things (config ppo_bc_kickstart.yaml, hash d81e70efee1a, git
18a68fd3f192):

1. Sticky-matched demos (data/demos_screen1_sticky.npz, 25 episodes, 6,826
   pairs, collected at sticky 0.25 where the scripted climber still clears
   12/15). The BC prior matches the training action-noise distribution.
2. An auxiliary BC cross-entropy term added to the PPO loss, coefficient 1.0
   annealed linearly to 0 over 400 updates (about 3.3M steps). This holds
   the policy near climbing behavior long enough for full clears to occur
   on-policy; with the scale transform a clear is worth about +42 versus +1
   per barrel jump, so once experienced, the environment reward sustains the
   behavior on its own.

## Observed (run log, W&B)

- update 60 (0.5M steps): clear100 0.06, the first nonzero in any run.
- update 70: 0.09; update 80: 0.13; update 90: 0.20, return100 933 and
  rising with each clear paying about 4,200 raw.

Full eval numbers will land in the ledger when the run finishes.
