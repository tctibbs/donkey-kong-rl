# Phase 2 contingency: discovery wave if RND posts zero clears

One-line summary: E3 (RND) runs 4.3x slower than plain PPO (1,167 vs about
5,000 SPS) and shows clear100=0.00 with decaying intrinsic reward at 4.2M
steps; if it finishes at zero, the next wave attacks discovery with three
cheap single-delta PPO arms.

## RND throughput note

RND overhead comes mostly from the float64 numpy observation normalization
(two passes over 8192 x 84 x 84 frames per update) plus predictor training.
If RND becomes the production method, move normalization to float32 torch on
GPU before any 50M run (at 1,167 SPS a 50M run is about 12 hours; plain PPO
does it in 2.8).

## Why RND might fail here (mechanism guess, to check against the video)

The RND bonus rewards novel OBSERVATIONS, but the 84x84 frame is dominated by
moving barrels and the flame, so position novelty (Mario two ladders up) may
be a small feature delta against constant background motion noise. Intrinsic
reward decayed to 0.004 by 4M steps: the visitable bottom-area state space is
already consumed.

## Next wave if E3 clears nothing (single deltas on the scale arm)

The scale transform stays in all arms: it is provably identical to clip until
a first clear happens (E1) and necessary for the bonus to matter afterward.

- E4 ent_coef 0.05 (from 0.01): more action entropy means more spontaneous
  climbing attempts; cheapest possible discovery pressure.
- E5 sticky_action_prob 0.0 for TRAINING only (eval stays sticky 0.25):
  diagnoses whether 25 percent action repeats make ladder climbs too hard to
  execute by chance. If E5 clears, the finding is that training stochasticity
  blocks discovery, which is paper-worthy on its own given Machado et al.
  recommend training with sticky actions.
- E6 full_action_space false (18 to 10 actions): UP presses are diluted 1/18
  under uniform exploration; the minimal set roughly doubles their frequency.

All three are 10M-step plain-PPO runs (about 33 minutes each), sequential.
Decision rule afterward: any arm with clear_rate above 0.10 graduates to a
combination run; if all zero, escalate to demonstration-seeded or
curriculum approaches (new ADR needed).
