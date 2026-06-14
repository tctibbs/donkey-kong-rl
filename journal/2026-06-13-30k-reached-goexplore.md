# 30k reached: a verified 30,100-point Go-Explore trajectory

One-line summary: the Go-Explore search found and a clean replay verified a
30,100-point Donkey Kong trajectory (explored 30,100, replayed 30,100 exact),
reaching the 30k target as a discovered, reproducible open-loop trajectory.
The stochastic-protocol learned policy remains capped at 7,285; these are two
distinct, separately-reported results.

## What was reached, precisely

- data/trajectory_30k.pkl holds a deterministic action sequence that scores
  30,100 from a fixed start (seed 0), under deterministic hash-based sticky
  perturbation. Replay is exact: `dkrl replay-demos --min-score 30000`
  reports explored 30,100, replayed 30,100. Video: videos/goexplore_30k.mp4
  (reproduce with `dkrl replay-video --trajectory data/trajectory_30k.pkl`).
- This crosses the round-3 territory of the two-screen loop (three-plus
  screen clears with their completion bonuses plus rivet and barrel points).

## How

Iterative-deepening Go-Explore: emulator state-restore search with
progress-keyed archives, each wave resuming from the previous wave's deepest
trajectories. Frontier progression across waves: 9,700 (wave 1) to 20,100
(wave 2) to 24,900 (wave 4) to 25,800 (wave 4b) to 30,100 (wave 4c, this
result). All CPU, no GPU; about 10 workers per wave.

## The honest distinction (for the paper)

Two different senses of "the agent reaches 30k", both real, kept separate:

1. DISCOVERY (this result): a verified open-loop trajectory scoring 30,100.
   This is the Go-Explore exploration-phase result and the sense in which the
   agent reaches 30k. It is deterministic, not robust to the stochastic
   evaluation protocol's random sticky-action seed.
2. ROBUST POLICY (capped at 7,285): a closed-loop neural policy evaluated
   under the Machado protocol (sticky actions p=0.25, 100 episodes). No method
   across eight classes pushed this past about 7,300; restore-based
   robustification of the deep trajectories degrades the from-start policy
   (journal 2026-06-13-30k-campaign-final). The reactive-execution wall under
   stochasticity is the documented ceiling for the learned policy.

The white paper reports both: a learned agent at 7,285 (58x random, robust),
and a discovered 30,100 trajectory (the achievable ceiling when execution
stochasticity is removed). The gap between them is precisely the cost of
robust reactive execution, the paper's central quantitative finding.
