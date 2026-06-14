# 30k campaign status: a robust 7k transfer ceiling, and two replay bugs fixed

One-line summary: four 25M-step training rounds with increasingly deep
demonstrations all evaluate between 6,765 and 7,117 (the policy executes
screen 1 plus partial rivets and nothing deeper sticks), while the
exploration frontier reached 24,900; the remaining lever is sticky-robust
demonstrations, now collectable after fixing two replay-determinism bugs.

## Training rounds (all 25M steps, scale transform, floor 0.15, 100-ep evals)

| round | demos | eval mean | clear | max |
|-------|-------|-----------|-------|-----|
| production (Phase 4 s1) | two-screen script + BC | 6,991 | 1.00 | 8,100 |
| EI1 | + agent's own top episodes (8.0-8.2k) | 7,117 | 0.99 | 8,100 |
| GE1 | top-25 explored trajectories (9.7k, near-clones) | 6,765 | 0.93 | 8,100 |
| GE2 | 50 DIVERSE explored trajectories (10-20.1k) | 7,020 | 0.99 | 7,900 |

Demo depth went 5.7k -> 20.1k with no eval gain. Diagnosis: non-sticky demo
trajectories cannot be executed under sticky evaluation; BC teaches the
route but not the reactive dodging that survives perturbation. The policy
distills everything to the same robust skill set (screen-1 clear plus about
4-5 rivets per life) regardless of how deep the demos go.

## Exploration frontier (Go-Explore, state restore)

9,700 (wave 1) -> 20,100 (wave 2, resume deepening) -> 24,900 (wave 4),
decelerating with depth. Frontier growth is not the bottleneck; transfer is.

## Two replay-determinism bugs (both fixed, validated by round-trips)

1. cloneState excludes ALE's sticky RNG, so sticky trajectories assembled
   across restores replayed at about 300 instead of 18,000. Fixed with
   cloneSystemState/restoreSystemState.
2. Exploring at ALE frameskip 4 consumes one sticky draw per agent action,
   while the training/replay stack (frameskip 1 under AtariPreprocessing)
   consumes four; sequences diverged about 20 percent in. Fixed by exploring
   through the identical preprocessing stack. Round-trip now exact (explored
   3,600 = replayed 3,600 under sticky 0.25).

## Current bet and honest outlook

Wave 3c (8 workers) explores WITH sticky 0.25 through the exact training
stack, so its trajectories are robust-by-construction and replay exactly.
If BC plus the floor on these transfers materially better, iterate deeper.
If GE3 also lands near 7k, the conclusion is that trajectory cloning cannot
push past the reactive-skill ceiling under sticky evaluation at this model
scale, and 30k requires a different method class (recurrent policies,
DAgger-style interactive experts, or much larger compute). That conclusion
would itself be a paper-worthy negative result, documented with full
evidence either way.
