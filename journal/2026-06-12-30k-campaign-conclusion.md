# 30k campaign conclusion: the 7k ceiling is architectural, not informational

One-line summary: five 25M-step training rounds spanning every demonstration
axis (depth 5.7k to 20.1k, diversity, perturbation robustness) all evaluate
between 6,765 and 7,123, while exploration proved 24,900-point play exists;
the bottleneck is the policy class, not the training data, and 30k requires
methods outside this project's scope.

## The five rounds (all 25M steps, scale, floor 0.15, 100-episode evals)

| round | demo axis tested | eval mean | clear | note |
|-------|------------------|-----------|-------|------|
| production s1 | script demos (5.7k) | 6,991 | 1.00 | Phase 4 |
| EI1 | self-demos (8.2k) | 7,117 | 0.99 | best mean |
| GE1 | explored depth (9.7k) | 6,765 | 0.93 | clones, brittle |
| GE2 | explored diversity (20.1k) | 7,020 | 0.99 | |
| GE3 | perturbation-robust demos | 7,123 | 1.00 | tightest std (742), same mean |

GE3's demos were explored UNDER deterministic sticky perturbation through
the exact training stack and replay bit-exactly (zero validation mismatches),
eliminating the robustness confound. The result: better consistency, same
depth. The policy converges to the same skill equilibrium every time:
perfect barrels-screen clearing plus 4 to 5 rivets per life.

## Why (assessment)

A feedforward CNN policy with 4-frame stacking must dodge flames that move
8 to 12 raw units per decision under 25 percent action-repeat noise. Deep
trajectories (ours reach 24,900) thread multi-second precision sequences;
cloning pressure (the BC floor) can hold a route, but the reactive skill of
surviving the rivets screen's flame traffic at round-2+ speeds appears to
exceed what this architecture extracts from any demonstration distribution
we could manufacture. Exploration is NOT the bottleneck (Go-Explore scaled
to 24,900 and kept rising); execution under stochasticity is.

## What 30k would take (recommendations, in rough order of promise)

1. Memory: recurrent or transformer policies that track flame trajectories
   across frames rather than reacting to 4-frame snapshots.
2. Interactive imitation (DAgger-style): an expert that can LABEL the
   policy's own visited states; our tree-search expert produces only
   open-loop trajectories. A state-conditional search expert (MCTS from
   restored states at every policy state) is feasible but is a compute
   project of its own.
3. Human demonstrations through a controller (distribution-matched,
   reactive, robust by nature).
4. Brute scale: 10 to 100x training compute with population-based methods.

## Where this leaves the project

- Best agent: GE3 (mean 7,123, clear 1.00, max 8,000) or EI1 (7,117, max
  8,100); both about 58x random and far past the only public hobbyist
  reference (about 300).
- The campaign produced reusable tooling: Go-Explore with exact replay
  under deterministic perturbation, self-demo harvesting, kickstarting with
  a floor. All gated, tested, and in the ledger.
- The negative result is well-evidenced and paper-strengthening: it cleanly
  separates discovery (solved by demonstrations) from robust execution
  (not solved by any demonstration variant at this scale).
