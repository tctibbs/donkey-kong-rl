# DAgger round 1: myopic search expert regresses the policy (5,990)

One-line summary: search-based DAgger with a random-shooting expert regressed
the policy to mean 5,990 (from the 7,123 GE3 base), because the myopic expert
is not better than the already-strong policy, so its labels inject noise
rather than correction.

## What happened

- Collected about 1,100 states the GE3 policy visits in the sticky env across
  10 parallel workers, relabeled each with the random-shooting + height-
  potential search expert (ledger kind=dagger_collect).
- Aggregated with the robust demos (97,328 pairs), BC-warm-started to 0.893
  action accuracy, kickstarted 15M steps (ppo_dagger_r1_15m).
- Evaluation (100 protocol episodes): mean 5,990, std 1,907, median 6,400,
  max 8,000, clear_rate 0.92. WORSE than the 7,123 base and the 7,285
  recurrent best.

## Diagnosis

DAgger improves a policy only when the expert is better than the policy at
the visited states. The random-shooting expert (24 depth-biased rollouts x
60-step horizon, scored by reward plus a height/board progress potential)
produces diverse, sensible-looking labels (climb and jump actions) but is
MYOPIC: a 60-step horizon cannot see a screen clear, so the height potential
is a local proxy that sometimes prefers climbing when the policy's learned
choice (dodge, reposition) is actually better. Relabeling GE3 toward this
expert pulls it off its own stronger behavior. ADR 0005 explicitly flagged
this risk ("the search expert from a single state may itself be myopic").

## Fix for round 2

Replace random shooting with a proper mini Go-Explore from each state: an
archive-based search (depth-keyed cells, action-burst expansion) that finds
genuinely deep continuations and returns the first action of the best one.
This is a real expert (it discovers high-scoring deep play, as the global
Go-Explore did), so its labels should exceed the policy. Restart DAgger from
the GE3 base (not the regressed round-1 policy) so improvements compound from
strength, not from a weakened start.
