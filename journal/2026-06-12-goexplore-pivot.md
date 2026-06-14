# 30k push: scripted dodging dead-ends, Go-Explore becomes the demo engine

One-line summary: six rule-based attempts at rivets-screen completion all
failed (flames move 8 to 12 raw units per agent step, faster than positional
rules can react), so demonstration generation moved to Go-Explore-style
state-restore search, which immediately found 9,700-point trajectories,
beyond anything the trained agent or script had ever reached.

## Why the script could not finish the rivets screen (measured)

- Rivets reset on death, so all eight must be collected in ONE life.
- The trained agent collects 4 to 5 rivets per life; scripts managed 1 to 2.
- Death post-mortem (24 deaths): 20 on open floor; 10 of 24 with the nearest
  detected flame more than 16 units away at decision time, because flames
  cover 8 to 12 units per agent step and positional snapshots go stale
  within one step. Velocity prediction, retreat logic, ladder-ambush guards,
  jump-window tuning, and a hammer-grab attempt (location never confirmed
  reachable) all measured at zero completions over 20-seed batches.

## Expert iteration alone is too slow

EI round 1 (self-demos at 8,000 to 8,200 merged with script demos, scale
transform, floor 0.15, 25M steps): eval mean 7,117, clear 0.99
(ppo_expert_iter1_25m_s1). A gain of about 130 over the 25M production run.
Rounds of this size do not reach 30,000.

## Go-Explore as demonstration generator

ale-py cloneState/restoreState round-trips exactly (verified), and the env
is deterministic without sticky actions and no-op starts, so action
sequences replay bit-exactly: explored score 9,700 replayed as 9,700.
Implementation (src/dkrl/explore.py): archive keyed by (boards cleared,
board, lives, y band, x band, score band), selection weighted toward depth,
expansion by held-action bursts (climb-friendly), mid-burst archiving.
Wave 1: 10 workers x 150k expansions found 9,700 (vs the agent's previous
best single episode of 8,300).

Training round ppo_goexplore1_25m (in flight): BC on the top 25 explored
trajectories (about 100k pairs, init accuracy 0.925) plus floor 0.15, scale
transform, 25M steps.

## Honest 30k assessment

- 30,000 mean requires roughly three full two-screen loops under sticky
  actions. The exploration frontier itself plateaued near 9,700 in wave 1
  (round-2 difficulty plus archive dilution at depth).
- Wave 2 (iterative deepening: archives reseeded from wave-1 best
  trajectories, 4 workers x 250k expansions) is running to push the frontier;
  the policy can only be as deep as its demonstrations.
- Expected realized-policy discount: execution under sticky actions lands
  below demo depth (E9 to E11 history suggests the policy eventually
  approaches demo level given the floor). Reaching a 30k EVAL MEAN therefore
  needs roughly 30k+ demos AND high-rate execution; this may take several
  deepening-plus-training rounds, and it is not yet certain the approach
  converges that deep. Progress per round is measured and honest.
