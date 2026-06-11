# Phase 3: what mattered (single-variable ablations on the kickstarting recipe)

One-line summary: the BC auxiliary FLOOR is the load-bearing component
(removing it collapses clears to zero), two-screen demos drive score, the
warm-start and long anneal help moderately, and reward CLIPPING, the thing
Phase 2 first "fixed", turns out to be the best transform once behavior is
demonstrated, producing the new best agent (mean 6,035, clear rate 0.98).

## Ablation table (100-episode Machado-protocol evals, all in the ledger)

Reference R = two-screen sticky demos, BC init 10 epochs, aux 1.0 annealed
over 400 updates to a 0.15 floor, scale transform, 10M steps.

| arm | delta vs R | eval mean | clear rate | reading |
|-----|------------|-----------|------------|---------|
| R seed 1 | none | 5,269 | 0.93 | |
| R seed 2 | seed | 5,227 | 0.94 | headline robust |
| R seed 3 | seed | 4,248 | 0.76 | seed variance is real |
| A5 clip seed 1 | clip instead of scale | 6,035 | 0.98 | NEW BEST; seeds 2,3 running |
| A1 floor 0.0 | no permanent pressure | 3,422 | 0.00 | total collapse; floor is necessary |
| A2 floor 0.05 | weaker floor | 4,591 | 0.81 | partial hold |
| A3 floor 0.3 | stronger floor | 4,392 | 0.78 | over-constrains; inverted U peaks near 0.15 |
| A6 anneal 100 | shorter anneal | 4,686 | 0.82 | longer handoff helps |
| A7 no BC init | aux loss only | 4,735 | 0.83 | aux loss is the load-bearing part |
| (E10) | single-screen demos | 3,289 | 0.70 | two-screen demos drive score |

## Interpretation for the paper

1. The permanent floor is necessary, not cosmetic: even with demos whose
   realized value beats farming, anneal-to-zero (A1) reverts to farming with
   clear rate 0.00. At anneal end the policy's execution is not yet good
   enough for clearing's expected value to win, so gradients still point at
   farming; the floor bridges that execution gap (E10's in-training clear
   rate kept RISING after the anneal, 0.47 to 0.81).
2. Reward magnitude (the scale transform) solved a problem gradients never
   reach. Discovery is handled by demonstrations; afterward, clipping's
   well-conditioned value targets optimize better (A5 beats R by about 15
   percent on score). ADR 0003's mechanism was right about WHY farming is
   clip-optimal, wrong that fixing magnitudes would matter: E1 showed the
   transform is unreachable, and A5 shows it is unnecessary.
3. The dose-response (0 / 0.05 / 0.15 / 0.3 yielding 0.00 / 0.81 / 0.93 /
   0.78 clear rates) is an inverted U: enough pressure to hold behavior,
   not so much that imitation fights optimization.
4. The BC warm-start and the long anneal are conveniences, not requirements
   (A6, A7 within about 10 to 15 percent of R).

## Production recipe (Phase 4)

ppo_abl_clip.yaml: two-screen sticky demos, BC init, aux 1.0 -> 0.15 floor
over 400 updates, CLIP transform. Seeds 2 and 3 in flight for the 3-seed
headline; Phase 4 scales this to the final agent.
