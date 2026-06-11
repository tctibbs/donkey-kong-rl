# E3: RND explores nothing useful here, zero clears, and costs score

One-line summary: PPO+RND at 10M steps posts clear_rate 0.00 (training and
evaluation) and evaluates at mean 3,275, BELOW the plain-PPO baseline's
4,027, so the intrinsic bonus bought no discovery and degraded farming.

## Run facts

- run_id ppo_rnd_10m_s1_20260611_033043, config_hash 767b9bf794cc,
  git e4525932689f (clean), wall 8,549 s at 1,168 SPS (4.5x slower than
  plain PPO due to float64 numpy obs normalization; see contingency journal).
- Training: best return100 3,447, final 3,242, clears 0 of 6,300+ episodes.
  Intrinsic reward decayed to about 0.005 by 4M steps and stayed flat:
  reachable-state novelty was consumed without ever visiting the upper
  platforms.
- Evaluation (100 protocol episodes): mean 3,275.0, std 895.4, median 3,650,
  max 4,200, clear_rate 0.00 (ledger row). Higher variance and lower mean
  than every other arm so far.

## Phase 2 scoreboard after three arms (all ledger-backed)

| arm | eval mean | clear rate |
|-----|-----------|------------|
| random | 122 | 0.00 |
| PPO clip baseline | 4,027 | 0.00 |
| E1 scale (bit-identical to baseline) | 4,027 | 0.00 |
| E2 scale + gamma 0.999 | 4,041 | 0.00 |
| E3 scale + RND | 3,275 | 0.00 |

Consistent with Taiga et al. (2020): exploration bonuses are not a free win,
and here RND actively hurt. The frame-global novelty signal seems dominated
by barrel and flame motion, not Mario's position (mechanism guess in the
contingency journal, to be checked against video).

## Next: discovery wave per the contingency plan

E4 (ent_coef 0.05), E5 (no sticky actions in training), E6 (minimal action
set), sequential 10M plain-PPO runs.
