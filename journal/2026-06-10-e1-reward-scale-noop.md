# E1: reward scaling was a literal no-op, and that is the finding

One-line summary: the reward-scale run (E1) reproduced the clipped baseline
bit for bit (1,220/1,220 identical metric rows, identical final weight hash
1ecd8320ddaa5ebf), because the farming policy's reward stream contains only
0s and 100s, which clip and scale-by-0.01 map to exactly the same values.

## Run facts

- E1 run_id ppo_scale_10m_s1_20260611_022213, config_hash 7f2cf45db038,
  git d40978b46d77 (clean), 1220 updates, wall 1,921 s, ledger train row.
- Metrics CSV diff vs baseline (ppo_baseline_10m_s1_20260611_012915): zero
  mismatches in mean_return_100 and episode counts across all 1,220 rows.
- Final checkpoint SHA-256 over state dict identical to the baseline's:
  1ecd8320ddaa5ebf. Same seeds plus torch_deterministic plus EnvPool seeding
  made the two trainings the same trajectory once rewards coincided.
- Therefore E1's evaluation is the baseline's evaluation (clear_rate 0.00);
  no separate eval row needed.

## Why this matters

The clipping pathology (ADR 0003) is not wrong, but it is unreachable: the
transform only changes learning if the agent ever RECEIVES a reward that
clip and scale treat differently (hammer 800, screen bonus up to 5,000). The
farming policy never does. The bottleneck is therefore pure DISCOVERY: no
reward re-weighting can matter until the agent experiences a hammer smash or
a screen clear at least once under the training policy.

This activates ADR 0003's E3 contingency: add exploration pressure (RND) on
top of the scale transform. The transform stays, because once RND gets the
agent its first clear, the unclipped bonus magnitude is what lets the value
function prefer climbing; with clipping the clear would be worth +1 and
likely ignored.

## E2 note

E2 (gamma 0.999 plus scale) still runs as planned: gamma changes GAE from
step one regardless of the reward stream, so it diverges immediately and
tests whether horizon alone finds a different basin.
