# Phase 1 results: PPO 10M scores 33x random, but it is all jump farming

One-line summary: the PPO baseline evaluates at mean 4,084 (33x the 122
random baseline) yet clears screen 1 in 0 of 25 detector episodes; it farms
barrel jumps at the bottom ramp until the bonus timer kills it.

## Run facts (all in ledger, run_id ppo_baseline_10m_s1_20260611_012915)

- Training: 9,994,240 steps (1220 updates of 8,192; config asked for 10M and
  the loop floors to whole updates), 31.6 minutes wall, 5,274 SPS,
  64 envs, 12 threads, git 3dc0ef990433 (clean), config_hash b84f51d8bfbe.
- Train-time return100: best 4,067, final 4,024.
- Evaluation (Machado protocol, 100 episodes, seed 1000): mean 4,084.0,
  std 449.4, median 4,200, max 4,500.
- Video (ale-py path, 3 episodes): returns 4,300 / 3,600 / 4,200, episode
  lengths about 3,400 agent steps. Cross-validates EnvPool eval numbers
  through an independent environment stack.
- W&B: runs p74fdcbo (crashed, Xid 79) and the completed attempt under
  project donkey-kong-rl.

## The local optimum

Frame montage from the video shows Mario at the bottom ramp in every sampled
frame, jumping barrels for 100 points each, never climbing a ladder. Episode
length per life (about 1,100 agent steps) matches the bonus timer running
out, which kills Mario. The research notes predicted exactly this farming
optimum (journal/../docs/research/2026-06-10-donkeykong-rl-landscape.md).

Why PPO settles here: climbing means abandoning a reliable 100-point income
stream for a long, zero-reward, death-risky action chain whose payoff (the
up-to-5,000 bonus) it has never experienced. Gamma 0.99 also discounts a
reward hundreds of steps away to near nothing.

## Implications for Phase 2

1. The screen-1-clear metric (single-step reward >= 1500, ADR 0002) is the
   primary progress metric now, not raw score.
2. Candidate levers, one ADR each when tried: longer horizon (gamma 0.999),
   exploration bonuses (RND), Random Latent Exploration, longer training,
   entropy schedule. Evidence says RND alone is not a free win at our budget.
3. Raw score can MISLEAD here: a farming policy scores 4,000 while a clumsy
   climbing policy might score less. Report both score and clear rate.

## Config hash caveat

The reward-transform schema change (reward_clip boolean replaced by
reward_transform plus reward_scale) changed the computed hash of the same
logical baseline config (b84f51d8bfbe before, 82e354e23fda after). Hashes are
only comparable within a schema version; ledger rows carry the full config
and git SHA, so provenance is unaffected.

## Harness validation

Breakout 3M-step run (same pipeline, env_id swapped, sticky actions on):
best return100 = 19.5, wall 602 s, run_id breakout_validation_3m_s1_20260611_021059.
Random Breakout play scores about 1.2; CleanRL's no-sticky reference reaches
roughly 400 at 10M steps with this curve shape passing through the tens at
3M. The pipeline reports sane numbers on a game with published references,
which validates the stats tracking and protocol plumbing.
