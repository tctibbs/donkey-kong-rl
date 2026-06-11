# ADR 0003: Phase 2 attacks reward clipping before exploration

Status: accepted. Date: 2026-06-10.

## Context

The Phase 1 PPO baseline (10M steps) reaches mean 4,084 (33x random) with a
0.00 screen-1 clear rate: it farms 100-point barrel jumps at the bottom ramp
until the bonus timer kills Mario (ADR 0002, journal 2026-06-10-phase1-results).

Standard Atari reward clipping maps every positive reward to +1. Donkey Kong's
payoff structure makes this pathological:

- One life of jump farming: about 13 jumps, so +13 clipped reward.
- Clearing the screen: the up-to-5,000 bonus arrives as ONE step, so +1
  clipped reward, plus whatever jumps happened on the way.

Under clipping, the observed farming policy is close to optimal. This is a
reward-design problem first, an exploration problem second. The exploration
literature (research notes, RLE and Taiga et al.) independently warns that
bonus methods are not free wins at our budget, so the cheap, causally-direct
intervention goes first.

## Decision

Add a typed reward transform to the config (`env.reward_transform`):

- `clip`: current behavior (EnvPool clips to [-1, 1]). Baseline.
- `scale`: raw rewards multiplied by `env.reward_scale` (default 0.01) in the
  training loop. A jump becomes +1, hammer +8, screen bonus up to +50, so
  relative magnitudes survive while staying in a PPO-friendly range.
- `none`: raw rewards, as a control for value-loss instability.

Phase 2 experiment ladder, one variable at a time from the Phase 1 baseline,
10M steps each, judged on screen-1 clear rate first and raw score second:

1. E1 `ppo_scale`: reward_transform=scale. Isolates the clipping pathology.
2. E2 `ppo_scale_g999`: E1 + gamma 0.999. At gamma 0.99 a reward 300 steps
   away is discounted to 5 percent; at 0.999 it keeps 74 percent. Decides
   whether horizon length gates climbing once magnitudes are visible.
3. E3: add RND (CleanRL ppo_rnd_envpool recipe) only if E1 and E2 still
   never clear, since the bonus cannot be credited if the agent never
   experiences a clear.

The winner scales to 50M steps as the Phase 2 deliverable.

## Consequences

- If E1 or E2 clears screen 1, the paper gets a clean causal story: a
  one-line reward transform beats an exploration-method upgrade.
- Value targets grow from about 13 to about 60 per life under `scale`;
  watch losses/value_loss for instability and fall back to advantage or
  return normalization if needed.
- Evaluation always reports raw game score regardless of training transform,
  so all runs stay comparable.
