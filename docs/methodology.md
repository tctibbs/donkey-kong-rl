# Evaluation methodology

Every headline number in this repo comes from the same protocol, so experiments are
comparable to each other and honest about stochasticity.

## The protocol

Evaluation follows the Machado et al. (2018) recommendations for the Arcade Learning
Environment:

- **Sticky actions, p = 0.25.** With probability 0.25 the environment repeats the agent's
  previous action instead of the one it just chose. This injects the stochasticity that stops
  an agent from memorizing a single fixed action sequence, and it is the single most important
  reason open-loop "discovery" scores do not transfer directly to a learned policy.
- **Full 18-action set.** No reduced action space.
- **Up to 30 no-op starts.** Each episode begins with a random number of no-ops so the agent
  cannot rely on a fixed start state.
- **108k-frame cap** (27,000 agent steps at frame skip 4), about 30 minutes of play.
- **Raw, unclipped scores.** Reported returns are the game's actual score, not the clipped
  training reward.
- **No episodic-life trick.** Losing a life does not end the episode; the agent plays out all
  its lives, as a human would.

Unless stated otherwise, an evaluation is **100 episodes at a fixed protocol seed (1000)**.

## What the numbers mean

- **Mean / median / max return** are over the evaluation episodes. When mean and median
  diverge, the distribution is skewed, so the [score-distribution figure](../experiments/10-iterated-sil-30k/figures/score_distribution.png)
  matters as much as the mean.
- **Screen-1 clear rate** is the fraction of episodes containing a single-step reward of at
  least 1,500. The only score events above 800 are screen-completion bonuses, and a measured
  clear pays about 4,200 in one step, so the detector has wide margins on both sides.
- **IQM (interquartile mean)** is the mean of the middle 50 percent of episodes, with a
  bootstrap confidence interval. It is more robust than the mean for the small sample sizes
  RL produces, and it is the metric to trust for cross-experiment comparison once a result is
  multi-seeded.

## Honesty notes

- **Open-loop vs closed-loop.** A "discovery" result (a fixed action sequence replayed
  deterministically, as in Go-Explore) is not the same as a robust policy that scores under
  sticky actions. The repo always reports these separately. The gap between them is a real,
  measured quantity, not a rounding error.
- **Seeds.** A single-seed result is preliminary. Headline claims use 3 to 5 seeds and report
  mean, standard deviation, and IQM. Where a result is still single-seed, the write-up says so.
- **Reproducibility.** Every run appends a row to `results/ledger.jsonl` keyed by config hash
  and git SHA, so any number here can be traced to the exact code and config that produced it.
