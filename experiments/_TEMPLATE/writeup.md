<!--
Experiment write-up template. Copy this folder to experiments/NN-slug/, fill it in,
and put every figure the write-up references in ./figures/ (committed). Keep the prose
self-contained so the folder lifts cleanly into a Medium draft: relative image links
resolve, and cross-cutting definitions live in docs/ so you can link instead of repeat.
Avoid em dashes (use periods, commas, or parentheses).
-->

# Experiment NN: Plain-English title that names the question

> One-line takeaway: the single sentence a reader should remember.

A two or three sentence hook. Anchor it to something concrete: the game, the previous
experiment's cliffhanger, or a surprising result. No theory yet.

## The question

Motivation and hypothesis, as prose. "Last time the agent plateaued at X. I suspected Y
was the bottleneck. This experiment tests whether Z fixes it."

## Method

<!-- Do NOT recite the standard MDP (observation, 18 actions, reward structure): it lives
     once in docs/intro.md. Link a term inline the first time it appears if useful, but do
     not add a "see the intro/glossary/methodology" pointer block. Only state what is
     genuinely SPECIFIC to this experiment (a reduced action set, a reward-shaping change).
     Write in the moment: build on PRIOR experiments, but do not reveal the outcomes of
     later ones. -->

Intuition before math. One paragraph of plain English on the algorithm or change, with an
analogy if it helps. Name what is NEW versus the previous experiment. Defer equations to a
linked reference.

## Configuration

Copy-pasteable, never a screenshot.

```yaml
# key knobs (full file: ./config.yaml)
algo: ppo
total_timesteps: 10_000_000
```

## Results

Headline numbers as a table first.

| Metric | Value |
|---|---|
| Mean episode return | x +/- std |
| Median | x |
| Max | x |
| Clear rate | x |
| Seeds | n |
| Training steps | t |

![Learning curve](./figures/learning_curve.png)

A before/after gameplay gif pair is the most legible device for a learning audience.

| Before | After |
|---|---|
| ![before](./figures/before.gif) | ![after](./figures/after.gif) |

## Discussion

What happened and why. Include negative results and dead ends honestly.

## Takeaway

Restate the one-liner, now earned by the evidence above.

## What's next

The cliffhanger that motivates the next experiment. Link it.

## Reproduce this

```bash
uv run dkrl train --config experiments/NN-slug/config.yaml
uv run dkrl plot results/metrics/<run_id>.csv --out experiments/NN-slug/figures/learning_curve.png
uv run dkrl plot-dist --run <run_id> --out experiments/NN-slug/figures/score_distribution.png
uv run dkrl video --checkpoint checkpoints/<run_id>/final.pt --out gameplay.mp4 --episodes 1 --seed 7
```

## References

Outbound links to deeper treatments, and cross-links to neighbouring experiments.
