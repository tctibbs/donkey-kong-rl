<!--
Experiment report template. Copy this file into the experiment's own folder (one experiment
per folder), beside its config, its results rows, and a figures/ directory. Keep the prose
self-contained: relative image links resolve, and shared definitions live in a shared docs/
location so you can link instead of repeat. No em dashes (use periods, commas, or parentheses).
-->

# Experiment NN: a plain-English title that names the question

> The single sentence a reader should remember. Capitalized, no label.

A two or three sentence hook. Anchor it to something concrete: the task, the previous
experiment's open question, or a surprising result. No theory yet.

## The question

Motivation and hypothesis, in the first person, in the moment. "Last time the result plateaued
at X. I suspected Y was the bottleneck. This experiment tests whether Z fixes it." Lead with
the question, not a restatement of the hook.

## Method

<!-- Do NOT recite shared setup (the data, the task, the standard pipeline): it lives once in
     the shared docs. Link a term inline the first time it matters, at most. State only what is
     SPECIFIC to this experiment. Write in the moment: build on prior experiments, never reveal
     the outcome of a later one. -->

Open with the hypothesis and its risk (what you expected, what could undercut it). Then the
approach: intuition before math. One paragraph of plain English on the change, with an analogy
if it helps. Name what is NEW versus the previous experiment. Defer equations to a linked reference.

## Configuration

Copy-pasteable, never a screenshot.

```yaml
# key knobs (full file: ./config.yaml)
seed: 1
# the handful of fields that define this run
```

## Results

Headline numbers as a table first.

| Metric | Value |
|---|---|
| Headline metric | mean +/- std |
| Median | x |
| Max | x |
| Secondary metric (e.g. accuracy, success rate) | x |
| Seeds | n |
| Steps or epochs | t |

Then a short paragraph interpreting the table: which rows matter and why.

![Learning curve](./figures/learning_curve.png)

Lean on visuals that build intuition at a glance (a learning curve, a distribution, a gif).
Every experiment should carry at least one. A before/after pair is the most legible device for
a general audience.

| Before | After |
|---|---|
| ![before](./figures/before.gif) | ![after](./figures/after.gif) |

## Discussion

What happened and why. Pay off "learn what mattered": name the decision that carried the
result. Include negative results, surprises, and limits honestly.

## Takeaway

Restate the one-liner, now earned by the evidence above.

## Reproduce this

```bash
# Adapt to your project's CLI and layout.
<train> --config experiments/NN-slug/config.yaml
<evaluate> <this run's output> --episodes 100
<plot> <this run's metrics> --out experiments/NN-slug/figures/learning_curve.png
```

## References

Outbound links to deeper treatments, and backward cross-links to neighbouring experiments.
Never a forward link; the index owns the ordering.
