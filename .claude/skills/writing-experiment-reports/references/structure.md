# Report anatomy

One experiment per folder. The folder holds the report (`writeup.md`), the exact config, the
headline results rows, and a `figures/` directory. The report is self-contained: every image
is a relative link into `figures/`, and shared background (the task, the evaluation protocol,
common methods) is linked from a shared `docs/` location rather than repeated.

These conventions assume each experiment owns a folder beside a results ledger and a shared
`docs/`. Adapt the paths, commands, and metric names to your project, but keep the discipline.

The sections below are in order. Not every experiment needs every one, but the spine
(takeaway, the question, method, results, discussion, takeaway) is always present.

## Title

`# Experiment NN: a plain-English title that names the question or the finding.`
Name the thing a reader would search for, not the method's acronym. "Why standard exploration
fails" beats "RND plus entropy ablation."

## Takeaway (the blockquote)

A single sentence, as a Markdown blockquote, directly under the title:

> The pruned model matches the baseline accuracy at a third of the parameters, and the gain
> comes from the attention layers, not the added depth.

- One sentence. Capitalized. No "Takeaway:" or "TL;DR:" label; the blockquote is the label.
- State the finding, not the topic. A reader who reads only this line should know the result
  and why it matters.

## Hook

Two or three sentences. Anchor to something concrete: the task, the previous experiment's open
question, or a surprising number. No theory yet, no method yet. Earn the read.

## The question

Motivation and hypothesis as prose, in the first person, in the moment. State what you
believed the bottleneck was and what this experiment would tell you. Lead with the actual
question, not a restatement of the hook. Keep it distinct from every other section: this is
"what I needed to know," not "what I did."

## Method

Open with the hypothesis and its risk, in your own words: what you expected to happen and what
could undercut it. (For example: "The hypothesis was that the cheaper teacher would be good
enough to guide the student. The obvious risk was that it might be too weak to beat the model
we already had.") The intuition should echo the question you opened with.

Then the approach. Intuition before math: one or two paragraphs of plain English on the
change, with an analogy if it earns its place. Name explicitly what is NEW versus the previous
experiment.

- Do NOT recite shared setup (the data, the task, the standard pipeline). It lives once in the
  shared docs. Link a term inline the first time it matters, at most.
- Do NOT add "see the intro / glossary / methodology" pointer blocks. Link inline or not at all.
- Write in the moment. Build on prior experiments; never reveal the outcome of a later one.
- Defer equations and full algorithm detail to a linked reference.

## Configuration

The handful of knobs that define this run, as copy-pasteable text in a fenced block, never a
screenshot. Point to the full config file for the rest.

## Results

Headline numbers as a table FIRST, before any prose. Fit the columns to the experiment: a
single run wants the headline metric, median, max, a secondary metric, seeds, and steps; an
ablation wants one row per arm with the knob it changed and the metric it produced.

| Metric | Value |
|---|---|
| Headline metric | mean +/- std |
| Median / Max | x / x |
| Secondary metric (e.g. accuracy, success rate) | x |
| Seeds | n |
| Steps or epochs | t |

Then a short paragraph that INTERPRETS the table: which rows matter and why. Do not make the
reader extract the meaning alone. (For example, open with "Three things fall out of this
table," then state the three findings.) Then the figures, and lean on them: a chart or a gif
conveys at a glance what a paragraph labors to say. Every experiment should carry at least one
visual that builds intuition. A learning curve shows the trajectory, a distribution shows the
spread, and a before/after pair is the single most legible device for a general audience, the
new behavior shown beside the baseline. Commit every figure and link it relatively into
`figures/`:

| Before | After |
|---|---|
| ![before](./figures/before.gif) | ![after](./figures/after.gif) |

Report raw numbers under the stated evaluation protocol. Flag single-seed results as
preliminary in the same breath as you state them.

## Discussion

Three jobs, in order. First, interpret the headline result and the mechanism behind it: which
decision carried it and which were along for the ride (bold the sub-findings if there are
several). This is where "learn what mattered" is paid off. Second, state the limits plainly
(single seed, narrow regime, confounds) rather than hoping no one notices. Third, say what the
result rules in or out about the problem itself. Include negative results, surprises, and dead
ends honestly.

## Takeaway

Restate the finding, now with the depth Discussion earned. You may rephrase the opener and add
what it rules in or out; it need not be a literal copy. A reader who skips the middle should
still leave with the result and why it matters.

## Reproduce this

Exact, copy-pasteable commands: train, evaluate, plot. Use real run identifiers where they
help. Inline comments are welcome in multi-step workflows (collect, then aggregate, then
train). Adapt the commands to the host project's CLI.

## References

Outbound links to deeper treatments (papers, shared docs) and cross-links to NEIGHBOURING
experiments. Backward links ("builds on experiment 6") are fine. Forward links ("next
experiment") are not; the index owns the ordering.

## The index entry

Add one row to the experiments index: the number, a linked title, and a one-line takeaway that
states the result. A good row names the finding or mechanism, not just the topic ("the first
crack in the ceiling"). Mark negative results, for example with "(Negative result)". The index
is the only place that encodes order, so it is the only place that looks forward and back.
