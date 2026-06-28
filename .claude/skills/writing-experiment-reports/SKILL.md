---
name: writing-experiment-reports
description: >-
  Writing up a finished machine-learning or reinforcement-learning experiment as a
  structured, self-contained report in a first-person, in-the-moment lab-notebook voice.
  Use when drafting or revising an experiment entry, baseline, ablation, sweep, results
  section, or kept negative result in an experiments series. Not for READMEs, code or API
  docs, commit messages, or papers that synthesize across experiments.
---

# Writing experiment reports

Turn a finished experiment into a report a curious outsider can read in order and learn
from. The result is only half the job; the other half is making it legible. What was
tried, what happened, and **why it mattered**. Treat "a reader can learn what mattered" as
equal in weight to the number itself. A report that states a score but cannot explain which
decision earned it has failed the more important test.

These reports read like a lab notebook kept in the moment, not a paper written in hindsight.
That voice is the whole point of this skill. Do not skip it.

## Project binding

This skill is project-agnostic: it carries the voice and the structure, not any one project's
specifics. The concrete bindings come from the host repo: its training and evaluation commands,
the metric that defines success, where the results ledger lives, and how experiments are laid
out. Pick them up from the repo's `AGENTS.md` or `CLAUDE.md` and an existing report under
`experiments/`, or ask. Keep the discipline; adapt the nouns.

## Workflow

A set of outcomes and constraints, not a rigid script. Adapt the order to the experiment.

1. **Start from the skeleton.** Copy `assets/writeup-template.md` into the experiment's own
   folder (one experiment per folder), beside its config, its results rows, and a `figures/`
   directory. Everything the report references is co-located, so the folder lifts cleanly
   into a draft elsewhere.
2. **Write each section for its job.** The anatomy and the convention for each section live
   in `references/structure.md`. Read it before drafting.
3. **Pull every number from the results ledger**, not from memory. Flag any single-seed
   number as preliminary. A result a reader cannot trace is a result they cannot trust.
4. **Match the house voice.** `references/voice.md` is the heart of this skill, with
   before/after rewrites. The voice is not decoration; it is the deliverable.
5. **Add one row to the experiments index** so the series stays navigable. The index owns
   the ordering; individual reports never point forward to "what's next."
6. **Run the checklist.** `references/checklist.md` before you call it done.

## Non-negotiables

- **No em dashes. Anywhere.** Use periods, commas, or parentheses. Check with a grep.
- **Write in the moment, in first person.** Recount what you did and wondered at the time.
  No omniscient hindsight ("the most useful result so far", "as we will see").
- **Self-contained, never forward-pointing.** No "What's next" section, no "next experiment"
  link. A report may build on earlier ones; it never presumes later ones.
- **Honest negative results.** Dead ends are kept for the lesson and labeled as negative in
  the index. A clean failure is often the most useful entry in the series.
- **Numbers trace to the ledger.** Every headline figure maps to a recorded run.
- **Your own voice, not policy.** Express motivation as the author's, never by quoting the
  repo's own rules back at the reader.

## The anatomy at a glance

Title, then a one-sentence **takeaway** as a blockquote at the very top (capitalized, no
label). A short **hook**. **The question** (motivation and hypothesis). **Method** (intuition
before math; only what is specific to this experiment). **Configuration** (copy-pasteable).
**Results** (headline table first, then figures that build intuition at a glance). **Discussion** (what happened and why,
negatives included). **Takeaway** (the opener, now earned). **Reproduce this** (exact
commands). **References** (papers and backward cross-links). Full detail in
`references/structure.md`.

## Reference files

- `references/structure.md` (section-by-section anatomy and conventions)
- `references/voice.md` (the house voice, with before/after rewrites)
- `references/checklist.md` (pre-publish checklist)
- `assets/writeup-template.md` (the skeleton to copy)
