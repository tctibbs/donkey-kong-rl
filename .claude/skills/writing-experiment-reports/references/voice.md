# The house voice

The voice is a lab notebook kept in the moment by the person running the experiments. It is
first person, it is honest, and it reads like it was written the day the run finished, not
reconstructed for a paper months later. Everything below serves that. The before/after pairs
are illustrative; the lesson is the edit, not the domain.

## Write in the moment, in first person

Recount what you did and what you wondered AT THE TIME. The reader is following you forward
through the work, not being lectured by someone who already knows how the series ends.

- No omniscient hindsight: "the most useful result so far", "as we will see", "this turned out
  to be the key." You did not know that yet.
- No forward references to later experiments.

One reliable way to signal the moment is to name what you did not yet know: "I had a model that
works and could not have told you which part earned it," or "the question I most wanted settled
was whether the bottleneck was the data or the architecture." That recounts the discovery
instead of reconstructing it.

Before (retrospective, ranks the result from outside the moment):

> This was the most useful result so far. The headline score looks like a win, but the task
> metric makes the failure unambiguous.

After (in the moment, lets the evidence speak):

> The headline score looks like a win, but the task metric makes the failure unambiguous.

## Lead with the real question, in your own words

Motivation is the author's, stated plainly. Do not quote the project's own rules or goals back
at the reader; that breaks the narrative and reads like compliance, not curiosity.

Before (cites policy, buries the question):

> The project treats understanding as a first-class goal, equal to the score, so a working
> system I cannot explain is half the job. In particular, did the preprocessing step I spent
> two experiments arguing about still matter ...

After (own voice, question first):

> Now that the model trained on clean inputs, did the preprocessing step I had argued over for
> two experiments still matter, or had the new data made that whole argument moot? That was the
> question I most wanted settled. ... a system I cannot explain is half a result.

## No em dashes. Ever.

Use periods, commas, or parentheses. This is absolute. Catch them with a grep for the em-dash
and en-dash characters before publishing.

## Be concrete

Real numbers, real run identifiers, real tables. "0.91 accuracy (95% CI 0.89 to 0.93)", not
"about 90 percent." A specific number invites a reader to check it; a vague one asks them to
trust you.

## Intuition before math

Explain the change in plain English, with an analogy if it helps, before any equation. Equations
and full algorithm detail go in a linked reference, not the body.

## Honest negatives

A clean failure is a first-class result. Say what you tried, why it was reasonable, how it
failed, and what the failure rules out. Do not soften it into a near-win. The series keeps its
dead ends on purpose; they are half the lesson.

Before (vague, learns nothing):

> The method did not improve results.

After (instructive, says what the failure rules out):

> The method only helps when the teacher beats the student, and here it did not. The cheap
> teacher was too noisy to beat the model we already had, and a teacher good enough to beat it
> cost about as much as solving the original problem. The barrier is the teacher, not the training.

## Cut

If a sentence carries no new information and no voice, delete it. The author of these reports
trims hard. Length is not thoroughness. A section may open with a one-sentence restatement of
its heading to orient the reader; that is not filler, but a third paragraph of throat-clearing
is.

## Linking, not pointer blocks

Do not write "(see the intro / methodology note / glossary)." Link a domain term inline the
first time it appears, to the shared glossary or background doc, once, not every occurrence.
After that, trust the reader. Cross-reference asides clutter the line and age badly.

## Forward-pointers are banned

No "What's next" section, no "the next experiment will ...". Each report ends on its own
takeaway. The index is the one place that knows the order.
