# ADR 0005: search-based DAgger for a robust deep-play policy

Status: accepted. Date: 2026-06-13.

## Context

A discovered Go-Explore trajectory scores 30,100 (open-loop, deterministic),
but the best closed-loop policy under the stochastic Machado protocol caps at
7,285 across eight method classes. The measured failure modes:

- Imitation of open-loop demonstrations (BC, distillation) caps at about
  7,000: the demos show one trajectory through state space, so the policy
  never learns to RECOVER when sticky-action noise pushes it off that path,
  and it dies on the rivets screen.
- RL from restored deep states (robustification) reaches high per-segment
  scores (1,600+ from restores) but forgets how to play from the true start
  (catastrophic forgetting / value-retraining distribution shift), collapsing
  full-game evaluation.

Both are instances of the classic imitation-learning failure: covariate shift
between the expert's state distribution and the learner's own. DAgger (Ross et
al. 2011) is the canonical fix: iteratively collect states the LEARNER visits,
label them with an EXPERT's action, aggregate, and retrain. DAgger normally
needs an interactive expert, which most settings lack.

## Decision

We have an interactive expert that prior approaches did not exploit: the
Go-Explore search can compute a near-optimal action from any state via
emulator state restore. Build search-based DAgger:

1. Roll the current policy in the real stochastic env (sticky 0.25), saving
   the emulator state at sampled timesteps along the way (these are the
   states the policy actually visits, including its mistakes).
2. For each saved state, run a short, horizon-limited Go-Explore search from
   that exact state to find the best short continuation; record its first
   action as the expert label (state -> best action).
3. Aggregate the new (observation, expert-action) pairs into the dataset and
   retrain the policy by behavior cloning (warm-started from the current
   policy), with the existing BC-floor machinery to stay stable.
4. Evaluate under the protocol; repeat for several DAgger rounds.

This directly attacks the measured failure: the training distribution becomes
the policy's OWN visited states (so it learns recovery), and the labels come
from search (a strong expert available everywhere), not from a single
open-loop trajectory.

Start with the feedforward GE3 agent as the base (no hidden state to
reconstruct at restored search points, so the search expert is well defined).
If it breaks past 7,300, extend to the recurrent policy.

## Consequences

- Compute: per-state search is the cost. Batch a few hundred on-policy states
  per round across CPU cores; each search is short (horizon about 200 agent
  steps, a few thousand restore-expansions). Expect tens of minutes per round.
- Honesty: this targets the ROBUST stochastic-eval policy, the genuinely open
  problem, distinct from the already-achieved 30,100 discovery trajectory.
- Risk: the search expert from a single state may itself be myopic at the
  horizon limit; mitigated by seeding each per-state search from the global
  30k trajectory bank so it has a strong reference continuation.
- Every DAgger round gets a ledger row and a journal note (round number,
  states relabeled, eval score), per project practice.
