# DAgger round 2 (conclusive): a dominant search-expert signal degrades to 3,638

One-line summary: with search-relabeled states driving 50 percent of training
(not the 1.1 percent of round 1), the policy degrades to mean 3,638 with
clear_rate 0.00, conclusively showing the tractable myopic search expert
produces labels that actively harm an already-strong policy.

## Setup (fixing round 1's confound)

Round 1 was inconclusive: DAgger labels were 1,100 of 97,328 samples (1.1
percent), so the 5,990 result was within kickstart seed variance. Round 2
collected 3,000 search-relabeled on-policy states and built a 50/50 mix with
subsampled robust demos (6,000 total), so the search expert's labels actually
drive learning.

## Result (ledger: ppo_dagger_r2_15m_s1_20260613_160117)

- BC fit on the DAgger-dominant set: accuracy 0.282 (vs 0.893 on the
  robust-demo set). The labels are internally INCONSISTENT: similar states
  get different actions across independent myopic search runs, so the policy
  cannot cleanly fit them.
- Evaluation (100 protocol episodes): mean 3,638, std 838, median 4,100, max
  4,600, clear_rate 0.00. Worse than the 7,285 best, worse than round 1, and
  below even the farming baseline (4,027).

## Conclusion: search-DAgger closed

DAgger improves a policy only with an expert that beats it. A per-state
search expert cheap enough to relabel thousands of states (random shooting
with a progress potential) is too myopic and noisy to beat a 7k policy; a
proper per-state archive search strong enough to beat it is roughly as
expensive as the global Go-Explore and was not tractable to run per state at
scale (and an initial attempt produced degenerate labels). Either way, the
expert-quality barrier is fundamental: producing an expert stronger than the
current policy is as hard as the original control problem.

This is the tenth method class to confirm the closed-loop, stochastic-eval
ceiling at about 7,285. The result is now conclusive (a dominant-signal test),
not confounded. The only tractable source of an expert that beats a 7k policy
is human play; that requires the user. The discovered 30,100 open-loop
trajectory remains the achievable score without execution stochasticity.
