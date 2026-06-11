# ADR 0002: Machado et al. evaluation protocol, also applied to training

Status: accepted. Date: 2026-06-10.

## Context

`DonkeyKong-v5` has no published benchmark to compare against, so the
credibility of every number rests on the evaluation protocol. Machado et al.
(2018), "Revisiting the Arcade Learning Environment", is the community
standard.

## Decision

Evaluation (locked in `EvalConfig`, stored with every checkpoint):

- Sticky actions with p=0.25.
- Full 18-action set.
- Episode cap of 27,000 agent steps (108k frames, 30 minutes).
- Up to 30 random no-op starts.
- No episodic-life trick, no reward clipping. Raw game score is reported.
- Actions sampled from the policy (stochastic evaluation).

Training environments also use sticky actions p=0.25 and the full action set
by default, so train-time stochasticity matches the eval protocol (Machado et
al. recommend injecting stochasticity during training, not only evaluation).
Episodic life and reward clipping stay on for training by default; both are
config fields and get ablated in Phase 3.

Baselines:

- Random policy, measured 2026-06-10 under this exact protocol over 100
  episodes: mean 122.0, std 92.3, max 400 (ledger, kind=random_baseline).
- Human-normalized scores have no canonical DonkeyKong reference. Any
  human-normalized number in the paper must state its reference explicitly;
  choosing that reference is deferred to Phase 4.

## Consequences

- Scores are directly defensible against the standard ALE methodology.
- Training under sticky actions is harder than the deterministic setting most
  hobbyist results use; our scores may look lower but are more robust. The
  sticky-actions-during-training choice is itself an ablation candidate.
