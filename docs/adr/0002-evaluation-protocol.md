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

"Survives screen 1" definition (added after Phase 1 evidence):

- An episode "clears screen 1" if it contains a single-step raw reward of at
  least 1,500. Rationale: the largest per-event score is the 800-point hammer
  smash; only the screen-completion bonus (remaining timer, up to 5,000) can
  exceed it in one step. The 1,500 threshold leaves margin over stacked
  multi-barrel jump bonuses. Caveat: no positive example has validated the
  detector yet; when an agent first clears, cross-check against video and
  revise this note.
- An agent "survives screen 1" if at least 50 percent of evaluation episodes
  clear screen 1.
- Measured 2026-06-10 (25 episodes each): PPO 10M baseline clear rate 0.00
  (mean return 3,840); random policy clear rate 0.00 (mean return 120). The
  PPO baseline scores 33x random purely by farming barrel jumps at the bottom
  of the ramps until the bonus timer expires (confirmed visually from the
  evaluation video). Beating the survives-screen-1 baseline is the explicit
  Phase 2 target.

## Consequences

- Scores are directly defensible against the standard ALE methodology.
- Training under sticky actions is harder than the deterministic setting most
  hobbyist results use; our scores may look lower but are more robust. The
  sticky-actions-during-training choice is itself an ablation candidate.
