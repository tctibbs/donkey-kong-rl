# Methods survey: what to try before human demonstrations

One-line summary: two genuinely-untried, single-GPU, principled methods can
plausibly break the 7,285 closed-loop ceiling. Self-Imitation Learning (SIL)
is the top pick. Model-based agents (EfficientZero, DreamerV3) are NOT
tractable on an 11 GB 1080 Ti.

## Genuinely worth trying (tractable, untried, targets my exact failure)

### 1. Self-Imitation Learning (SIL), top pick

- What: an off-policy actor-critic add-on to A2C/PPO that replays the agent's
  OWN high-return past episodes (transitions where return R exceeds the value
  estimate V(s), i.e. positive advantage), pulling the policy toward its good
  decisions within the RL loop. Published to "significantly improve A2C on
  several hard-exploration Atari games" (Oh et al. 2018, arXiv 1806.05635).
- Why it fits: my failure is converting rare high-return experiences into a
  robust closed-loop policy. BC does this offline (covariate shift, caps at
  7k); restore-reset does it with distribution shift (forgets the start). SIL
  does it ON-POLICY inside PPO, no restore and no separate eval distribution,
  exactly the gap. It is also the mechanism Go-Explore itself uses to
  robustify (see #2).
- Decisive advantage for my case: SIL-from-Demonstrations (arXiv 2203.10905)
  SEEDS the self-imitation buffer with external demonstrations. I have the
  30,100 trajectory and many deep trajectories. Seeding SIL with them means
  the policy self-imitates 30k-level play from step 1, inside the on-policy
  learner, warm-started from the 7,285 agent.
- Cost: a replay buffer plus a periodic off-policy update on top of the
  existing PPO loop. Single GPU, cheap. Directly buildable on train/ppo.py.

### 2. Go-Explore robustification done correctly (goal-conditioned + SIL)

- What I did wrong: my restore-reset backward-curriculum was the NAIVE
  version. The published Go-Explore (Ecoffet et al., Nature 2021, "First
  return, then explore") robustifies a deterministic trajectory into a
  stochastic-eval policy with a GOAL-CONDITIONED policy trained by
  self-imitation that learns to RETURN to archive cells, plus the backward
  algorithm. The goal-conditioning is what avoids the from-start forgetting I
  hit (the policy is conditioned on which cell it is returning to).
- Verdict: the proven path from a 30k deterministic trajectory to a robust
  policy. More engineering than SIL; do it if plain SIL helps but plateaus.

## Not tractable on this hardware (honest)

### EfficientZero / EfficientZero V2 (model-based, MuZero-family)

- Superhuman Atari-100k in about 2 hours of GAMEPLAY (EZ-V2, ICML 2024,
  arXiv 2403.00564), code at github.com/Shengjiewang-Jason/EfficientZeroV2.
- But the original EfficientZero needs 4 x 20 GB GPUs, 7 hours per game
  (arXiv 2111.00210); the 10 GB path is a per-worker allocation hack, not a
  fit into 11 GB. Pascal (no bf16, limited support) compounds it. Verdict:
  not realistically trainable to strong play on a single 1080 Ti in days.

### DreamerV3 (world model)

- Trained on a single A100; consumer 24 GB RTX 3090 hits out-of-memory after
  about 12 h / 800k steps (github danijar/dreamerv3 issue 152). On 11 GB
  Pascal: not feasible. Verdict: out.

## Recommendation

Try SIL next (seeded with the deep trajectories, warm-started from the 7,285
agent). It is the single most tractable, principled, untried lever and is the
exact robustification mechanism the discovery method (Go-Explore) was designed
to pair with. If it helps but plateaus, escalate to the full goal-conditioned
Go-Explore robustification. Human demonstrations remain the fallback only if
both self-imitation paths plateau, not before.

## Sources

- Self-Imitation Learning: https://arxiv.org/abs/1806.05635
- Self-Imitation Learning from Demonstrations: https://arxiv.org/pdf/2203.10905
- Go-Explore (Nature 2021): https://gwern.net/doc/reinforcement-learning/exploration/2021-ecoffet.pdf
- EfficientZero: https://arxiv.org/abs/2111.00210
- EfficientZero V2: https://arxiv.org/pdf/2403.00564 ; code https://github.com/Shengjiewang-Jason/EfficientZeroV2
- DreamerV3: https://github.com/danijar/dreamerv3 ; OOM report issue 152
