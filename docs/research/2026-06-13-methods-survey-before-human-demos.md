# Methods survey: what to try before human demonstrations

One-line summary: of the methods surveyed, Self-Imitation Learning (SIL) was the top pick, and
it was the right call. Model-based agents (EfficientZero, DreamerV3) were ruled out as
intractable on an 11 GB 1080 Ti.

**Outcome.** SIL was tried and it worked. Self-imitation from a permanent (non-evicting)
demonstration buffer broke the 7,285 ceiling (mean 11,293), and iterated rounds of it reached a
3-seed mean of 32,420 (IQM 34,033, best single game 59,500), without ever needing the human
demonstrations this survey was written to defer. It later hit a ceiling of its own near 30k, a
training-stability wall rather than a method failure (see
[experiment 11](../../experiments/11-past-30k-and-the-sil-ceiling/writeup.md)). The full
goal-conditioned Go-Explore robustification (candidate 2 below) was never needed; the
permanent-demo-buffer form of SIL was enough. The hardware ruling on model-based held too: when a
higher mean came back into question at the very end, EfficientZero and DreamerV3 were still out of
reach on this GPU. The survey reasoning that led there is kept below as it was written.

## The candidates (tractable, principled, untried at the time)

### 1. Self-Imitation Learning (SIL), the top pick

- What: an off-policy actor-critic add-on to A2C/PPO that replays the agent's OWN high-return
  past episodes (transitions where return R exceeds the value estimate V(s), i.e. positive
  advantage), pulling the policy toward its good decisions within the RL loop. Published to
  "significantly improve A2C on several hard-exploration Atari games" (Oh et al. 2018,
  arXiv 1806.05635).
- Why it fits: the open problem is converting rare high-return experiences into a robust
  closed-loop policy. BC does this offline (covariate shift, caps at 7k); restore-reset does it
  with distribution shift (forgets the start). SIL does it ON-POLICY inside PPO, no restore and
  no separate eval distribution, which is exactly the gap. It is also the mechanism Go-Explore
  itself uses to robustify (see candidate 2).
- Decisive advantage: SIL-from-Demonstrations (arXiv 2203.10905) SEEDS the self-imitation buffer
  with external demonstrations. The 30,100 trajectory and many deep trajectories are already in
  hand. Seeding SIL with them means the policy self-imitates 30k-level play from step 1, inside
  the on-policy learner, warm-started from the 7,285 agent.
- Cost: a replay buffer plus a periodic off-policy update on top of the existing PPO loop. Single
  GPU, cheap. Directly buildable on train/ppo.py.

### 2. Go-Explore robustification done correctly (goal-conditioned + SIL)

- What the earlier attempt got wrong: the restore-reset backward-curriculum was the naive version.
  The published Go-Explore (Ecoffet et al., Nature 2021, "First return, then explore") robustifies
  a deterministic trajectory into a stochastic-eval policy with a GOAL-CONDITIONED policy trained
  by self-imitation that learns to RETURN to archive cells, plus the backward algorithm. The
  goal-conditioning is what avoids the from-start forgetting the naive curriculum hit (the policy
  is conditioned on which cell it is returning to).
- Verdict: a proven path from a 30k deterministic trajectory to a robust policy, but more
  engineering than SIL. It was the planned escalation if plain SIL plateaued. In the event, plain
  SIL did not need it.

## Not tractable on this hardware (honest)

### EfficientZero / EfficientZero V2 (model-based, MuZero-family)

- Superhuman Atari-100k in about 2 hours of GAMEPLAY (EZ-V2, ICML 2024, arXiv 2403.00564), code at
  github.com/Shengjiewang-Jason/EfficientZeroV2.
- But the original EfficientZero needs 4 x 20 GB GPUs, 7 hours per game (arXiv 2111.00210); the
  10 GB path is a per-worker allocation hack, not a fit into 11 GB. Pascal (no bf16, limited
  support) compounds it. Verdict: not realistically trainable to strong play on a single 1080 Ti
  in days.

### DreamerV3 (world model)

- Trained on a single A100; consumer 24 GB RTX 3090 hits out-of-memory after about 12 h / 800k
  steps (github danijar/dreamerv3 issue 152). On 11 GB Pascal: not feasible. Verdict: out.

## Recommendation, and what was done

SIL was the call: seed the buffer with the deep trajectories, warm-start from the 7,285 agent, and
self-imitate inside the on-policy loop. That is what was built, and it broke the ceiling on the
first try, then climbed across several iterated rounds to a 32k mean. The fallbacks were never
reached: the goal-conditioned Go-Explore robustification was not needed, and the human
demonstrations this survey was hedging against were never collected.

## Sources

- Self-Imitation Learning: https://arxiv.org/abs/1806.05635
- Self-Imitation Learning from Demonstrations: https://arxiv.org/pdf/2203.10905
- Go-Explore (Nature 2021): https://gwern.net/doc/reinforcement-learning/exploration/2021-ecoffet.pdf
- EfficientZero: https://arxiv.org/abs/2111.00210
- EfficientZero V2: https://arxiv.org/pdf/2403.00564 ; code https://github.com/Shengjiewang-Jason/EfficientZeroV2
- DreamerV3: https://github.com/danijar/dreamerv3 ; OOM report issue 152
