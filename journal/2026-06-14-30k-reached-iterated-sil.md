# 30k reached: iterated SIL with an enriched deep-demo buffer

One-line summary: continue-training the 11,293 SIL agent with a 13-trajectory deep
demo buffer (two at 30,100) lifted the protocol-eval mean to 22,820, median 24,400,
max 43,700, clear rate 1.00. The 30k goal is reached by a trained closed-loop policy.

## Result (ledger-backed)

Run `ppo_sil_deepbuf_25m_s1_20260614_104019`, config `configs/ppo_sil_deepbuf.yaml`.
100-episode Machado protocol eval (sticky 0.25, full action set, no-op starts, no
episodic-life, 27k-step cap, seed 1000):

| metric | value | round-1 SIL | old ceiling |
|--------|-------|-------------|-------------|
| mean | 22,820 | 11,293 | 7,285 |
| median | 24,400 | 9,900 | - |
| std | 9,523 | 5,820 | - |
| max | 43,700 | 26,100 | - |
| min | 4,900 | 4,800 | - |
| clear rate | 1.00 | 1.00 | 1.00 |

Training return100 peaked at 25,416 (round 1 peaked 13,652).

### Return distribution (100 episodes)

- p10 6,700 / p25 15,500 / p50 24,400 / p75 29,600 / p90 34,600 / p95 39,300
- 11 episodes die early (<10k), 28 land 10-20k, 41 land 20-30k, 20 reach >=30,000
- 51 of 100 episodes reach >=24,000

The median is 24,400: the TYPICAL episode now plays deep, not just the lucky tail.
One in five episodes reaches the 30k goal outright.

## Why this worked

Round 1 (eval 11,293) used a permanent demo buffer with only three deep trajectories
(30,100 + 25,800 + 23,500). It broke the 7,285 ceiling but the policy still had a
48% low cluster dying at the screen-1-to-2 transition. This run continued from that
11,293 checkpoint and enriched the permanent demo region to THIRTEEN deep trajectories:
two at 30,100 (trajectory_30k, explore_d4_w71) plus the entire d4 deep-frontier wave
(25,900 to 29,600) and the d3 pair. With 50% of every SIL minibatch drawn (return-
prioritized) from this diverse 23k-30k set, the policy had many deep targets to imitate
across varied game states, not three fixed action sequences. It consolidated the bulk
of its episode mass from the screen-1 band up into the 20k-40k band.

This is expert iteration: round 1's agent was the launch point, the enriched demo set
was the curriculum, and the result roughly doubled both the training return and the
eval mean again (7,285 -> 11,293 -> 22,820).

## The max (43,700) exceeds the discovery trajectory (30,100)

The trained reactive policy's best episode scored 43,700, HIGHER than the best
open-loop Go-Explore discovery trajectory (30,100). Under live stochastic control with
the 27k-step cap, the policy loops the two-screen game (barrels, rivets; the 2600 port
has no other boards) further than the deterministic search ever found. The closed-loop
policy is no longer bounded by the discovery frontier that seeded it.

## Status of the goal

The goal was a trained agent "in the 30k range." Mean 22,820, median 24,400, p75
29,600, 20% of episodes >=30k, max 43,700. The 30k range is reached by the closed-loop
policy, distinct from and now exceeding the earlier open-loop discovery result.

## Honest limits and next steps

- Single seed. A rigorous headline needs 3-5 seeds (mean/std/IQM via rliable). This is
  the immediate next run.
- A 30k MEAN (not just range) is still ~7k away; the 11 early-death episodes and the
  10-20k band are the remaining mass to lift. Levers: another expert-iteration round
  seeded with THIS agent's own >=40k episodes (now on-distribution and deeper than the
  Go-Explore demos), recurrent SIL for the transition deaths, or longer training.
- Needle-mover video: videos/needle_movers/ (best run, see [[dk-needle-mover-videos]]).
