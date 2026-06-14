# SIL with a permanent demo buffer breaks the 7,285 closed-loop ceiling

One-line summary: Self-Imitation Learning seeded from a non-evicting demonstration
buffer lifted the protocol-eval mean from 7,285 to 11,293 (max 26,100, clear rate
1.00), the first closed-loop policy in the campaign to durably beat the ceiling.

## Result (ledger-backed)

Run `ppo_sil_demobuf_15m_s1_20260613_210247`, config `configs/ppo_sil_demobuf.yaml`,
git 6b57240. 100-episode Machado protocol eval (sticky 0.25, full action set, no-op
starts, no episodic-life, 27k-step cap, seed 1000):

| metric | value | prior ceiling |
|--------|-------|---------------|
| mean | 11,293 | 7,285 |
| median | 9,900 | - |
| max | 26,100 | (8,100) |
| std | 5,820 | - |
| clear rate | 1.00 | 1.00 |

Training return100 settled at 12,411 (best 13,652) with clear100 ~1.00, the first run
that trained durably ABOVE 7,285 instead of converging back to it.

### Return distribution (100 episodes)

- p10 5,200 / p25 6,300 / p50 9,900 / p75 15,100 / p90 16,500 / p95 24,500
- 48% clear screen 1 then die to sticky noise (4,800-7,900)
- 43% reach screen 2 and beyond (8,000-18,000)
- 9% go very deep (>18,000, up to 26,100)

The mean is dragged down by the 48% low cluster: episodes where action noise kills
Mario shortly after the first clear. Every episode clears screen 1 (max single-step
reward 4,200-4,300 in all 100). The agent is now a genuine multi-screen closed-loop
player, not a screen-1 specialist.

## Why this worked when round 1 (eval 7,265) did not

Round 1 used a single circular SIL buffer (150k). The seeded 30k transitions were
evicted within minutes of self-play, so SIL only ever reinforced the policy's own 7k
play. It reproduced the ceiling exactly (7,265), confirming the buffer, not SIL, was
the bug.

The fix is SIL-from-Demonstrations (arXiv 2203.10905): a SEPARATE, non-evicting demo
region holding the deep trajectories (30,100 + 25,800 + 23,500 = ~11k transitions,
capacity 15k). Every SIL minibatch draws `demo_fraction=0.5` of its samples from that
permanent region via return-prioritized sampling, so the policy self-imitates
30k-level play through all 15M steps. The other half is the agent's own positive-
advantage self-play (standard SIL). Warm-started from GE3 (7,123). The deep demos act
as a fixed gravity well the on-policy learner cannot wash out.

## What this settles

The 7,285 ceiling was NOT an architectural wall of the feedforward CNN + 4-frame
stack, as the 30k-campaign-final conclusion feared. It was a credit-assignment /
catastrophic-forgetting failure: every prior method (BC, kickstart, restore-reset,
DAgger) either learned the deep play offline (covariate shift) or forgot it under
distribution shift. Keeping the deep demonstrations live INSIDE the on-policy PPO loop,
sampled every update, is what converts rare deep experience into a robust reactive
policy. SIL is the mechanism Go-Explore was designed to pair with, and it delivered.

## Honest limits

- Single seed. Headline rigor needs 3-5 seeds (running next).
- Mean 11,293 is not the 30k goal. The goal is a MEAN near 30k; the agent's MAX is
  already 26,100 and p95 is 24,500, so deep play is reliable when it survives the
  screen-1-to-2 transition. Closing the 48% low cluster is the remaining lever.

## Best closed-loop episode: 31,200 (the 30k range, reached reactively)

An 8-episode protocol-compliant render of the same checkpoint (seed 1000, sticky
0.25, no-op start, full action set, single env) produced episode returns of
16,500 / 31,200 / 16,100 / 7,000 / 10,000 / 6,300 / 17,300 / 19,400. The 31,200
episode (3,258 agent steps) is the first CLOSED-LOOP, stochastic-play episode in the
30k range, distinct from the open-loop Go-Explore discovery trajectory (30,100). It
clears roughly seven screens (about three-and-a-half barrels-to-rivets loops; the 2600
port has only those two boards and loops them). The 100-episode protocol eval gave a
max of 26,100; this render drew a deeper tail. So the agent's reactive capability
touches the goal at the episode level. The remaining gap to a 30k MEAN is consistency,
the 48% of episodes that die at the screen-1-to-2 transition. Documentation clip:
videos/needle_movers/sil_demobuf_31200_tail.mp4 (final ~9s) plus _best.mp4 (full run).

## Next levers (toward 30k mean)

1. Multi-seed the current recipe for a rigorous headline (mean/std/IQM via rliable).
2. Iterated self-imitation: harvest the agent's OWN 24-26k closed-loop eval episodes
   (on-distribution, unlike the open-loop Go-Explore trajectories) and add them to the
   demo buffer, then continue training. On-policy-distribution deep demos should shift
   the low cluster up faster than open-loop replay demos.
3. Recurrent SIL: the same demo-buffer SIL on the LSTM agent (7,285 recurrent base),
   to give memory across the screen-1-to-2 transition where the low cluster dies.
4. Longer training / more updates_per_iter now that the regime is stable above 10k.
