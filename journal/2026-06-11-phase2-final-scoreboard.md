# Phase 2 final scoreboard: two-screen kickstarting wins on both axes

One-line summary: E11 (two-screen demos, BC warm-start, annealed auxiliary
loss with a 0.15 permanent floor, scale transform) evaluates at mean 5,269
with a 0.93 clear rate, beating the farming baseline by 31 percent on score
and nearly doubling the survives-screen-1 bar.

## Scoreboard (all rows: 100-episode Machado-protocol evaluations, in ledger)

| arm | eval mean | clear rate | note |
|-----|-----------|------------|------|
| random | 122 | 0.00 | |
| PPO clip baseline | 4,027 | 0.00 | farms barrel jumps |
| E1 scale | = baseline | 0.00 | bit-identical (farming emits only 100s) |
| E2 scale+gamma 0.999 | 4,041 | 0.00 | |
| E3 scale+RND | 3,275 | 0.00 | exploration bonus hurt |
| E4 ent 0.05 | 3,371 | 0.00 | |
| E5 no-sticky train | 3,881 | 0.00 | stochasticity not the blocker |
| E6 minimal actions | 4,079 | 0.00 | |
| E8 BC init only | 4,075 | 0.00 | catastrophic forgetting, 0 train clears |
| E9 BC + annealed aux | 3,761 | 0.00 | 532 train clears, collapsed at anneal end |
| E10 E9 + 0.15 floor | 3,289 | 0.70 | first protocol-passing climber |
| E11 E10 + two-screen demos | 5,269 | 0.93 | WINNER, run ppo_bc_twoscreen_10m_s1_20260611_094412 |

E11 details: median 5,500, max 6,800, std 1,483 (a small failed-climb mode
remains), train clear100 reached 0.97, best train return100 5,464, git
406f237b1903, config hash 39046445ad30, 10M steps in 35 minutes.

## The causal chain the ablations establish

1. Reward clipping makes farming near-optimal, but fixing magnitudes alone
   is a no-op (E1): the policy never EXPERIENCES a differently-valued event.
2. Generic exploration (RND, entropy, action sets, stochasticity removal)
   never produces one successful climb chain (E3 to E6): the barrier is a
   long zero-reward, death-risky behavior sequence.
3. Demonstrations cross the barrier, but BC init alone forgets (E8), and
   annealed-to-zero pressure collapses (E9) because clearing was EV-negative
   at achievable skill with single-screen competence.
4. A permanent floor holds behavior long enough for PPO to compound
   execution (E10: 47 to 81 percent in-training after the anneal).
5. Making post-clear play valuable (two-screen demos) aligns score with
   clearing, and the policy then beats farming outright (E11).

## Production recipe for Phase 3

E11's config (ppo_bc_twoscreen.yaml) is the production arm. Phase 3 ablates
one variable at a time against it: aux floor level (0, 0.05, 0.15, 0.3),
anneal length, demo set (single vs two-screen), reward transform under the
winner, and seeds for the headline run.
