# E10: the BC pressure floor produces the first protocol-passing climber

One-line summary: keeping a permanent 0.15 BC auxiliary coefficient (after
annealing from 1.0 over 400 updates) yields an agent that clears screen 1 in
70 percent of evaluation episodes, the first to pass the ADR 0002
survives-screen-1 bar (at least 50 percent).

## Run facts (ledger, run_id ppo_bc_floor_10m_s1_20260611_090659)

- Config ppo_bc_floor.yaml, hash d4c83e77abf0, git 4393e405956f (clean),
  single-screen sticky demos, scale transform, 10M steps, wall 2,079 s.
- In-training clear100: 0.48 at 1.7M, 0.47 at the anneal boundary (3.3M,
  where E9 had already collapsed to 0.03), then RISING on the floor alone to
  0.81 by 10M. PPO improved climb execution once the floor kept the behavior
  alive: the floor does not just preserve the prior, it lets on-policy
  optimization compound it.
- Evaluation (100 protocol episodes, no auxiliary anything): mean 3,289.0,
  std 2,066.1, median 4,500, max 5,000, clear_rate 0.70.
- The distribution is bimodal: successful episodes land around 4,500 to
  5,000 (clear bonus plus some rivet points), failures around 100 to 300.
- Video (videos/ppo_bc_floor_eval.mp4): episode 1 clears and scores 4,700.

## Scoreboard position

E10 passes the survives-screen-1 criterion but its mean score (3,289) still
trails the farming baseline (4,027) because failed climbs pay almost nothing
and the policy has no rivets-screen competence (single-screen demos). E11
(two-screen demos, same floor recipe, mean demo return 5,728) attacks
exactly that gap and is the candidate production arm.
