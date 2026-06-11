# Why clears collapse: clearing screen 1 is EV-negative at achievable skill

One-line summary: E9's clear rate collapsed exactly when the BC pressure
annealed away because, with only barrels-screen competence, attempting a
climb is worth less in expectation than farming; the fix is to make
post-clear play valuable (two-screen demos), not to push exploration harder.

## The arithmetic (raw score per life)

- Farming the bottom ramp: about 13 jumps at 100 each before the bonus timer
  kills Mario, roughly 1,300 per life, reliably. Three lives: about 4,000.
- A successful clear: about 4,200 bonus plus some jumps, then the policy
  reaches the rivets screen it has never seen, dies quickly, and subsequent
  lives respawn on rivets where it scores nothing. Episode total: about 4,400.
- A failed climb: about 200.
- At E9's peak 25 percent execution reliability:
  0.25 x 4,400 + 0.75 x 200 = about 1,250 per episode, far below farming's
  4,000. PPO abandoning climbs the moment the auxiliary pressure faded was
  CORRECT optimization, not a failure of credit assignment.

E9 evidence (ledger run ppo_bc_kickstart_10m_s1_20260611_083209): clear100
rose 0.06 to 0.25 during the anneal (532 train clears total), then collapsed
to 0.00 by update 400 when aux hit zero; final eval mean 3,761, clear_rate 0.

## Implication

"Survives screen 1" and "maximizes score" genuinely conflict at low skill.
The escape is to raise the value of clearing: a policy competent on BOTH
screens earns about 4,400 (clear) + 400 to 1,300 (rivet points) and
eventually the second 5,000-ish bonus, dominating farming outright.

## Action taken

- Extended the scripted demonstrator to the rivets screen (zigzag sweep;
  ladder layout x in 41/61/97/117, floors y 133/105/77/49/21 from the
  disassembly; two one-pixel ladder-top bugs found by RAM tracing).
- New demo set data/demos_twoscreen_sticky.npz: 25 episodes, 20,725 pairs,
  mean return 5,728 under sticky 0.25.
- E10 (running): aux floor 0.15 with the OLD single-screen demos, testing
  whether permanent pressure can hold clears against adverse EV.
- E11 (next): two-screen demos plus aux floor, where the EV math finally
  favors the demonstrated behavior.
