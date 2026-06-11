# Phase 4: final agent, 3 seeds, pooled mean 6,884 and clear rate 0.97

One-line summary: the production recipe at 25M steps essentially solves the
barrels screen (seed 1 cleared 100/100 evaluation episodes) and scores
consistently on the rivets screen, pooling to mean 6,884 (95 percent CI
6,724 to 7,027), IQM 7,071, across 300 protocol episodes.

## Run facts (ledger)

- configs/ppo_final.yaml, hash 3b17b4abc327, git 3011b9735976, 25M steps
  each (about 88 minutes per run at about 4,700 SPS).
- Seed evals (100 episodes each):
  s1 mean 6,991, clear 1.00, max 8,100 (ppo_final_25m_s1_20260611_163526);
  s2 mean 6,958, clear 0.95, max 8,300 (ppo_final_25m_s2_20260611_180356);
  s3 mean 6,702, clear 0.97, max 8,200 (ppo_final_25m_s3_20260611_193239).
- Pooled (dkrl stats, 10k-resample percentile bootstrap): mean 6,884
  [6,724, 7,027], IQM 7,071 [6,951, 7,191], median 7,100, std 1,327,
  clear rate 0.973.
- Artifacts: results/plots/final_learning_curves.png,
  videos/ppo_final_best.mp4 (three episodes, 6,100 to 6,500, all clearing
  and then working the rivets screen), paper section 2.5 filled.

## Reading

The 25M horizon converted the 10M recipe's bimodal eval distribution into a
reliable one: the failed-climb mode is nearly gone and score variance
halved. Visible headroom: none of the episodes complete the rivets screen
(no second 5,000 bonus observed in the eval maxima); a rivets-completing
demonstrator or longer training are the obvious next pushes beyond this
study's budget.
