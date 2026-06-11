# Research notes: the ALE DonkeyKong RL landscape

One-line summary: no published paper reports an ALE DonkeyKong score, the
game is two alternating screens with a 5,000-point completion bonus, and the
exploration-bonus literature warns against assuming RND or NoisyNets beat
plain PPO at sub-200M-frame budgets.

## Published baselines: none

- DonkeyKong is outside the canonical Atari-57, so Agent57, R2D2, MuZero,
  Rainbow, and Go-Explore never report it (https://arxiv.org/pdf/2003.13350).
- Machado et al. evaluate 60 games; DonkeyKong does not appear in their
  tables (https://ar5iv.labs.arxiv.org/html/1709.06009).
- Dopamine baselines confirmed absent (no donkeykong file in
  https://github.com/google/dopamine/tree/master/baselines).
- The ROM has been supported since ALE 0.5.1
  (https://ale.farama.org/release_notes/index.html), env documented at
  https://ale.farama.org/environments/donkey_kong/.
- Closest academic work uses a custom Java clone, not ALE (Wiering et al.
  2018, https://link.springer.com/chapter/10.1007/978-3-319-76892-2_11).
- Hobbyist evidence is thin: a Rainbow-style project reports 200 to 300
  rewards at 40k frames, stuck at barrel-jumping
  (https://devpost.com/software/rl_model_donkeykong).

Implication: our own measured curves are the reference. The gap itself is the
paper's contribution.

## Game structure (Atari 2600 port, from the official manual)

Source: https://atariage.com/manual_html_page.php?SoftwareLabelID=149

- Two screens only, alternating with rising difficulty: barrels, then rivets.
  No elevator or conveyor stages in the 2600 port.
- Scoring: barrel/fireball jump 100; rivet removal 100; hammer smash 800;
  per-screen bonus timer starts at 5,000 and counts down; awarded on screen
  completion; Mario dies if it reaches zero.
- Three lives. No final level; play loops until lives are gone.
- Reward shape: semi-sparse. 100-point jumps are reachable by near-random
  play; the 5,000 bonus requires climbing all four floors (a long
  uninterrupted behavior chain with no intermediate reward).
- Known human exploits (unverified, GameFAQs snippets): jump-farming spots
  that yield repeated 100s. Watch for RL agents finding score farming local
  optima instead of climbing.

## Method evidence at modest budgets

- Random Latent Exploration (RLE), 2024 (https://arxiv.org/abs/2407.13755):
  on all 57 games at 40M frames, 5 seeds, "NoisyNet and RND do not
  consistently improve over PPO" while RLE beats PPO, RND, and NoisyNet with
  probability of improvement above 0.5. RLE is a simple PPO add-on (random
  latent-conditioned reward bonus).
- Taïga et al. 2020 (https://arxiv.org/abs/1908.02388): exploration bonuses
  on a Rainbow base do not significantly help outside Montezuma-class games
  and can hurt games where exploration is not the issue.
- CleanRL ppo_rnd_envpool.py (verified against source): dual value heads
  (ext_coef 2.0, int_coef 1.0), extrinsic gamma 0.999 vs intrinsic 0.99,
  non-episodic intrinsic returns, RewardForwardFilter plus RunningMeanStd
  intrinsic normalization, observation normalization warmed up by a random
  agent, lr 1e-4, 128 envs. Headline Montezuma results burned about 2B
  frames; expect far less at our budget.
- Notable detail for DonkeyKong: the 5,000-point bonus argues for testing a
  higher extrinsic gamma (0.999) even in plain PPO, since the payoff arrives
  hundreds of steps after the enabling actions.

## Human reference anchors (community leaderboards, not canonical)

- Top leaderboard scores: 430,100 down to 36,500 in the top 10
  (https://www.lvlupscore.com/system/18/game/6412?tab=game.leaderboard).
- 87,500 in a 15-minute capped game
  (https://recordsetter.com/world-record/highest-points-earned-15-minute-game-donkey-kong-atari-2600/22794).
- About 30,000 reported as a casual personal best
  (http://www.vizzed.com/boards/thread.php?id=29855).
- Working anchors for normalization (our synthesis, must be labeled as such
  in the paper): 5,000 to 10,000 is one-to-two-screen casual play; 30,000 is
  decent casual; 80,000+ is a good dedicated human.
