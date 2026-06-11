# Methods and Results (draft)

Status: draft for the white paper. Every number in this document is backed
by a row in `results/ledger.jsonl` (run ids cited inline) and reproduces
from the tagged commits.

## 1. Methods

### 1.1 Environment and evaluation protocol

We study `ALE/DonkeyKong-v5`, the Atari 2600 port of Donkey Kong, through
EnvPool 0.8.4. The 2600 port has two alternating screens: barrels (ramps)
and rivets. Scoring: 100 per barrel or fireball jumped, 100 per rivet
removed, 800 per hammer smash, and a per-screen bonus timer that starts at
5,000, ticks down during play, is awarded on screen completion, and kills
the player if it reaches zero. To our knowledge no published paper reports a
score on this environment; it lies outside the canonical Atari-57 set.

Evaluation follows Machado et al. (2018): sticky actions p=0.25, the full
18-action set, up to 30 no-op starts, a 27,000-agent-step cap (108k frames,
30 minutes), no episodic-life signal, and raw (unclipped) scores. Unless
stated otherwise every evaluation is 100 episodes at a fixed protocol seed.
Training environments also use sticky actions and the full action set.
Observations are 84x84 grayscale, frame skip 4, stack 4.

Headline metrics: raw episode score (mean, median, IQM with 95 percent
bootstrap CIs) and the screen-1 clear rate, defined as the fraction of
episodes containing a single-step raw reward of at least 1,500. The only
score events above 800 are screen-completion bonuses, and a measured clear
pays about 4,200 in one step, so the detector has a wide margin on both
sides. An agent "survives screen 1" if at least half its evaluation
episodes clear.

### 1.2 Base agent

PPO with the Nature CNN trunk, adapted from CleanRL's ppo_atari_envpool
(64 envs, rollout 128, lr 2.5e-4 annealed, gamma 0.99, GAE 0.95, 4
minibatches, 4 epochs, clip 0.1, entropy 0.01). All hyperparameters are
typed config fields; every run records its config hash and git SHA in the
ledger. Hardware: one GTX 1080 Ti and 12 CPU cores; throughput is about
5,000 agent steps per second (CPU-bound), so a 10M-step run takes about 33
minutes.

### 1.3 Demonstration kickstarting

A rule-based demonstrator reads the emulator RAM (player position, climb
state, barrel and firefox positions, byte map verified against the Dennis
Debro disassembly) and plays both screens: it climbs the barrels-screen
ladder chain while jumping nearby barrels, and zigzags the rivets screen.
Under the training distribution (sticky 0.25) it clears the barrels screen
in roughly 80 percent of attempts and scores 5,728 on average across kept
episodes. Demonstrations are 25 successful episodes (about 20,700
state-action pairs) collected at fixed seeds; the dataset regenerates from
the repository with one command and is not shipped in git.

Kickstarting has two parts: (1) behavior-cloning initialization, 10 epochs
of cross-entropy on the demo pairs (reaches about 88 percent action
accuracy); (2) a BC auxiliary term added to the PPO loss, coefficient 1.0
annealed linearly over the first 400 updates to a permanent floor of 0.15,
with a fresh demo minibatch each gradient step.

## 2. Results

### 2.1 Baselines and the farming optimum

Random policy: mean 122 (std 92, max 400), clear rate 0.00
(run kind=random_baseline). PPO at 10M steps: mean 4,027, clear rate 0.00
(ppo_baseline_10m_s1_20260611_012915). The baseline's score is 33x random,
but video and episode-length analysis show it never climbs: it farms
100-point barrel jumps at the bottom ramp until the bonus timer kills each
life. Under standard reward clipping this is near-optimal: the 5,000-point
completion bonus clips to +1, the same as one jump.

### 2.2 Standard interventions all fail to discover climbing

Single-variable arms at 10M steps, each evaluated over 100 protocol
episodes, all with clear rate 0.00:

| arm | eval mean | run id suffix |
|-----|-----------|---------------|
| reward scaling (x0.01 instead of clip) | identical to baseline | see below |
| + gamma 0.999 | 4,041 | ppo_scale_g999_10m_s1 |
| + RND | 3,275 | ppo_rnd_10m_s1 |
| entropy 0.05 | 3,371 | ppo_ent05_10m_s1 |
| no sticky actions (training) | 3,881 | ppo_nosticky_10m_s1 |
| minimal action set | 4,079 | ppo_minactions_10m_s1 |

Two findings deserve emphasis. First, the reward-scaling run reproduced the
clipped baseline bit for bit (identical metrics at all 1,220 updates and an
identical final weight hash): a farming policy emits only 0 and 100-point
rewards, which clip and scale map to the same values, so the transform a
priori cannot matter until the agent first experiences a hammer or a bonus.
Second, RND consumed reachable-state novelty without one successful climb
and degraded the score. The blocker is a multi-second, zero-reward,
death-risky action chain, not insufficient exploration noise.

### 2.3 Kickstarting, forgetting, and the expected-value trap

Behavior cloning alone (ppo_bc_10m_s1) collapses: 0 clears in 6,277
training episodes; PPO's dense farming gradients erase the prior before a
single clear pays out. Adding the annealed auxiliary loss
(ppo_bc_kickstart_10m_s1) produces the first clears (532 during training,
in-training clear rate peaking at 0.25) but collapses to zero exactly when
the coefficient anneals away. The arithmetic explains why: with
barrels-screen-only competence, a clear is worth about 4,400 once and a
failed climb about 200, so at 25 percent execution reliability climbing is
worth about 1,250 per episode against farming's reliable 4,000. Abandoning
climbs was correct optimization. "Survives screen 1" and "maximizes score"
genuinely conflict at low skill.

The two working levers follow directly:

1. A permanent auxiliary floor (0.15) holds the behavior while PPO improves
   execution: in-training clear rate rose from 0.47 at the anneal boundary
   to 0.81 by 10M steps (ppo_bc_floor_10m_s1), evaluating at 0.70 clear
   rate, the first agent past the survives-screen-1 bar.
2. Two-screen demonstrations make post-clear play valuable, aligning score
   with clearing: ppo_bc_twoscreen_10m_s1 evaluates at mean 5,269 (median
   5,500, max 6,800), clear rate 0.93, beating the farming baseline by 31
   percent.

### 2.4 What mattered (ablations)

One-variable ablations against the two-screen recipe (10M steps each, 100
protocol episodes):

| arm | eval mean | clear rate |
|-----|-----------|------------|
| reference (floor 0.15, anneal 400, BC init, scale) seeds 1-3 | 5,269 / 5,227 / 4,248 | 0.93 / 0.94 / 0.76 |
| clip transform seeds 1-3 | 6,035 / 4,503 / 4,983 | 0.98 / 0.75 / 0.84 |
| floor 0.0 | 3,422 | 0.00 |
| floor 0.05 | 4,591 | 0.81 |
| floor 0.3 | 4,392 | 0.78 |
| anneal 100 | 4,686 | 0.82 |
| no BC init (aux only) | 4,735 | 0.83 |
| single-screen demos | 3,289 | 0.70 |

The floor is the load-bearing component: removing it collapses clears to
zero even with two-screen demos (independently reproduced at a second eval
seed), and the dose-response is an inverted U peaking near 0.15. The reward
transform is irrelevant once behavior is demonstrated (clip vs scale within
seed noise across 3 seeds each). The warm-start and long anneal help but
are not required.

### 2.5 Final agent

The production recipe (clip transform, two-screen demos, BC init, aux 1.0
annealed to a 0.15 floor over 400 updates) trained for 25M steps (100M
frames, about 88 minutes each) at seeds 1 to 3
(ppo_final_25m_s{1,2,3}). Per-seed evaluations (100 protocol episodes
each): means 6,991 / 6,958 / 6,702, clear rates 1.00 / 0.95 / 0.97,
maxima 8,100 / 8,300 / 8,200.

Pooled over 300 episodes (percentile bootstrap, 10,000 resamples): mean
6,884 (95 percent CI 6,724 to 7,027), IQM 7,071 (95 percent CI 6,951 to
7,191), median 7,100, std 1,327, clear rate 0.973. That is 56x the random
baseline and 71 percent above the strongest no-demonstration agent (the
farming baseline at 4,027), with the failed-climb mode nearly eliminated
(seed 1 cleared all 100 evaluation episodes). Episode maxima above 8,000
indicate consistent rivets-screen scoring on top of the barrels clear; full
rivets-screen completion (a second bonus) remains the visible headroom.

The 25M horizon mattered: the same recipe at 10M steps evaluated at
5,174 (mean of seed means) with clear rate 0.86; the extra 15M steps under
the 0.15 floor raised both score and reliability substantially.

### 2.6 Reproducibility

Every run records its config (hashed), git SHA, and seed in an append-only
JSONL ledger; learning curves are committed as per-run CSVs; demonstrations
regenerate from seeds; the evaluation protocol is fixed in code. The
pipeline, baselines, and headline results were re-verified from fresh
context at each phase boundary, including an independent re-evaluation of
the headline checkpoint at an unseen seed (mean 5,076, clear rate 0.90,
within sampling noise of the reported 5,269 / 0.93).

## 3. Notes for the full paper

- Human-normalized scores need an explicit reference; no canonical
  DonkeyKong human baseline exists. Community anchors: about 30,000 for a
  decent casual player. Our best agents reach about 20 percent of that.
- Limitations: demonstrations come from a hand-written RAM policy, so the
  method assumes a scriptable expert exists; single game; 1080 Ti budget.
- The RND throughput note (4.5x slowdown from float64 observation
  normalization) belongs in an appendix if RND is revisited.
