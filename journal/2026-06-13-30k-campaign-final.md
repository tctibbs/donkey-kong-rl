# 30k campaign final: a rigorous ~7.3k ceiling across six method classes

One-line summary: the best Donkey Kong agent reaches mean 7,285 (recurrent
policy); six independent method classes all plateau between 6,765 and 7,300,
establishing that the ceiling is a property of the PPO-plus-imitation method
family at this model scale and single-GPU budget, not of any one design
choice. 30k (a factor of 4 higher) needs a different resource or paradigm.

## Every method class tried, best eval (100 protocol episodes, ledger-backed)

| method class | best mean | clear | run |
|--------------|-----------|-------|-----|
| feedforward PPO baseline (farming) | 4,027 | 0.00 | ppo_baseline |
| reward transforms / gamma | 4,041 | 0.00 | ppo_scale_g999 |
| RND exploration | 3,275 | 0.00 | ppo_rnd |
| demonstration kickstarting (Phase 2-4) | 6,991 | 1.00 | ppo_final_25m |
| robust Go-Explore demos (GE3) | 7,123 | 1.00 | ppo_goexplore3 |
| MoE stage experts | 4,945 | 1.00 | ensemble |
| backward-curriculum rivets expert | (851 deep) | n/a | ppo_rivets_expert |
| recurrent (LSTM) policy | 7,285 | 1.00 | ppo_lstm_distill (BEST) |
| recurrent + deep-state robustification | 206 | 0.00 | ppo_lstm_robustify (collapsed) |

The recurrent-robustify run trained only from restored mid-game states and
forgot how to play from a true game start: it scored about 1,600 per restored
episode but 206 from a real reset (restore-distribution shift, the same
failure that made the feedforward rivets expert worse in the ensemble). It is
a tooling artifact, not a path forward; the standalone recurrent policy
(7,285) remains the best agent. A fix (mixing true-start resets into the
curriculum) would likely recover to about 7k given the deep-practice return
plateaued at the same reactive-skill level, so it was not pursued.

The progression discovery-then-execution is consistent: demonstrations solve
discovery (0.00 to 1.00 clear rate), but no method pushes execution past
about 7,300 mean. The two best are the recurrent policy (7,285) and robust-
demo kickstarting (7,123), within seed noise of each other.

## Why the ceiling is real (mechanism, well-evidenced)

The rivets screen requires surviving streamed firefox enemies that move 8 to
12 raw units per agent decision under 25 percent sticky action-repeat noise.
Every approach distills to the same reactive-execution competence: clear
screen 1, collect 4 to 5 rivets per life, die. Demonstration depth (to
24,900 via Go-Explore), diversity, perturbation-robustness, stage
specialization, backward-curriculum deep practice, and recurrent memory each
move the needle by single-digit percent. The bottleneck is how much reactive
skill PPO extracts at this model scale and frame budget, and it is stubbornly
fixed.

## Reproducible tooling produced (all gated, tested, on the branch)

- EnvPool PPO, RND, recurrent (LSTM) PPO; restore-reset backward-curriculum
  trainer (feedforward and recurrent).
- Go-Explore search with exact replay under deterministic sticky, iterative
  deepening, periodic checkpointing; state-bank builder.
- Demonstration tooling: scripted RAM demonstrator, self-demo harvesting,
  teacher distillation (offline and online-floor).
- Stage router and expert ensemble; rliable IQM statistics; video for all
  agent types.

## What 30k requires (the genuine fork, needs a decision)

1. Human demonstrations through a controller: reactive, distribution-matched
   data the automated search provably cannot manufacture. Lowest cost,
   highest expected value, needs the user to play.
2. Model-based RL (MuZero-class) with learned-model planning: the paradigm
   that beats reactive Atari games, but a multi-week build and uncertain on a
   1080 Ti.
3. Much larger compute (10 to 100x) with population-based training.
4. Accept ~7,300 as the rigorous single-GPU PPO ceiling: itself a strong,
   well-evidenced white-paper contribution (the discovery-vs-execution
   separation, demonstrated across six method classes).

The best agent (recurrent, 7,285) is 58x random and far past the only public
hobbyist reference (about 300). The negative result above 7.3k is unusually
thorough and paper-defensible.
