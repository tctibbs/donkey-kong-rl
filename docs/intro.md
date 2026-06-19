# Start here

This project is a personal experiment in how far I can get a reinforcement learning agent to
play Atari Donkey Kong on one modest machine, and in understanding which design choices
actually make a difference. It is written to be read as a series. If you want the story, go to the
[experiments index](../experiments/README.md) and read in order. If you want the punchline,
it is at the bottom of this page.

## The game

`ALE/DonkeyKong-v5` is the Atari 2600 port of Donkey Kong, run through the Arcade Learning
Environment. It is a hard target for RL for three reasons:

- **Sparse, delayed reward.** The big payoff is a per-screen completion bonus that only
  arrives after a long, death-risky climb. A lazy agent can farm small points at the bottom
  forever and never attempt the climb.
- **Long horizon.** A good game runs thousands of steps and loops the screens many times.
- **Off the beaten path.** It is not in the canonical Atari-57 benchmark, so there is almost
  no published baseline to compare against.

The 2600 port has only two screens, **barrels** (climb the girders, jump or hammer the
barrels) and **rivets**, which then loop at rising difficulty. There are no conveyor or
elevator screens (those exist only in the arcade original). A high score means looping the
barrels-rivets pair many times, not reaching a new level.

## How the game is framed as an MDP

- **Observation:** four stacked 84x84 grayscale frames (frame skip 4). No RAM or privileged
  state is given to the learning agent.
- **Actions:** the full 18-action Atari set.
- **Reward:** 100 per barrel or fireball jumped, 100 per rivet removed, 800 per hammer smash,
  and a per-screen bonus timer that starts at 5,000, counts down during play, is awarded on
  completion, and kills the player if it hits zero.

See the [glossary](glossary.md) for the algorithms and the [methodology note](methodology.md)
for exactly how agents are evaluated and what the headline numbers mean.

## Where it ended up

A random policy scores about 122 and plain PPO about 4,027 (never clearing a screen). A long
campaign of standard methods plateaued at a robust 7,285. The breakthrough was self-imitation
learning from a permanent demonstration buffer: it lifted the evaluation mean to 11,293 and
then, with a richer demo buffer and another round of training, to **22,820 (median 24,400,
best single game 43,700), reaching the 30k range**. The full arc is in the
[experiments](../experiments/README.md).
