# Background

Background on the game and how it is set up as a reinforcement learning problem. For the story
and the results, read the [experiments](../experiments/README.md) in order; for the algorithms
and the exact evaluation protocol, see the [glossary](glossary.md) and the
[methodology note](methodology.md).

## The game

`ALE/DonkeyKong-v5` is the Atari 2600 port of Donkey Kong, run through the Arcade Learning
Environment. It is a hard target for RL for three reasons:

- **Sparse, delayed reward.** The big payoff is a per-screen completion bonus that only
  arrives after a long, death-risky climb. A lazy agent can farm small points at the bottom
  forever and never attempt the climb.
- **Long horizon.** A good game runs thousands of steps and loops the screens many times.
- **Off the beaten path.** It is not in the canonical Atari-57 benchmark, so there is almost
  no published baseline to compare against.

The 2600 port has only two screens, **barrels** (climb the girders, jump or hammer the barrels)
and **rivets**, which then loop at rising difficulty. The arcade original (Nintendo, 1981) had
four distinct stages, adding a conveyor-belt pie factory and a spring-and-elevator screen
between the two. The home version is a much smaller thing, and the reason is the hardware. The
port was built by Coleco, which held the home-console rights to Donkey Kong, and it had to fit
on a 4 KB cartridge for a machine with 128 bytes of RAM and a video chip (the TIA) designed in
the mid-1970s for Pong-era games that draw a handful of objects per scanline. Two of the four
stages simply did not fit, so the conveyors and elevators were cut. It did not help that Coleco
was simultaneously shipping a near-arcade-quality Donkey Kong as the pack-in for its own
ColecoVision, which made the Atari version look like the afterthought it was. So here a high
score means looping the barrels-rivets pair many times, not reaching a new level.

## How the game is framed as an MDP

- **Observation:** four stacked 84x84 grayscale frames (frame skip 4). No RAM or privileged
  state is given to the learning agent.
- **Actions:** the full 18-action Atari set.
- **Reward:** 100 per barrel or fireball jumped, 100 per rivet removed, 800 per hammer smash,
  and a per-screen bonus timer that starts at 5,000, counts down during play, is awarded on
  completion, and kills the player if it hits zero.
