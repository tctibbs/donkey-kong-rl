# Experiments

The project as a series, one experiment per folder, in order. Each folder is a
self-contained article: the write-up, the exact `config.yaml`, the headline `results.csv`,
and the `figures/` it embeds. Clone [`_TEMPLATE/`](_TEMPLATE/writeup.md) to add one.
Experiments 02 and 08 are negative results, kept because the dead ends are half the lesson.

| # | Experiment | Takeaway |
|---|------------|----------|
| 01 | [The PPO baseline that never climbs](01-ppo-baseline/writeup.md) | Plain PPO scores 4,027 (33x random) but clears the first screen 0 of 100 times; it farms 100-point barrel jumps and never climbs. |
| 02 | [Why standard exploration fails](02-why-exploration-fails/writeup.md) | Negative result. Reward shaping, RND, entropy, and action-set tweaks all stay at 0.00 clears. The blocker is a long zero-reward action chain, not exploration noise. |
| 03 | [Demonstration kickstarting breaks the first screen](03-demonstration-kickstart/writeup.md) | A permanent behavior-cloning floor plus two-screen demos produces the first reliable clears: mean 5,269, clear rate 0.93. |
| 04 | [What actually mattered](04-what-mattered/writeup.md) | Ablations: the BC pressure floor is the load-bearing component (remove it and clears collapse to zero); the reward transform is irrelevant once behavior is demonstrated. |
| 05 | [The 25M final agent](05-final-agent-25m/writeup.md) | The production recipe over 3 seeds: pooled mean 6,884, IQM 7,071, clear 0.97. A solid, boring plateau. |
| 06 | [Go-Explore finds a 30,100 path](06-go-explore-discovery/writeup.md) | A state-restore search discovers a verified 30,100-point open-loop trajectory, but a distilled policy holds only 7,123. The gap is the whole problem. |
| 07 | [Recurrent policies and the closed-loop ceiling](07-recurrent-policy/writeup.md) | An LSTM distilled from the Go-Explore agent reaches 7,285, the best robust policy of the pre-SIL era and the ceiling that held across method classes. |
| 08 | [Search-based DAgger](08-search-dagger-negative/writeup.md) | Negative result. Relabeling policy states with a search expert degrades to 3,638; a tractable per-state expert is too myopic to beat the policy. |
| 09 | [Breaking the ceiling with a permanent demo buffer](09-sil-permanent-demo-buffer/writeup.md) | Self-imitation from a non-evicting demo buffer lifts the mean from 7,285 to 11,293, the first real crack in the ceiling. |
| 10 | [Reaching the 30k range with iterated SIL](10-iterated-sil-30k/writeup.md) | Continuing from experiment 9 with a richer 13-trajectory deep-demo buffer reaches mean 22,820, median 24,400, max 43,700. The 30k range, by a trained policy. |
| 11 | [Past 30k, and the iterated-SIL ceiling](11-past-30k-and-the-sil-ceiling/writeup.md) | One more round lifts the mean to 30,887 (max 59,200, so 50k is reachable in one game), but six attempts to push further all collapse. About 30k is a hard ceiling for this method, set by training instability. |

Raw lab notes behind these articles live in [`../journal/`](../journal); shared background
(the game, the evaluation protocol, the algorithms) is in [`../docs/`](../docs).
