# Glossary

Defined once here so the experiment write-ups can link instead of repeating.

**PPO (Proximal Policy Optimization).** The base on-policy RL algorithm used throughout. It
improves the policy from freshly collected rollouts while clipping each update so the new
policy cannot move too far from the old one in a single step. Every experiment starts from a
PPO learner; the experiments differ in what is added on top.

**Behavior cloning (BC).** Supervised learning of a policy from demonstrations: predict the
action a demonstrator took in each state. Cheap and fast, but it suffers from covariate shift
(the agent drifts into states the demonstrator never visited) and PPO's own gradients can
erase a cloned behavior before it ever pays off.

**Kickstarting / auxiliary BC floor.** Adding a behavior-cloning term to the PPO loss so the
demonstrated behavior is held in place while PPO improves execution. A permanent small "floor"
on this term prevents the behavior from washing out once the main annealing finishes. This was
the load-bearing trick that produced the first reliable screen clears.

**RND (Random Network Distillation).** An intrinsic-motivation method that rewards the agent
for visiting states a randomly initialized network predicts poorly (a novelty bonus). It did
not help here; the blocker was a long zero-reward action chain, not insufficient exploration.

**Go-Explore.** A discovery method that saves and restores emulator states to systematically
reach deep parts of the game, producing a high-scoring but **open-loop** trajectory (a fixed
action sequence). It found a 30,100-point path. Turning such a trajectory into a robust policy
is a separate problem (see SIL).

**DAgger (Dataset Aggregation).** An interactive imitation method that relabels the states the
current policy actually visits with an expert's action. It needs an expert that beats the
policy; a tractable per-state search expert proved too myopic, so this was a negative result.

**Self-Imitation Learning (SIL).** An off-policy add-on to PPO that replays the agent's own
past transitions whose return beat the value estimate (positive advantage), pulling the policy
toward its own good decisions. Combined with a **permanent, non-evicting demonstration buffer**
(half of every SIL minibatch drawn from seeded deep play), this is what broke the score
ceiling. See [Oh et al. 2018](https://arxiv.org/abs/1806.05635).

**Expert iteration.** Using the previous round's trained agent as the starting point for the
next round, optionally with a better demonstration set each time. The 30k result came from
iterating SIL this way.

**Sticky actions.** Environment stochasticity (probability 0.25 of repeating the previous
action) that prevents memorized action sequences. See the [methodology note](methodology.md).

**Clear rate.** The fraction of evaluation episodes in which the agent completes the first
screen at least once. See the [methodology note](methodology.md) for how it is detected.

**IQM (interquartile mean).** The mean of the middle 50 percent of episode scores, a
robust-to-outliers summary preferred for RL evaluation.
