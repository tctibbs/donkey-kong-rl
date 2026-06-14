# Recurrent policy: building it correctly took three tries

One-line summary: an LSTM policy needs online teacher distillation (not
single-frame BC, not offline distillation) to retain a feedforward expert's
behavior while fine-tuning; with that fix it holds GE3's clearing skill and
fine-tunes under sticky noise, testing whether memory breaks the 7k ceiling.

## Why the first two recurrent attempts failed

1. Single-frame BC kickstart (ppo_lstm_kickstart): BC trained the LSTM on
   demo frames with a zeroed hidden state, which never occurs during real
   rollout (hidden state is carried). The prior did not transfer; the run
   sat at return 61, clear 0.00 through 9.8M steps. Killed.
2. Offline distillation, no floor (ppo_lstm_distill, v1): distilling GE3 into
   the LSTM over real rollout sequences reached 0.887 action-match accuracy
   and the LSTM cleared at 0.36 on update 10. But PPO with no holding term
   destroyed the behavior by update 30 (return 96, clear 0.00). Same
   catastrophic-forgetting mode as feedforward E8.

## The fix: online distillation floor

Distill GE3 into the LSTM with correct recurrence (CNN trunk copied; actions
matched over (T, N) rollout chunks with hidden-state propagation, 0.887
accuracy), THEN keep the teacher live during PPO and add an annealed
teacher-matching cross-entropy to the loss (coef 1.0 to a 0.4 floor over 800
updates). This is canonical kickstarting (Schmitt et al.) with a recurrent
student. Result: the policy starts at about 3,800 (clear 0.60) and CLIMBS
(4,522 / 0.67 by update 40) instead of collapsing. The floor holds behavior
while PPO improves execution, exactly as the feedforward floor did.

## The test in progress

ppo_lstm_distill_25m: if the LSTM's flame-velocity memory lets it dodge the
rivets screen better than the feedforward ceiling, eval should exceed 7,123.
If it lands at about 7k like everything else, the ceiling survives even an
architecture change, which is the strongest possible form of the negative
result. Either outcome is decisive and documented.
