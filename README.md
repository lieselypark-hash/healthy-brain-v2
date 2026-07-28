# healthy-brain-v2
healthy brain

## Model Difference Table

| Component | Normal Model | Parkinson Model | Parkinson Zero-RPE (Evaluation Only) |
|---|---|---|---|
| Agent implementation | a2c_rpe_model.A2CAgent | parkinsons_a2c_rpe_model.A2CAgent | parkinsons_a2c_rpe_model.A2CAgent |
| Network architecture | Shared MLP trunk + actor/critic heads | Same as normal | Same as Parkinson |
| TD/RPE signal used for learning | Raw TD delta: r + gamma * V(next) - V(curr) | Parkinson-impaired TD delta (stochastic transmission and scaling) | Forced to zero (via surviving_fraction=0, transmission_probability=0) |
| Advantage path in actor update | Normalized advantages | Unnormalized impaired advantages | Not used for training (evaluation only) |
| Motor execution | Deterministic action execution | Random slowness and freeze episodes can block movement actions | Same motor behavior as Parkinson |
| Key impairment parameters | None | surviving_fraction, transmission_probability, movement_execution_probability, freeze_episode_probability, freeze_min_steps, freeze_max_steps | surviving_fraction=0.0, transmission_probability=0.0 |
| Training availability | Yes (train.py --agent_variant normal or normal_no_shaping) | Yes (train.py --agent_variant parkinsons or parkinsons_no_shaping) | No |
| Evaluation availability | Via evaluate.py --agent_variant normal or normal_no_shaping | Via evaluate_parkinsons.py --agent_variant parkinsons or parkinsons_no_shaping | Via evaluate_parkinsons.py --agent_variant parkinsons_zero_rpe or parkinsons_zero_rpe_no_shaping |

Notes:

- Parkinson and normal share the same observation space, action space, and network shape.
- The main differences are in how RPE is transformed and whether movement execution is intermittently blocked.
- The zero-RPE variant is intentionally evaluation-only to simulate an extreme impairment without changing training.

## Training and Evaluation Results

After training, metrics and plots are saved automatically:

- results/training_metrics.csv
- results/success_rate.png
- results/reward.png

After evaluation, metrics and plots are saved automatically:

- results/evaluation_metrics.csv
- results/success_rate.png
- results/reward.png

Training metrics include per-episode reward, success, cumulative success rate,
rolling success rate, dopamine/RPE statistics, entropy coefficient, and learning rate.

Generate plots manually from any metrics CSV:

```bash
python results.py
```

Custom input/output paths:

```bash
python results.py --metrics results/training_metrics.csv \
	--success_out results/custom_success.png \
	--reward_out results/custom_reward.png
```

Generate a PETH-adjacent TD/RPE diagram aligned to cue onset and reward events,
with curves for beginning/middle/end of training:

```bash
python peth_rpe.py
```

Outputs:

- results/rpe_peth_begin_mid_end.png
- results/rpe_components_begin_mid_end.png
- results/rpe_peth_event_counts.csv
