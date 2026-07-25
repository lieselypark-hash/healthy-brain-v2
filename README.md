# healthy-brain-v2
healthy brain

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
- results/rpe_peth_event_counts.csv
