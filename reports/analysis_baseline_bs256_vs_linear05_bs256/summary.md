# Jigsaw Pair Analysis

## Baseline
- run_dir: `/data/chenwei/remote_lab/runs/jigsaw_baseline_bs256_gpu3`
- final_train_loss: `2.99091894`
- final_eval_loss: `3.0412756`
- reported_training_time_sec: `21139.248926192522`
- reported_training_flops: `0`
- final_layer_ratios: `[0.39280459, 0.10323305, 0.05147457, 0.17743714]`
- final_epoch_eval_loss: `3.0412756`

## Interval
- run_dir: `/data/chenwei/remote_lab/runs/jigsaw_interval_linear_lambda05_bs256_gpu2`
- final_train_loss: `2.99183592`
- final_eval_loss: `3.04915619`
- reported_training_time_sec: `21106.709602693096`
- reported_training_flops: `1701209702400`
- final_layer_ratios: `[0.35662538, 0.14917111, 0.08363549, 0.16557328]`
- final_epoch_eval_loss: `3.04915619`
- regularization_epochs: `20`

## Delta (Interval - Baseline)
- final_train_loss_delta: `0.00091698`
- final_eval_loss_delta: `0.00788059`
- reported_training_time_sec_delta: `-32.5393235`
- reported_training_flops_delta: `1701209702400.0`
- final_layer_ratio_delta: `[-0.03617921, 0.04593806, 0.03216092, -0.01186386]`
