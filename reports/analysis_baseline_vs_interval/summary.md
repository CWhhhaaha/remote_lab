# Jigsaw Pair Analysis

## Baseline
- run_dir: `/data/chenwei/remote_lab/runs/jigsaw_baseline_gpu2`
- final_train_loss: `2.9776626`
- final_eval_loss: `3.02767285`
- reported_training_time_sec: `26171.481995135546`
- reported_training_flops: `0`
- final_layer_ratios: `[0.38517717, 0.09941749, 0.07381436, 0.1842414]`

## Interval
- run_dir: `/data/chenwei/remote_lab/runs/jigsaw_interval_gpu3`
- final_train_loss: `2.9893607`
- final_eval_loss: `3.04066942`
- reported_training_time_sec: `26807.59598612599`
- reported_training_flops: `13596046131200`
- final_layer_ratios: `[0.36741066, 0.09616585, 0.05742116, 0.17875347]`
- regularization_epochs: `20`

## Delta (Interval - Baseline)
- final_train_loss_delta: `0.0116981`
- final_eval_loss_delta: `0.01299657`
- reported_training_time_sec_delta: `636.11399099`
- reported_training_flops_delta: `13596046131200.0`
- final_layer_ratio_delta: `[-0.01776651, -0.00325164, -0.0163932, -0.00548793]`
