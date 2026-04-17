# CIFAR-10 Recipe Pair Analysis

## Baseline
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit12_baseline_b2048_recipe_ema999_gpu5`
- final_train_loss: `1.17931455`
- final_eval_loss: `0.51744223`
- final_eval_accuracy: `0.8367`
- final_eval_loss_raw: `0.50271188`
- final_eval_accuracy_raw: `0.8402`
- final_eval_loss_ema: `0.51744223`
- final_eval_accuracy_ema: `0.8367`
- reported_training_time_sec: `1152.3647374082357`
- reported_training_flops: `0`
- final_layer_ratios: `[0.195888, 0.26971242, 0.24918061, 0.40365049, 0.38075012, 0.46634883, 0.4060089, 0.48125637, 0.48537403, 0.46124876, 0.45370707, 0.49285415]`

## Interval
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit12_zeroasym_b2048_recipe_ema999_gpu2`
- final_train_loss: `1.19761167`
- final_eval_loss: `0.54540645`
- final_eval_accuracy: `0.8268`
- final_eval_loss_raw: `0.52647794`
- final_eval_accuracy_raw: `0.8328`
- final_eval_loss_ema: `0.54540645`
- final_eval_accuracy_ema: `0.8268`
- reported_training_time_sec: `952.6712607480586`
- reported_training_flops: `259522560000`
- final_layer_ratios: `[0.00019673, 0.00059884, 0.00117684, 0.00092259, 0.0007403, 0.00036712, 0.00036191, 0.00043828, 0.00041728, 0.00039146, 0.00042945, 0.00036149]`
- regularization_epochs: `200`

## Delta (Interval - Baseline)
- final_train_loss_delta: `0.01829712`
- final_eval_loss_delta: `0.02796422`
- final_eval_accuracy_delta: `-0.0099`
- final_eval_loss_raw_delta: `0.02376606`
- final_eval_accuracy_raw_delta: `-0.0074`
- final_eval_loss_ema_delta: `0.02796422`
- final_eval_accuracy_ema_delta: `-0.0099`
- reported_training_time_sec_delta: `-199.69347666`
- final_layer_ratio_delta: `[-0.19569127, -0.26911358, -0.24800377, -0.4027279, -0.38000982, -0.46598171, -0.40564699, -0.48081809, -0.48495675, -0.4608573, -0.45327762, -0.49249266]`
