# CIFAR-10 Pair Analysis

## Baseline
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit6_baseline_bs512_gpu2`
- final_train_loss: `0.07122558`
- final_eval_loss: `0.7838699`
- final_eval_accuracy: `0.8267`
- reported_training_time_sec: `2071.226449565962`
- reported_training_flops: `0`
- final_layer_ratios: `[0.47344983, 0.31212988, 0.50484842, 0.48012924, 0.47425255, 0.48916316]`

## Interval
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit6_interval_bs512_gpu6`
- final_train_loss: `0.07121521`
- final_eval_loss: `0.77723701`
- final_eval_accuracy: `0.8313`
- reported_training_time_sec: `2924.059907639399`
- reported_training_flops: `2018472099840`
- final_layer_ratios: `[0.48587891, 0.27898693, 0.44059896, 0.48883519, 0.46866173, 0.47953311]`
- regularization_epochs: `30`

## Delta (Interval - Baseline)
- final_train_loss_delta: `-1.037e-05`
- final_eval_loss_delta: `-0.00663289`
- final_eval_accuracy_delta: `0.0046`
- reported_training_time_sec_delta: `852.83345807`
- reported_training_flops_delta: `2018472099840.0`
- final_layer_ratio_delta: `[0.01242908, -0.03314295, -0.06424946, 0.00870595, -0.00559082, -0.00963005]`
