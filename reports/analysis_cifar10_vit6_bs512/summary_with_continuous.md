# CIFAR-10 Triple Analysis

## Baseline
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit6_baseline_bs512_gpu2`
- final_train_loss: `0.07122558`
- final_eval_loss: `0.7838699`
- final_eval_accuracy: `0.8267`
- reported_training_time_sec: `2057.331208784133`
- final_layer_ratios: `[0.47344983, 0.31212988, 0.50484842, 0.48012924, 0.47425255, 0.48916316]`

## Interval
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit6_interval_bs512_gpu6`
- final_train_loss: `0.07121521`
- final_eval_loss: `0.77723701`
- final_eval_accuracy: `0.8313`
- reported_training_time_sec: `2156.0969095844775`
- final_layer_ratios: `[0.48587891, 0.27898693, 0.44059896, 0.48883519, 0.46866173, 0.47953311]`
- regularization_epochs: `30`

## Continuous
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit6_interval_bs512_continuous_200ep_gpu3`
- final_train_loss: `0.07151094`
- final_eval_loss: `0.81626039`
- final_eval_accuracy: `0.8175`
- reported_training_time_sec: `964.787099711597`
- final_layer_ratios: `[0.23217927, 0.26071808, 0.3481819, 0.39520454, 0.47818974, 0.61825943]`
- regularization_epochs: `200`
