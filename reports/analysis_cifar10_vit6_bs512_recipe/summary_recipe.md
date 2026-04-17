# CIFAR-10 Recipe Pair Analysis

## Baseline
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit6_baseline_bs512_recipe_gpu4`
- final_train_loss: `0.65475483`
- final_eval_loss: `2.16469816`
- final_eval_accuracy: `0.1932`
- final_eval_loss_raw: `0.36223253`
- final_eval_accuracy_raw: `0.8958`
- final_eval_loss_ema: `2.16469816`
- final_eval_accuracy_ema: `0.1932`
- reported_training_time_sec: `2912.8135423138738`
- reported_training_flops: `0`
- final_layer_ratios: `[0.09623061, 0.14793858, 0.26491582, 0.19686449, 0.27233976, 0.45624176]`

## Interval
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit6_interval_bs512_recipe_every5_gpu6`
- final_train_loss: `0.63370952`
- final_eval_loss: `2.17301425`
- final_eval_accuracy: `0.1904`
- final_eval_loss_raw: `0.34227736`
- final_eval_accuracy_raw: `0.8991`
- final_eval_loss_ema: `2.17301425`
- final_eval_accuracy_ema: `0.1904`
- reported_training_time_sec: `2189.613919083029`
- reported_training_flops: `6728240332800`
- final_layer_ratios: `[0.20002379, 0.25041059, 0.30013382, 0.35020146, 0.45221204, 0.61322373]`
- regularization_epochs: `100`

## Delta (Interval - Baseline)
- final_train_loss_delta: `-0.02104531`
- final_eval_loss_delta: `0.00831609`
- final_eval_accuracy_delta: `-0.0028`
- final_eval_loss_raw_delta: `-0.01995517`
- final_eval_accuracy_raw_delta: `0.0033`
- final_eval_loss_ema_delta: `0.00831609`
- final_eval_accuracy_ema_delta: `-0.0028`
- reported_training_time_sec_delta: `-723.19962323`
- final_layer_ratio_delta: `[0.10379318, 0.10247201, 0.035218, 0.15333697, 0.17987228, 0.15698197]`
