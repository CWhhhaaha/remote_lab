# CIFAR-10 Recipe Multi-Run Analysis

## Baseline
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit12_baseline_b2048_recipe_ema999_gpu5`
- experiment_name: `cifar10-vit12-baseline-b2048-recipe-ema999-v1`
- epochs: `200`
- final_train_loss: `1.17931455`
- final_eval_loss: `0.51744223`
- final_eval_accuracy: `0.8367`
- final_eval_loss_raw: `0.50271188`
- final_eval_accuracy_raw: `0.8402`
- final_eval_loss_ema: `0.51744223`
- final_eval_accuracy_ema: `0.8367`
- reported_training_time_sec: `1152.3647374082357`
- reported_training_flops: `0`
- regularization_epochs: `0`
- final_layer_ratios: `[0.195888, 0.26971242, 0.24918061, 0.40365049, 0.38075012, 0.46634883, 0.4060089, 0.48125637, 0.48537403, 0.46124876, 0.45370707, 0.49285415]`

## Interval
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit12_interval_b2048_recipe_ema999_front10_every5_gpu6`
- experiment_name: `cifar10-vit12-interval-linear-lambda05-b2048-recipe-ema999-front10-every5-v1`
- epochs: `200`
- final_train_loss: `1.19832091`
- final_eval_loss: `0.54552389`
- final_eval_accuracy: `0.8287`
- final_eval_loss_raw: `0.5321701`
- final_eval_accuracy_raw: `0.834`
- final_eval_loss_ema: `0.54552389`
- final_eval_accuracy_ema: `0.8287`
- reported_training_time_sec: `1039.2407461795956`
- reported_training_flops: `62285414400`
- regularization_epochs: `48`
- final_layer_ratios: `[0.11057656, 0.20824367, 0.25511703, 0.29227343, 0.32137483, 0.3516207, 0.36474952, 0.37095159, 0.39943683, 0.4404285, 0.46681011, 0.44704872]`

## Reverse
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit12_interval_b2048_recipe_ema999_reverse070_080_front10_every5_gpu2`
- experiment_name: `cifar10-vit12-interval-linear-lambda05-b2048-recipe-ema999-reverse070-080-front10-every5-v1`
- epochs: `200`
- final_train_loss: `1.19412913`
- final_eval_loss: `0.54103938`
- final_eval_accuracy: `0.8272`
- final_eval_loss_raw: `0.52969538`
- final_eval_accuracy_raw: `0.8326`
- final_eval_loss_ema: `0.54103938`
- final_eval_accuracy_ema: `0.8272`
- reported_training_time_sec: `909.1608891841024`
- reported_training_flops: `62285414400`
- regularization_epochs: `48`
- final_layer_ratios: `[0.70210081, 0.70038259, 0.70057797, 0.70017302, 0.70132065, 0.70009875, 0.70197999, 0.70099425, 0.70026153, 0.70026469, 0.70043254, 0.70078838]`

## ZeroAsym
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit12_zeroasym_b2048_recipe_ema999_gpu2`
- experiment_name: `cifar10-vit12-interval-linear-lambda05-b2048-recipe-ema999-zeroasym-continuous-200ep-v1`
- epochs: `200`
- final_train_loss: `1.19761167`
- final_eval_loss: `0.54540645`
- final_eval_accuracy: `0.8268`
- final_eval_loss_raw: `0.52647794`
- final_eval_accuracy_raw: `0.8328`
- final_eval_loss_ema: `0.54540645`
- final_eval_accuracy_ema: `0.8268`
- reported_training_time_sec: `952.6712607480586`
- reported_training_flops: `259522560000`
- regularization_epochs: `200`
- final_layer_ratios: `[0.00019673, 0.00059884, 0.00117684, 0.00092259, 0.0007403, 0.00036712, 0.00036191, 0.00043828, 0.00041728, 0.00039146, 0.00042945, 0.00036149]`

## Latent-r32
- run_dir: `/data/chenwei/remote_lab/runs/cifar10_vit12_layersym_latent_b2048_recipe_r32_gpu1`
- experiment_name: `cifar10-vit12-layersym-latent-b2048-recipe-r32-v1`
- epochs: `200`
- final_train_loss: `1.13109085`
- final_eval_loss: `0.57376039`
- final_eval_accuracy: `0.8165`
- final_eval_loss_raw: `0.55090562`
- final_eval_accuracy_raw: `0.8241`
- final_eval_loss_ema: `0.57376039`
- final_eval_accuracy_ema: `0.8165`
- reported_training_time_sec: `894.0467516332865`
- reported_training_flops: `0`
- regularization_epochs: `0`
- final_layer_ratios: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
