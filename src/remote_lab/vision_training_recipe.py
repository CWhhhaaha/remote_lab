from __future__ import annotations

import copy
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import ViTForImageClassification

from remote_lab.vision_training import (
    RunPaths,
    apply_symmetric_query_key_initialization,
    build_cifar10_loaders,
    build_vit_model,
    compute_layer_asymmetry_ratios,
    compute_reg_loss,
    count_analysis_flops,
    count_reg_flops,
    estimate_model_flops,
    format_epoch_summary,
    format_init_summary,
    maybe_sync,
    prepare_run_paths,
    regularization_active,
    safe_mean,
    set_seed,
    write_json,
)


class ModelEma:
    def __init__(self, model: ViTForImageClassification, decay: float, device: torch.device) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).to(device)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: ViTForImageClassification) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(value):
                value.copy_(model_value)
            else:
                value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off_value = smoothing / max(num_classes - 1, 1)
    on_value = 1.0 - smoothing
    result = torch.full((labels.size(0), num_classes), off_value, device=labels.device, dtype=torch.float32)
    result.scatter_(1, labels.unsqueeze(1), on_value)
    return result


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def rand_bbox(size: tuple[int, int, int, int], lam: float) -> tuple[int, int, int, int]:
    _, _, height, width = size
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = torch.randint(0, width, (1,)).item()
    cy = torch.randint(0, height, (1,)).item()

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def apply_mixup_or_cutmix(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    switch_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_mixup = mixup_alpha > 0.0
    use_cutmix = cutmix_alpha > 0.0
    if not use_mixup and not use_cutmix:
        return pixel_values, smooth_one_hot(labels, num_classes, label_smoothing)

    perm = torch.randperm(labels.size(0), device=labels.device)
    labels_a = smooth_one_hot(labels, num_classes, label_smoothing)
    labels_b = smooth_one_hot(labels[perm], num_classes, label_smoothing)

    choose_cutmix = use_cutmix and (not use_mixup or torch.rand(1).item() < switch_prob)
    if choose_cutmix:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        x1, y1, x2, y2 = rand_bbox(tuple(pixel_values.shape), lam)
        mixed = pixel_values.clone()
        mixed[:, :, y1:y2, x1:x2] = pixel_values[perm, :, y1:y2, x1:x2]
        box_area = max(x2 - x1, 0) * max(y2 - y1, 0)
        adjusted_lam = 1.0 - (box_area / float(pixel_values.size(-1) * pixel_values.size(-2)))
        targets = adjusted_lam * labels_a + (1.0 - adjusted_lam) * labels_b
        return mixed, targets

    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
    mixed = lam * pixel_values + (1.0 - lam) * pixel_values[perm]
    targets = lam * labels_a + (1.0 - lam) * labels_b
    return mixed, targets


def evaluate_model_recipe(
    model: ViTForImageClassification,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float | None, float | None, float]:
    losses: list[float] = []
    correct = 0
    total = 0
    model.eval()
    maybe_sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(pixel_values=pixel_values).logits
            losses.append(float(criterion(logits, labels).item()))
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
    maybe_sync(device)
    duration = time.perf_counter() - start
    accuracy = (correct / total) if total > 0 else None
    return (safe_mean(losses) if losses else None), accuracy, duration


def train_vision_recipe_experiment(
    config: dict[str, Any],
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    del project_root
    training = config["training"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    regularization = config.get("regularization", {})
    instrumentation = config.get("instrumentation", {})
    augmentation = config.get("augmentation", {})
    ema_config = config.get("ema", {})

    set_seed(int(config.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(dataset_config.get("num_classes", 10))
    model = build_vit_model(model_config, num_classes=num_classes)
    if config.get("initialization", {}).get("query_key") == "symmetric":
        apply_symmetric_query_key_initialization(model)
    model.to(device)

    train_loader, test_loader = build_cifar10_loaders(dataset_config, training)
    grad_accum_steps = int(training.get("gradient_accumulation_steps", 1))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(float(training.get("beta1", 0.9)), float(training.get("beta2", 0.999))),
    )

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_optimizer_steps = steps_per_epoch * int(training["max_epochs"])
    warmup_epochs = int(training.get("warmup_epochs", 0))
    warmup_steps = warmup_epochs * steps_per_epoch
    warmup_start_factor = float(training.get("warmup_start_factor", 0.033))

    def lr_lambda(current_step: int) -> float:
        if total_optimizer_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            progress = current_step / max(warmup_steps, 1)
            return warmup_start_factor + progress * (1.0 - warmup_start_factor)
        if total_optimizer_steps == warmup_steps:
            return 1.0
        cosine_progress = (current_step - warmup_steps) / max(total_optimizer_steps - warmup_steps, 1)
        cosine_progress = min(max(cosine_progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    label_smoothing = float(training.get("label_smoothing", 0.0))
    eval_criterion = nn.CrossEntropyLoss()
    run_paths: RunPaths = prepare_run_paths(output_dir)

    eval_every_epochs = int(training.get("eval_every_epochs", 1))
    intervals = regularization.get("intervals", {}).get("layers", []) if regularization.get("enabled") else []
    reg_schedule = regularization.get("schedule", [])
    lambda_value = float(regularization.get("lambda", 0.0))
    penalty = str(regularization.get("penalty", "linear_hinge"))

    mixup_alpha = float(augmentation.get("mixup_alpha", 0.0))
    cutmix_alpha = float(augmentation.get("cutmix_alpha", 0.0))
    switch_prob = float(augmentation.get("switch_prob", 0.5))

    batch_example_pixels, batch_example_labels = next(iter(train_loader))
    batch_example = {"pixel_values": batch_example_pixels.to(device), "labels": batch_example_labels.to(device)}
    task_flops_per_microbatch, flops_method = estimate_model_flops(model, batch_example)
    hidden_size = int(model_config["hidden_size"])
    num_layers = int(model_config["num_hidden_layers"])
    reg_flops_per_microbatch = count_reg_flops(hidden_size, num_layers)
    analysis_flops_per_epoch = count_analysis_flops(hidden_size, num_layers)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and bool(training.get("use_amp", True)))
    use_amp = scaler.is_enabled()
    optimizer.zero_grad(set_to_none=True)

    ema_enabled = bool(ema_config.get("enabled", False))
    ema_model = ModelEma(model, decay=float(ema_config.get("decay", 0.99998)), device=device) if ema_enabled else None
    ema_update_every_steps = int(ema_config.get("update_every_steps", 32))

    epoch_metrics: list[dict[str, Any]] = []
    ratio_history: list[dict[str, Any]] = []

    global_microbatches = 0
    global_optimizer_steps = 0
    total_training_time = 0.0
    total_eval_time = 0.0
    total_analysis_time = 0.0
    total_task_flops = 0
    total_reg_flops = 0
    total_analysis_flops = 0

    maybe_sync(device)
    init_analysis_start = time.perf_counter()
    initial_layer_ratios = compute_layer_asymmetry_ratios(model)
    maybe_sync(device)
    init_analysis_time = time.perf_counter() - init_analysis_start
    total_analysis_time += init_analysis_time
    total_analysis_flops += analysis_flops_per_epoch

    ratio_history.append({"epoch": 0, "layer_asymmetry_ratio": [round(v, 8) for v in initial_layer_ratios]})
    epoch_metrics.append(
        {
            "epoch": 0,
            "regularization_active": False,
            "avg_task_loss": None,
            "avg_total_loss": None,
            "avg_reg_loss": None,
            "training_time_sec": 0.0,
            "evaluation_time_sec": None,
            "eval_loss": None,
            "eval_accuracy": None,
            "eval_loss_raw": None,
            "eval_accuracy_raw": None,
            "eval_loss_ema": None,
            "eval_accuracy_ema": None,
            "analysis_time_sec": round(init_analysis_time, 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "layer_asymmetry_ratio": [round(v, 8) for v in initial_layer_ratios],
        }
    )
    print(
        format_init_summary(
            experiment_name=config.get("experiment_name", "remote-lab-vision-recipe-train"),
            total_epochs=int(training["max_epochs"]),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            analysis_time_sec=init_analysis_time,
            layer_ratios=initial_layer_ratios,
        ),
        flush=True,
    )

    use_tqdm = os.isatty(sys.stdout.fileno()) if hasattr(sys.stdout, "fileno") else False
    progress = tqdm(range(1, int(training["max_epochs"]) + 1), desc=config.get("experiment_name", "remote-lab-vision-recipe-train"), disable=not use_tqdm)
    amp_context = torch.autocast if hasattr(torch, "autocast") else None

    for epoch in progress:
        model.train()
        epoch_total_losses: list[float] = []
        epoch_task_losses: list[float] = []
        epoch_reg_losses: list[float] = []
        reg_enabled = bool(regularization.get("enabled")) and regularization_active(epoch, reg_schedule)
        eval_loss: float | None = None
        eval_accuracy: float | None = None
        eval_time_sec: float | None = None
        eval_loss_raw: float | None = None
        eval_accuracy_raw: float | None = None
        eval_loss_ema: float | None = None
        eval_accuracy_ema: float | None = None

        maybe_sync(device)
        epoch_train_start = time.perf_counter()

        for microbatch_idx, (pixel_values, labels) in enumerate(train_loader, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mixed_pixels, soft_targets = apply_mixup_or_cutmix(
                pixel_values,
                labels,
                num_classes=num_classes,
                label_smoothing=label_smoothing,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                switch_prob=switch_prob,
            )

            autocast_context = (
                amp_context(device_type=device.type, dtype=torch.float16, enabled=use_amp)
                if amp_context is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = model(pixel_values=mixed_pixels).logits
                task_loss = soft_target_cross_entropy(logits, soft_targets)
                total_loss = task_loss
                reg_loss_value = 0.0
                if reg_enabled:
                    reg_loss, _ = compute_reg_loss(model, intervals, penalty=penalty)
                    reg_loss_value = float(reg_loss.detach().item())
                    total_loss = total_loss + lambda_value * reg_loss

            scaled_loss = total_loss / grad_accum_steps
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if reg_enabled:
                total_reg_flops += reg_flops_per_microbatch

            if microbatch_idx % grad_accum_steps == 0 or microbatch_idx == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_optimizer_steps += 1
                if ema_model is not None and global_optimizer_steps % max(ema_update_every_steps, 1) == 0:
                    ema_model.update(model)

            global_microbatches += 1
            if task_flops_per_microbatch is not None:
                total_task_flops += task_flops_per_microbatch

            epoch_task_losses.append(float(task_loss.detach().item()))
            epoch_total_losses.append(float(total_loss.detach().item()))
            if reg_enabled:
                epoch_reg_losses.append(reg_loss_value)

        maybe_sync(device)
        epoch_train_time = time.perf_counter() - epoch_train_start
        total_training_time += epoch_train_time

        maybe_sync(device)
        analysis_start = time.perf_counter()
        layer_ratios = compute_layer_asymmetry_ratios(model)
        maybe_sync(device)
        analysis_time = time.perf_counter() - analysis_start
        total_analysis_time += analysis_time
        total_analysis_flops += analysis_flops_per_epoch

        if eval_every_epochs > 0 and epoch % eval_every_epochs == 0:
            eval_loss_raw, eval_accuracy_raw, raw_duration = evaluate_model_recipe(model, test_loader, device, eval_criterion)
            total_eval_time += raw_duration
            eval_time_sec = raw_duration
            if ema_model is not None:
                eval_loss_ema, eval_accuracy_ema, ema_duration = evaluate_model_recipe(ema_model.module, test_loader, device, eval_criterion)
                total_eval_time += ema_duration
                eval_time_sec += ema_duration
                eval_loss = eval_loss_ema
                eval_accuracy = eval_accuracy_ema
            else:
                eval_loss = eval_loss_raw
                eval_accuracy = eval_accuracy_raw

        ratio_history.append({"epoch": epoch, "layer_asymmetry_ratio": [round(v, 8) for v in layer_ratios]})
        epoch_metrics.append(
            {
                "epoch": epoch,
                "regularization_active": reg_enabled,
                "avg_task_loss": round(safe_mean(epoch_task_losses), 8),
                "avg_total_loss": round(safe_mean(epoch_total_losses), 8),
                "avg_reg_loss": round(safe_mean(epoch_reg_losses), 8) if epoch_reg_losses else None,
                "training_time_sec": round(epoch_train_time, 6),
                "evaluation_time_sec": round(eval_time_sec, 6) if eval_time_sec is not None else None,
                "eval_loss": round(eval_loss, 8) if eval_loss is not None else None,
                "eval_accuracy": round(eval_accuracy, 8) if eval_accuracy is not None else None,
                "eval_loss_raw": round(eval_loss_raw, 8) if eval_loss_raw is not None else None,
                "eval_accuracy_raw": round(eval_accuracy_raw, 8) if eval_accuracy_raw is not None else None,
                "eval_loss_ema": round(eval_loss_ema, 8) if eval_loss_ema is not None else None,
                "eval_accuracy_ema": round(eval_accuracy_ema, 8) if eval_accuracy_ema is not None else None,
                "analysis_time_sec": round(analysis_time, 6),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "layer_asymmetry_ratio": [round(v, 8) for v in layer_ratios],
            }
        )

        avg_total_loss = safe_mean(epoch_total_losses)
        avg_task_loss = safe_mean(epoch_task_losses)
        avg_reg_loss = safe_mean(epoch_reg_losses) if epoch_reg_losses else None
        if use_tqdm:
            progress.set_postfix(
                loss=f"{avg_total_loss:.4f}",
                reg="on" if reg_enabled else "off",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                acc="n/a" if eval_accuracy is None else f"{eval_accuracy:.3f}",
            )
        print(
            format_epoch_summary(
                experiment_name=config.get("experiment_name", "remote-lab-vision-recipe-train"),
                epoch=epoch,
                total_epochs=int(training["max_epochs"]),
                total_loss=avg_total_loss,
                task_loss=avg_task_loss,
                reg_loss=avg_reg_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                training_time_sec=epoch_train_time,
                eval_time_sec=eval_time_sec,
                eval_loss=eval_loss,
                eval_accuracy=eval_accuracy,
                analysis_time_sec=analysis_time,
                reg_enabled=reg_enabled,
                layer_ratios=layer_ratios,
            ),
            flush=True,
        )

    final_row = epoch_metrics[-1] if epoch_metrics else {}
    model.save_pretrained(run_paths.model_dir)
    if ema_model is not None:
        ema_model.module.save_pretrained(run_paths.model_dir / "ema")

    metrics = {
        "experiment_name": config.get("experiment_name"),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset_name": dataset_config.get("name"),
        "epochs": int(training["max_epochs"]),
        "global_microbatches": global_microbatches,
        "global_optimizer_steps": global_optimizer_steps,
        "final_train_loss": final_row.get("avg_total_loss"),
        "final_eval_loss": final_row.get("eval_loss"),
        "final_eval_accuracy": final_row.get("eval_accuracy"),
        "final_eval_loss_raw": final_row.get("eval_loss_raw"),
        "final_eval_accuracy_raw": final_row.get("eval_accuracy_raw"),
        "final_eval_loss_ema": final_row.get("eval_loss_ema"),
        "final_eval_accuracy_ema": final_row.get("eval_accuracy_ema"),
        "training_time_sec": round(total_training_time, 6),
        "evaluation_time_sec": round(total_eval_time, 6),
        "analysis_time_sec": round(total_analysis_time, 6),
        "task_flops": total_task_flops if instrumentation.get("measure_flops", False) else None,
        "reg_flops": total_reg_flops if instrumentation.get("measure_flops", False) else None,
        "analysis_flops": total_analysis_flops if instrumentation.get("measure_flops", False) else None,
        "reported_training_flops": (
            total_task_flops + total_reg_flops
            if instrumentation.get("exclude_analysis_flops", True)
            else total_task_flops + total_reg_flops + total_analysis_flops
        )
        if instrumentation.get("measure_flops", False)
        else None,
        "reported_training_time_sec": (
            total_training_time if instrumentation.get("exclude_analysis_time", True) else total_training_time + total_analysis_time
        ),
        "flops_estimation_method": flops_method,
        "ema_enabled": ema_enabled,
    }

    summary = {
        "config": config,
        "metrics": metrics,
        "epoch_metrics_path": str(run_paths.epoch_metrics_path),
        "ratio_history_path": str(run_paths.ratio_path),
        "model_dir": str(run_paths.model_dir),
    }

    write_json(run_paths.metrics_path, metrics)
    write_json(run_paths.epoch_metrics_path, epoch_metrics)
    write_json(run_paths.ratio_path, ratio_history)
    write_json(run_paths.summary_path, summary)
    return summary
