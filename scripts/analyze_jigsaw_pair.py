#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and interval-reg Jigsaw runs and export summary plots."
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        required=True,
        help="Run directory for the baseline experiment.",
    )
    parser.add_argument(
        "--interval-run",
        type=Path,
        required=True,
        help="Run directory for the interval-regularized experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where analysis artifacts will be written.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Jigsaw 4-layer",
        help="Prefix used in plot titles.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    epoch_metrics = load_json(run_dir / "analysis" / "epoch_metrics.json")
    ratio_history = load_json(run_dir / "analysis" / "layer_asymmetry_by_epoch.json")
    return {
        "run_dir": str(run_dir.resolve()),
        "metrics": metrics,
        "epoch_metrics": epoch_metrics,
        "ratio_history": ratio_history,
    }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    epoch_metrics = run["epoch_metrics"]
    final_epoch = epoch_metrics[-1] if epoch_metrics else {}
    return {
        "run_dir": run["run_dir"],
        "experiment_name": metrics.get("experiment_name"),
        "epochs": metrics.get("epochs"),
        "final_train_loss": metrics.get("final_train_loss"),
        "final_eval_loss": metrics.get("final_eval_loss"),
        "reported_training_time_sec": metrics.get("reported_training_time_sec"),
        "training_time_sec": metrics.get("training_time_sec"),
        "analysis_time_sec": metrics.get("analysis_time_sec"),
        "reported_training_flops": metrics.get("reported_training_flops"),
        "task_flops": metrics.get("task_flops"),
        "reg_flops": metrics.get("reg_flops"),
        "analysis_flops": metrics.get("analysis_flops"),
        "final_layer_ratios": final_epoch.get("layer_asymmetry_ratio"),
        "regularization_epochs": sum(1 for row in epoch_metrics if row.get("regularization_active")),
        "final_epoch_eval_loss": final_epoch.get("eval_loss"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = payload["baseline"]
    interval = payload["interval"]
    delta = payload["delta"]
    lines = [
        "# Jigsaw Pair Analysis",
        "",
        "## Baseline",
        f"- run_dir: `{baseline['run_dir']}`",
        f"- final_train_loss: `{baseline['final_train_loss']}`",
        f"- final_eval_loss: `{baseline['final_eval_loss']}`",
        f"- reported_training_time_sec: `{baseline['reported_training_time_sec']}`",
        f"- reported_training_flops: `{baseline['reported_training_flops']}`",
        f"- final_layer_ratios: `{baseline['final_layer_ratios']}`",
        f"- final_epoch_eval_loss: `{baseline['final_epoch_eval_loss']}`",
        "",
        "## Interval",
        f"- run_dir: `{interval['run_dir']}`",
        f"- final_train_loss: `{interval['final_train_loss']}`",
        f"- final_eval_loss: `{interval['final_eval_loss']}`",
        f"- reported_training_time_sec: `{interval['reported_training_time_sec']}`",
        f"- reported_training_flops: `{interval['reported_training_flops']}`",
        f"- final_layer_ratios: `{interval['final_layer_ratios']}`",
        f"- final_epoch_eval_loss: `{interval['final_epoch_eval_loss']}`",
        f"- regularization_epochs: `{interval['regularization_epochs']}`",
        "",
        "## Delta (Interval - Baseline)",
        f"- final_train_loss_delta: `{delta['final_train_loss_delta']}`",
        f"- final_eval_loss_delta: `{delta['final_eval_loss_delta']}`",
        f"- reported_training_time_sec_delta: `{delta['reported_training_time_sec_delta']}`",
        f"- reported_training_flops_delta: `{delta['reported_training_flops_delta']}`",
        f"- final_layer_ratio_delta: `{delta['final_layer_ratio_delta']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def numeric_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 8)


def list_delta(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if a is None or b is None or len(a) != len(b):
        return None
    return [round(float(y) - float(x), 8) for x, y in zip(a, b)]


def plot_loss_curves(
    baseline_epochs: list[dict[str, Any]],
    interval_epochs: list[dict[str, Any]],
    output_path: Path,
    title_prefix: str,
) -> None:
    epochs_a = [row["epoch"] for row in baseline_epochs]
    epochs_b = [row["epoch"] for row in interval_epochs]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.plot(epochs_a, [row["avg_total_loss"] for row in baseline_epochs], label="Baseline total loss", linewidth=2.0)
    ax.plot(epochs_b, [row["avg_total_loss"] for row in interval_epochs], label="Interval total loss", linewidth=2.0)
    ax.plot(epochs_a, [row["avg_task_loss"] for row in baseline_epochs], label="Baseline task loss", linewidth=1.4, linestyle="--")
    ax.plot(epochs_b, [row["avg_task_loss"] for row in interval_epochs], label="Interval task loss", linewidth=1.4, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix}: Loss by Epoch")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_time_curves(
    baseline_epochs: list[dict[str, Any]],
    interval_epochs: list[dict[str, Any]],
    output_path: Path,
    title_prefix: str,
) -> None:
    epochs_a = [row["epoch"] for row in baseline_epochs]
    epochs_b = [row["epoch"] for row in interval_epochs]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.plot(epochs_a, [row["training_time_sec"] for row in baseline_epochs], label="Baseline train_sec", linewidth=2.0)
    ax.plot(epochs_b, [row["training_time_sec"] for row in interval_epochs], label="Interval train_sec", linewidth=2.0)
    ax.plot(epochs_a, [row["analysis_time_sec"] for row in baseline_epochs], label="Baseline analysis_sec", linewidth=1.4, linestyle="--")
    ax.plot(epochs_b, [row["analysis_time_sec"] for row in interval_epochs], label="Interval analysis_sec", linewidth=1.4, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title(f"{title_prefix}: Time by Epoch")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_eval_curves(
    baseline_epochs: list[dict[str, Any]],
    interval_epochs: list[dict[str, Any]],
    output_path: Path,
    title_prefix: str,
) -> None:
    baseline_points = [(row["epoch"], row["eval_loss"]) for row in baseline_epochs if row.get("eval_loss") is not None]
    interval_points = [(row["epoch"], row["eval_loss"]) for row in interval_epochs if row.get("eval_loss") is not None]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    if baseline_points:
        ax.plot(
            [epoch for epoch, _ in baseline_points],
            [loss for _, loss in baseline_points],
            label="Baseline eval loss",
            linewidth=2.0,
        )
    if interval_points:
        ax.plot(
            [epoch for epoch, _ in interval_points],
            [loss for _, loss in interval_points],
            label="Interval eval loss",
            linewidth=2.0,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eval loss")
    ax.set_title(f"{title_prefix}: Eval Loss by Epoch")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_curves(
    baseline_ratios: list[dict[str, Any]],
    interval_ratios: list[dict[str, Any]],
    output_path: Path,
    title_prefix: str,
) -> None:
    epochs_a = [row["epoch"] for row in baseline_ratios]
    epochs_b = [row["epoch"] for row in interval_ratios]
    num_layers = len(baseline_ratios[0]["layer_asymmetry_ratio"])
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3.2 * num_layers), dpi=180, sharex=True)
    if num_layers == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        ax.plot(
            epochs_a,
            [row["layer_asymmetry_ratio"][layer_idx] for row in baseline_ratios],
            label="Baseline",
            linewidth=2.0,
        )
        ax.plot(
            epochs_b,
            [row["layer_asymmetry_ratio"][layer_idx] for row in interval_ratios],
            label="Interval",
            linewidth=2.0,
        )
        ax.set_ylabel(f"L{layer_idx + 1} ratio")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
        ax.legend()

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"{title_prefix}: Layerwise Asymmetry Ratio", y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_run = load_run(args.baseline_run)
    interval_run = load_run(args.interval_run)

    baseline_summary = summarize_run(baseline_run)
    interval_summary = summarize_run(interval_run)
    delta = {
        "final_train_loss_delta": numeric_delta(
            baseline_summary["final_train_loss"],
            interval_summary["final_train_loss"],
        ),
        "final_eval_loss_delta": numeric_delta(
            baseline_summary["final_eval_loss"],
            interval_summary["final_eval_loss"],
        ),
        "reported_training_time_sec_delta": numeric_delta(
            baseline_summary["reported_training_time_sec"],
            interval_summary["reported_training_time_sec"],
        ),
        "reported_training_flops_delta": numeric_delta(
            baseline_summary["reported_training_flops"],
            interval_summary["reported_training_flops"],
        ),
        "final_layer_ratio_delta": list_delta(
            baseline_summary["final_layer_ratios"],
            interval_summary["final_layer_ratios"],
        ),
    }

    payload = {
        "baseline": baseline_summary,
        "interval": interval_summary,
        "delta": delta,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", payload)
    write_markdown(args.output_dir / "summary.md", payload)
    plot_loss_curves(
        baseline_run["epoch_metrics"],
        interval_run["epoch_metrics"],
        args.output_dir / "loss_by_epoch.png",
        args.title_prefix,
    )
    plot_time_curves(
        baseline_run["epoch_metrics"],
        interval_run["epoch_metrics"],
        args.output_dir / "time_by_epoch.png",
        args.title_prefix,
    )
    plot_eval_curves(
        baseline_run["epoch_metrics"],
        interval_run["epoch_metrics"],
        args.output_dir / "eval_loss_by_epoch.png",
        args.title_prefix,
    )
    plot_ratio_curves(
        baseline_run["ratio_history"],
        interval_run["ratio_history"],
        args.output_dir / "ratio_by_epoch.png",
        args.title_prefix,
    )

    print(f"analysis_dir={args.output_dir.resolve()}")
    print(f"summary_json={(args.output_dir / 'summary.json').resolve()}")
    print(f"summary_md={(args.output_dir / 'summary.md').resolve()}")
    print(f"loss_plot={(args.output_dir / 'loss_by_epoch.png').resolve()}")
    print(f"time_plot={(args.output_dir / 'time_by_epoch.png').resolve()}")
    print(f"eval_plot={(args.output_dir / 'eval_loss_by_epoch.png').resolve()}")
    print(f"ratio_plot={(args.output_dir / 'ratio_by_epoch.png').resolve()}")


if __name__ == "__main__":
    main()
