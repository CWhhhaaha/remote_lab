#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline, interval, and continuous-reg CIFAR-10 ViT runs and export summary plots."
    )
    parser.add_argument("--baseline-run", type=Path, required=True, help="Run directory for the baseline experiment.")
    parser.add_argument("--interval-run", type=Path, required=True, help="Run directory for the interval-regularized experiment.")
    parser.add_argument("--continuous-run", type=Path, required=True, help="Run directory for the continuous-regularized experiment.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where analysis artifacts will be written.")
    parser.add_argument("--title-prefix", default="CIFAR-10 ViT-6", help="Prefix used in plot titles.")
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
        "final_eval_accuracy": metrics.get("final_eval_accuracy"),
        "reported_training_time_sec": metrics.get("reported_training_time_sec"),
        "reported_training_flops": metrics.get("reported_training_flops"),
        "final_layer_ratios": final_epoch.get("layer_asymmetry_ratio"),
        "regularization_epochs": sum(1 for row in epoch_metrics if row.get("regularization_active")),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    baseline = payload["baseline"]
    interval = payload["interval"]
    continuous = payload["continuous"]
    lines = [
        "# CIFAR-10 Triple Analysis",
        "",
        "## Baseline",
        f"- run_dir: `{baseline['run_dir']}`",
        f"- final_train_loss: `{baseline['final_train_loss']}`",
        f"- final_eval_loss: `{baseline['final_eval_loss']}`",
        f"- final_eval_accuracy: `{baseline['final_eval_accuracy']}`",
        f"- reported_training_time_sec: `{baseline['reported_training_time_sec']}`",
        f"- final_layer_ratios: `{baseline['final_layer_ratios']}`",
        "",
        "## Interval",
        f"- run_dir: `{interval['run_dir']}`",
        f"- final_train_loss: `{interval['final_train_loss']}`",
        f"- final_eval_loss: `{interval['final_eval_loss']}`",
        f"- final_eval_accuracy: `{interval['final_eval_accuracy']}`",
        f"- reported_training_time_sec: `{interval['reported_training_time_sec']}`",
        f"- final_layer_ratios: `{interval['final_layer_ratios']}`",
        f"- regularization_epochs: `{interval['regularization_epochs']}`",
        "",
        "## Continuous",
        f"- run_dir: `{continuous['run_dir']}`",
        f"- final_train_loss: `{continuous['final_train_loss']}`",
        f"- final_eval_loss: `{continuous['final_eval_loss']}`",
        f"- final_eval_accuracy: `{continuous['final_eval_accuracy']}`",
        f"- reported_training_time_sec: `{continuous['reported_training_time_sec']}`",
        f"- final_layer_ratios: `{continuous['final_layer_ratios']}`",
        f"- regularization_epochs: `{continuous['regularization_epochs']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_curves(
    runs: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
    title: str,
    ylabel: str,
    key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for label, rows in runs:
        points = [(row["epoch"], row.get(key)) for row in rows if row.get(key) is not None]
        if not points:
            continue
        ax.plot([x for x, _ in points], [y for _, y in points], label=label, linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    runs: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
    title_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    styles = {
        "Baseline": "-",
        "Interval": "-",
        "Continuous": "-",
    }
    for label, rows in runs:
        ax.plot(
            [row["epoch"] for row in rows],
            [row["avg_total_loss"] for row in rows],
            label=f"{label} total loss",
            linewidth=2.0,
            linestyle=styles.get(label, "-"),
        )
    for label, rows in runs:
        ax.plot(
            [row["epoch"] for row in rows],
            [row["avg_task_loss"] for row in rows],
            label=f"{label} task loss",
            linewidth=1.4,
            linestyle="--",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix}: Loss by Epoch")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_curves(
    runs: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
    title_prefix: str,
) -> None:
    num_layers = len(runs[0][1][0]["layer_asymmetry_ratio"])
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3.2 * num_layers), dpi=180, sharex=True)
    if num_layers == 1:
        axes = [axes]
    for layer_idx, ax in enumerate(axes):
        for label, rows in runs:
            ax.plot(
                [row["epoch"] for row in rows],
                [row["layer_asymmetry_ratio"][layer_idx] for row in rows],
                label=label,
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
    continuous_run = load_run(args.continuous_run)

    payload = {
        "baseline": summarize_run(baseline_run),
        "interval": summarize_run(interval_run),
        "continuous": summarize_run(continuous_run),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary_with_continuous.json", payload)
    write_markdown(args.output_dir / "summary_with_continuous.md", payload)

    epoch_runs = [
        ("Baseline", baseline_run["epoch_metrics"]),
        ("Interval", interval_run["epoch_metrics"]),
        ("Continuous", continuous_run["epoch_metrics"]),
    ]
    ratio_runs = [
        ("Baseline", baseline_run["ratio_history"]),
        ("Interval", interval_run["ratio_history"]),
        ("Continuous", continuous_run["ratio_history"]),
    ]

    plot_loss_curves(epoch_runs, args.output_dir / "loss_by_epoch_with_continuous.png", args.title_prefix)
    plot_metric_curves(
        epoch_runs,
        args.output_dir / "eval_loss_by_epoch_with_continuous.png",
        f"{args.title_prefix}: Eval Loss by Epoch",
        "Eval loss",
        "eval_loss",
    )
    plot_metric_curves(
        epoch_runs,
        args.output_dir / "eval_accuracy_by_epoch_with_continuous.png",
        f"{args.title_prefix}: Eval Accuracy by Epoch",
        "Accuracy",
        "eval_accuracy",
    )
    plot_metric_curves(
        epoch_runs,
        args.output_dir / "time_by_epoch_with_continuous.png",
        f"{args.title_prefix}: Train Time by Epoch",
        "Seconds",
        "training_time_sec",
    )
    plot_ratio_curves(ratio_runs, args.output_dir / "ratio_by_epoch_with_continuous.png", args.title_prefix)

    print(f"analysis_dir={args.output_dir.resolve()}")
    print(f"summary_json={(args.output_dir / 'summary_with_continuous.json').resolve()}")
    print(f"summary_md={(args.output_dir / 'summary_with_continuous.md').resolve()}")
    print(f"loss_plot={(args.output_dir / 'loss_by_epoch_with_continuous.png').resolve()}")
    print(f"eval_loss_plot={(args.output_dir / 'eval_loss_by_epoch_with_continuous.png').resolve()}")
    print(f"eval_accuracy_plot={(args.output_dir / 'eval_accuracy_by_epoch_with_continuous.png').resolve()}")
    print(f"time_plot={(args.output_dir / 'time_by_epoch_with_continuous.png').resolve()}")
    print(f"ratio_plot={(args.output_dir / 'ratio_by_epoch_with_continuous.png').resolve()}")


if __name__ == "__main__":
    main()
