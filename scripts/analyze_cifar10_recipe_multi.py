#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple CIFAR-10 recipe runs and export combined summary plots."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification in the form LABEL=/abs/path/to/run_dir. Repeat for multiple runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where analysis artifacts will be written.")
    parser.add_argument("--title-prefix", default="CIFAR-10 recipe comparison", help="Prefix used in plot titles.")
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid run specification: {spec}. Expected LABEL=/path/to/run")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    run_dir = Path(raw_path).expanduser().resolve()
    if not label:
        raise ValueError(f"Empty label in run specification: {spec}")
    return label, run_dir


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
        "final_eval_loss_raw": metrics.get("final_eval_loss_raw"),
        "final_eval_accuracy_raw": metrics.get("final_eval_accuracy_raw"),
        "final_eval_loss_ema": metrics.get("final_eval_loss_ema"),
        "final_eval_accuracy_ema": metrics.get("final_eval_accuracy_ema"),
        "reported_training_time_sec": metrics.get("reported_training_time_sec"),
        "reported_training_flops": metrics.get("reported_training_flops"),
        "ema_enabled": metrics.get("ema_enabled"),
        "final_layer_ratios": final_epoch.get("layer_asymmetry_ratio"),
        "regularization_epochs": sum(1 for row in epoch_metrics if row.get("regularization_active")),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# CIFAR-10 Recipe Multi-Run Analysis", ""]
    for label, summary in payload["runs"].items():
        lines.extend(
            [
                f"## {label}",
                f"- run_dir: `{summary['run_dir']}`",
                f"- experiment_name: `{summary['experiment_name']}`",
                f"- epochs: `{summary['epochs']}`",
                f"- final_train_loss: `{summary['final_train_loss']}`",
                f"- final_eval_loss: `{summary['final_eval_loss']}`",
                f"- final_eval_accuracy: `{summary['final_eval_accuracy']}`",
                f"- final_eval_loss_raw: `{summary['final_eval_loss_raw']}`",
                f"- final_eval_accuracy_raw: `{summary['final_eval_accuracy_raw']}`",
                f"- final_eval_loss_ema: `{summary['final_eval_loss_ema']}`",
                f"- final_eval_accuracy_ema: `{summary['final_eval_accuracy_ema']}`",
                f"- reported_training_time_sec: `{summary['reported_training_time_sec']}`",
                f"- reported_training_flops: `{summary['reported_training_flops']}`",
                f"- regularization_epochs: `{summary['regularization_epochs']}`",
                f"- final_layer_ratios: `{summary['final_layer_ratios']}`",
                "",
            ]
        )
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


def plot_dual_metric_curves(
    runs: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
    title: str,
    ylabel: str,
    raw_key: str,
    ema_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for label, rows in runs:
        raw_points = [(row["epoch"], row.get(raw_key)) for row in rows if row.get(raw_key) is not None]
        ema_points = [(row["epoch"], row.get(ema_key)) for row in rows if row.get(ema_key) is not None]
        if raw_points:
            ax.plot([x for x, _ in raw_points], [y for _, y in raw_points], label=f"{label} raw", linewidth=2.0)
        if ema_points:
            ax.plot(
                [x for x, _ in ema_points],
                [y for _, y in ema_points],
                label=f"{label} EMA",
                linewidth=1.8,
                linestyle="--",
            )
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
    for label, rows in runs:
        ax.plot(
            [row["epoch"] for row in rows],
            [row["avg_total_loss"] for row in rows],
            label=f"{label} total",
            linewidth=2.0,
        )
    for label, rows in runs:
        ax.plot(
            [row["epoch"] for row in rows],
            [row["avg_task_loss"] for row in rows],
            label=f"{label} task",
            linewidth=1.3,
            linestyle="--",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix}: Loss by Epoch")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(ncol=2)
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
    parsed = [parse_run_spec(spec) for spec in args.run]
    loaded = [(label, load_run(run_dir)) for label, run_dir in parsed]

    payload = {
        "runs": {label: summarize_run(run) for label, run in loaded},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary_multi_recipe.json", payload)
    write_markdown(args.output_dir / "summary_multi_recipe.md", payload)

    epoch_runs = [(label, run["epoch_metrics"]) for label, run in loaded]
    ratio_runs = [(label, run["ratio_history"]) for label, run in loaded]

    plot_loss_curves(epoch_runs, args.output_dir / "loss_by_epoch_multi_recipe.png", args.title_prefix)
    plot_metric_curves(
        epoch_runs,
        args.output_dir / "eval_loss_by_epoch_multi_recipe.png",
        f"{args.title_prefix}: Eval Loss by Epoch",
        "Eval loss",
        "eval_loss",
    )
    plot_metric_curves(
        epoch_runs,
        args.output_dir / "eval_accuracy_by_epoch_multi_recipe.png",
        f"{args.title_prefix}: Eval Accuracy by Epoch",
        "Accuracy",
        "eval_accuracy",
    )
    plot_dual_metric_curves(
        epoch_runs,
        args.output_dir / "eval_loss_raw_vs_ema_multi_recipe.png",
        f"{args.title_prefix}: Eval Loss (Raw vs EMA)",
        "Eval loss",
        "eval_loss_raw",
        "eval_loss_ema",
    )
    plot_dual_metric_curves(
        epoch_runs,
        args.output_dir / "eval_accuracy_raw_vs_ema_multi_recipe.png",
        f"{args.title_prefix}: Eval Accuracy (Raw vs EMA)",
        "Accuracy",
        "eval_accuracy_raw",
        "eval_accuracy_ema",
    )
    plot_metric_curves(
        epoch_runs,
        args.output_dir / "time_by_epoch_multi_recipe.png",
        f"{args.title_prefix}: Train Time by Epoch",
        "Seconds",
        "training_time_sec",
    )
    plot_ratio_curves(ratio_runs, args.output_dir / "ratio_by_epoch_multi_recipe.png", args.title_prefix)

    print(f"analysis_dir={args.output_dir.resolve()}")
    print(f"summary_json={(args.output_dir / 'summary_multi_recipe.json').resolve()}")
    print(f"summary_md={(args.output_dir / 'summary_multi_recipe.md').resolve()}")
    print(f"loss_plot={(args.output_dir / 'loss_by_epoch_multi_recipe.png').resolve()}")
    print(f"eval_loss_plot={(args.output_dir / 'eval_loss_by_epoch_multi_recipe.png').resolve()}")
    print(f"eval_accuracy_plot={(args.output_dir / 'eval_accuracy_by_epoch_multi_recipe.png').resolve()}")
    print(f"eval_loss_raw_ema_plot={(args.output_dir / 'eval_loss_raw_vs_ema_multi_recipe.png').resolve()}")
    print(f"eval_accuracy_raw_ema_plot={(args.output_dir / 'eval_accuracy_raw_vs_ema_multi_recipe.png').resolve()}")
    print(f"time_plot={(args.output_dir / 'time_by_epoch_multi_recipe.png').resolve()}")
    print(f"ratio_plot={(args.output_dir / 'ratio_by_epoch_multi_recipe.png').resolve()}")


if __name__ == "__main__":
    main()
