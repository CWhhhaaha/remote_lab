#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline, interval, and symmetric-init Jigsaw runs and export summary plots."
    )
    parser.add_argument("--baseline-run", type=Path, required=True, help="Run directory for baseline.")
    parser.add_argument("--interval-run", type=Path, required=True, help="Run directory for interval regularization.")
    parser.add_argument("--symm-run", type=Path, required=True, help="Run directory for symmetric initialization.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for analysis outputs.")
    parser.add_argument("--title-prefix", default="Jigsaw 4-layer", help="Prefix used in plot titles.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


EPOCH_SUMMARY_RE = re.compile(
    r"\[epoch_summary\]\s+experiment=(?P<experiment>\S+)\s+"
    r"epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+"
    r"total_loss=(?P<total_loss>\S+)\s+"
    r"task_loss=(?P<task_loss>\S+)\s+"
    r"reg_loss=(?P<reg_loss>\S+)\s+"
    r"reg_active=(?P<reg_active>\S+)\s+"
    r"lr=(?P<lr>\S+)\s+"
    r"train_sec=(?P<train_sec>\S+)\s+"
    r"(?:eval_sec=(?P<eval_sec>\S+)\s+eval_loss=(?P<eval_loss>\S+)\s+)?"
    r"analysis_sec=(?P<analysis_sec>\S+)\s+"
    r"ratios=\[(?P<ratios>.+)\]"
)


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, "n/a"):
        return None
    return float(value)


def parse_ratios(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        _, value = chunk.strip().split("=")
        values.append(float(value))
    return values


def load_run_from_log(run_dir: Path) -> dict[str, Any]:
    train_log = run_dir / "train.log"
    if not train_log.exists():
        raise FileNotFoundError(f"Missing run artifacts and train.log: {run_dir}")

    epoch_metrics: list[dict[str, Any]] = []
    ratio_history: list[dict[str, Any]] = []
    experiment_name: str | None = None
    total_epochs: int | None = None

    for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EPOCH_SUMMARY_RE.search(line)
        if match is None:
            continue

        experiment_name = match.group("experiment")
        total_epochs = int(match.group("total_epochs"))
        layer_ratios = [round(value, 8) for value in parse_ratios(match.group("ratios"))]
        epoch = int(match.group("epoch"))
        reg_loss = parse_optional_float(match.group("reg_loss"))
        eval_loss = parse_optional_float(match.group("eval_loss"))
        eval_time = parse_optional_float(match.group("eval_sec"))

        epoch_metrics.append(
            {
                "epoch": epoch,
                "regularization_active": match.group("reg_active") == "yes",
                "avg_task_loss": round(float(match.group("task_loss")), 8),
                "avg_total_loss": round(float(match.group("total_loss")), 8),
                "avg_reg_loss": round(reg_loss, 8) if reg_loss is not None else None,
                "training_time_sec": round(float(match.group("train_sec")), 6),
                "evaluation_time_sec": round(eval_time, 6) if eval_time is not None else None,
                "eval_loss": round(eval_loss, 8) if eval_loss is not None else None,
                "analysis_time_sec": round(float(match.group("analysis_sec")), 6),
                "learning_rate": float(match.group("lr")),
                "layer_asymmetry_ratio": layer_ratios,
            }
        )
        ratio_history.append(
            {
                "epoch": epoch,
                "layer_asymmetry_ratio": layer_ratios,
            }
        )

    if not epoch_metrics or experiment_name is None or total_epochs is None:
        raise ValueError(f"Could not parse epoch summaries from {train_log}")

    final_epoch = epoch_metrics[-1]
    metrics = {
        "experiment_name": experiment_name,
        "epochs": total_epochs,
        "final_train_loss": final_epoch["avg_total_loss"],
        "final_eval_loss": final_epoch.get("eval_loss"),
        "reported_training_time_sec": sum(row["training_time_sec"] for row in epoch_metrics),
        "training_time_sec": sum(row["training_time_sec"] for row in epoch_metrics),
        "evaluation_time_sec": sum(row["evaluation_time_sec"] or 0.0 for row in epoch_metrics),
        "analysis_time_sec": sum(row["analysis_time_sec"] for row in epoch_metrics),
        "reported_training_flops": None,
        "task_flops": None,
        "reg_flops": None,
        "analysis_flops": None,
    }
    return {
        "run_dir": str(run_dir.resolve()),
        "metrics": metrics,
        "epoch_metrics": epoch_metrics,
        "ratio_history": ratio_history,
    }


def load_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    epoch_metrics_path = run_dir / "analysis" / "epoch_metrics.json"
    ratio_history_path = run_dir / "analysis" / "layer_asymmetry_by_epoch.json"
    if metrics_path.exists() and epoch_metrics_path.exists() and ratio_history_path.exists():
        metrics = load_json(metrics_path)
        epoch_metrics = load_json(epoch_metrics_path)
        ratio_history = load_json(ratio_history_path)
        return {
            "run_dir": str(run_dir.resolve()),
            "metrics": metrics,
            "epoch_metrics": epoch_metrics,
            "ratio_history": ratio_history,
        }
    return load_run_from_log(run_dir)


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
        "evaluation_time_sec": metrics.get("evaluation_time_sec"),
        "analysis_time_sec": metrics.get("analysis_time_sec"),
        "reported_training_flops": metrics.get("reported_training_flops"),
        "task_flops": metrics.get("task_flops"),
        "reg_flops": metrics.get("reg_flops"),
        "analysis_flops": metrics.get("analysis_flops"),
        "final_layer_ratios": final_epoch.get("layer_asymmetry_ratio"),
        "final_epoch_eval_loss": final_epoch.get("eval_loss"),
        "regularization_epochs": sum(1 for row in epoch_metrics if row.get("regularization_active")),
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
    symm = payload["symm"]
    lines = [
        "# Jigsaw Triple Analysis",
        "",
        "## Baseline",
        f"- run_dir: `{baseline['run_dir']}`",
        f"- final_train_loss: `{baseline['final_train_loss']}`",
        f"- final_eval_loss: `{baseline['final_eval_loss']}`",
        f"- reported_training_time_sec: `{baseline['reported_training_time_sec']}`",
        f"- final_layer_ratios: `{baseline['final_layer_ratios']}`",
        "",
        "## Interval",
        f"- run_dir: `{interval['run_dir']}`",
        f"- final_train_loss: `{interval['final_train_loss']}`",
        f"- final_eval_loss: `{interval['final_eval_loss']}`",
        f"- reported_training_time_sec: `{interval['reported_training_time_sec']}`",
        f"- final_layer_ratios: `{interval['final_layer_ratios']}`",
        f"- regularization_epochs: `{interval['regularization_epochs']}`",
        "",
        "## Symm",
        f"- run_dir: `{symm['run_dir']}`",
        f"- final_train_loss: `{symm['final_train_loss']}`",
        f"- final_eval_loss: `{symm['final_eval_loss']}`",
        f"- reported_training_time_sec: `{symm['reported_training_time_sec']}`",
        f"- final_layer_ratios: `{symm['final_layer_ratios']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_curves(
    *,
    baseline_epochs: list[dict[str, Any]],
    interval_epochs: list[dict[str, Any]],
    symm_epochs: list[dict[str, Any]],
    output_path: Path,
    title: str,
    ylabel: str,
    key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for label, rows in [
        ("Baseline", baseline_epochs),
        ("Interval", interval_epochs),
        ("Symm", symm_epochs),
    ]:
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


def plot_ratio_curves(
    baseline_ratios: list[dict[str, Any]],
    interval_ratios: list[dict[str, Any]],
    symm_ratios: list[dict[str, Any]],
    output_path: Path,
    title_prefix: str,
) -> None:
    num_layers = len(baseline_ratios[0]["layer_asymmetry_ratio"])
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3.2 * num_layers), dpi=180, sharex=True)
    if num_layers == 1:
        axes = [axes]
    for layer_idx, ax in enumerate(axes):
        for label, rows in [
            ("Baseline", baseline_ratios),
            ("Interval", interval_ratios),
            ("Symm", symm_ratios),
        ]:
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
    symm_run = load_run(args.symm_run)

    payload = {
        "baseline": summarize_run(baseline_run),
        "interval": summarize_run(interval_run),
        "symm": summarize_run(symm_run),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary_with_symm.json", payload)
    write_markdown(args.output_dir / "summary_with_symm.md", payload)

    plot_metric_curves(
        baseline_epochs=baseline_run["epoch_metrics"],
        interval_epochs=interval_run["epoch_metrics"],
        symm_epochs=symm_run["epoch_metrics"],
        output_path=args.output_dir / "loss_by_epoch_with_symm.png",
        title=f"{args.title_prefix}: Loss by Epoch",
        ylabel="Loss",
        key="avg_total_loss",
    )
    plot_metric_curves(
        baseline_epochs=baseline_run["epoch_metrics"],
        interval_epochs=interval_run["epoch_metrics"],
        symm_epochs=symm_run["epoch_metrics"],
        output_path=args.output_dir / "eval_loss_by_epoch_with_symm.png",
        title=f"{args.title_prefix}: Eval Loss by Epoch",
        ylabel="Eval loss",
        key="eval_loss",
    )
    plot_metric_curves(
        baseline_epochs=baseline_run["epoch_metrics"],
        interval_epochs=interval_run["epoch_metrics"],
        symm_epochs=symm_run["epoch_metrics"],
        output_path=args.output_dir / "time_by_epoch_with_symm.png",
        title=f"{args.title_prefix}: Train Time by Epoch",
        ylabel="Seconds",
        key="training_time_sec",
    )
    plot_ratio_curves(
        baseline_ratios=baseline_run["ratio_history"],
        interval_ratios=interval_run["ratio_history"],
        symm_ratios=symm_run["ratio_history"],
        output_path=args.output_dir / "ratio_by_epoch_with_symm.png",
        title_prefix=args.title_prefix,
    )

    print(f"analysis_dir={args.output_dir.resolve()}")
    print(f"summary_json={(args.output_dir / 'summary_with_symm.json').resolve()}")
    print(f"summary_md={(args.output_dir / 'summary_with_symm.md').resolve()}")
    print(f"loss_plot={(args.output_dir / 'loss_by_epoch_with_symm.png').resolve()}")
    print(f"eval_plot={(args.output_dir / 'eval_loss_by_epoch_with_symm.png').resolve()}")
    print(f"time_plot={(args.output_dir / 'time_by_epoch_with_symm.png').resolve()}")
    print(f"ratio_plot={(args.output_dir / 'ratio_by_epoch_with_symm.png').resolve()}")


if __name__ == "__main__":
    main()
