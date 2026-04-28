#!/usr/bin/env python3
"""Draw a robust total-comparison report for ImageNet 30-epoch experiments.

This script is intentionally defensive:
- it discovers run directories from several likely report/raw locations;
- it skips missing methods instead of crashing;
- it only compares diagnostics that share a compatible definition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "reports" / "imagenet_total_comparison"
FIG_ROOT = REPORT_ROOT / "figures"
TABLE_ROOT = REPORT_ROOT / "tables"


@dataclass(frozen=True)
class RunSpec:
    label: str
    short: str
    rel_dir: str
    color: str
    marker: str
    linestyle: str = "-"
    diagnostics_role: str | None = None
    qk_params_fallback: int | None = None


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        label="Baseline",
        short="Base",
        rel_dir="runs/imagenet1k_vit12_baseline_recipe_30ep_gpu5",
        color="#4C72B0",
        marker="o",
        linestyle="-",
        qk_params_fallback=14155776,
    ),
    RunSpec(
        label="BMB $r$=64",
        short="BMB",
        rel_dir="runs/imagenet1k_vit12_bmb_recipe_r64_30ep_gpu7",
        color="#DD8452",
        marker="s",
        linestyle="-",
        diagnostics_role="bmb",
        qk_params_fallback=1228800,
    ),
    RunSpec(
        label="BBT $r$=64",
        short="BBT",
        rel_dir="runs/imagenet1k_vit12_bbt_recipe_r64_30ep_gpu0",
        color="#55A868",
        marker="o",
        linestyle="--",
        diagnostics_role="operator_identity",
        qk_params_fallback=589824,
    ),
    RunSpec(
        label="BMB-UV $r$=64,$s$=64",
        short="BMB-UV64",
        rel_dir="runs/imagenet1k_vit12_bmbuv_recipe_r64_s64_30ep_gpu1",
        color="#C44E52",
        marker="o",
        linestyle="-.",
        diagnostics_role="uv",
        qk_params_fallback=1769472,
    ),
    RunSpec(
        label="FullyShared",
        short="FullShr",
        rel_dir="runs/fs",
        color="#8172B2",
        marker="D",
        linestyle="-",
        qk_params_fallback=7077888,
    ),
    RunSpec(
        label="LowRank $r$=32",
        short="LowR32",
        rel_dir="runs/lr32",
        color="#937860",
        marker="P",
        linestyle="-",
        qk_params_fallback=7667712,
    ),
    RunSpec(
        label="BMB-UV $r$=32,$s$=32",
        short="BMB-UV32",
        rel_dir="runs/bmbuv",
        color="#DA8BC3",
        marker="X",
        linestyle="-",
        diagnostics_role="uv",
        qk_params_fallback=589824,
    ),
    RunSpec(
        label="PartialShared $r$=48",
        short="PartShr",
        rel_dir="runs/ps48",
        color="#8C8C8C",
        marker="*",
        linestyle="-",
        qk_params_fallback=8847360,
    ),
)


RAW_SEARCH_ROOTS = (
    REPO_ROOT,
    REPO_ROOT / "reports" / "imagenet_total_comparison" / "raw",
    REPO_ROOT / "reports" / "imagenet_triplet_compare" / "raw",
    REPO_ROOT / "reports" / "imagenet_30ep_bmb_vs_baseline" / "raw",
)


# Some methods had inconsistent QK summaries across earlier metrics exports.
# For scatter-plot fairness, we trust these explicit architecture-level totals.
QK_PARAMS_CORRECTED = {
    "FullyShared": 7077888,
    "LowRank $r$=32": 7667712,
    "PartialShared $r$=48": 8847360,
}


def setup_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def resolve_run_dir(spec: RunSpec) -> Path | None:
    for root in RAW_SEARCH_ROOTS:
        candidate = root / spec.rel_dir
        if (candidate / "metrics.json").exists():
            return candidate
    return None


def qk_param_total(metrics: dict, spec: RunSpec) -> int:
    if spec.label in QK_PARAMS_CORRECTED:
        return int(QK_PARAMS_CORRECTED[spec.label])
    theory = metrics.get("attention_theory_summary", {})
    params = metrics.get("parameter_summary", {})
    for key in ("qk_weight_params_total", "qk_score_params"):
        if key in theory and theory[key]:
            return int(theory[key])
        if key in params and params[key]:
            return int(params[key])
    if "qk_score_params" in params and params["qk_score_params"]:
        return int(params["qk_score_params"])
    if spec.qk_params_fallback is not None:
        return int(spec.qk_params_fallback)
    raise KeyError(f"Unable to determine QK params for {spec.label}")


def load_run(spec: RunSpec) -> dict | None:
    run_dir = resolve_run_dir(spec)
    if run_dir is None:
        return None
    metrics = load_json(run_dir / "metrics.json")
    epoch_path = run_dir / "analysis" / "epoch_metrics.json"
    epoch_metrics = load_json(epoch_path) if epoch_path.exists() else []
    return {
        "spec": spec,
        "dir": run_dir,
        "metrics": metrics,
        "epoch_metrics": epoch_metrics,
        "qk_params_total": qk_param_total(metrics, spec),
    }


def collect_runs() -> list[dict]:
    runs: list[dict] = []
    for spec in RUN_SPECS:
        run = load_run(spec)
        if run is not None:
            runs.append(run)
        else:
            print(f"[skip] {spec.label}: metrics.json not found")
    return runs


def metric_series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in rows:
        epoch = row.get("epoch")
        value = row.get(key)
        if epoch is None or epoch == 0 or value is None:
            continue
        xs.append(float(epoch))
        ys.append(float(value))
    return np.asarray(xs), np.asarray(ys)


def plot_accuracy_vs_qk_params(runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    for run in runs:
        spec = run["spec"]
        metrics = run["metrics"]
        x = run["qk_params_total"] / 1e6
        y = metrics["final_eval_accuracy"] * 100.0
        ax.scatter(x, y, color=spec.color, marker=spec.marker, s=140, zorder=3)
        ax.annotate(spec.short, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("QK Path Parameters (M)")
    ax.set_ylabel("Final Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs. QK Parameter Size Trade-off")
    fig.savefig(FIG_ROOT / "figA_accuracy_vs_qk_params.pdf", bbox_inches="tight")
    fig.savefig(FIG_ROOT / "figA_accuracy_vs_qk_params.png", bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(runs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    panels = [
        ("eval_accuracy", "Validation Top-1", "Top-1 Accuracy (%)", lambda y: 100.0 * y),
        ("eval_top5_accuracy", "Validation Top-5", "Top-5 Accuracy (%)", lambda y: 100.0 * y),
        ("eval_loss", "Validation Loss", "Cross-Entropy Loss", lambda y: y),
        ("avg_total_loss", "Training Loss", "Training Loss", lambda y: y),
    ]
    for ax, (key, title, ylabel, transform) in zip(axes.flat, panels):
        for run in runs:
            xs, ys = metric_series(run["epoch_metrics"], key)
            if len(xs) == 0:
                continue
            spec = run["spec"]
            ax.plot(
                xs,
                transform(ys),
                color=spec.color,
                linestyle=spec.linestyle,
                marker=spec.marker,
                markersize=2.4,
                markevery=3,
                linewidth=1.35,
                alpha=0.95,
                label=spec.short,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 30)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    axes[0, 0].legend(loc="lower right", frameon=False, ncol=2)
    fig.suptitle("ImageNet-1K 30-Epoch Training Curves", fontsize=12, y=1.01)
    fig.savefig(FIG_ROOT / "figB_learning_curves.pdf", bbox_inches="tight")
    fig.savefig(FIG_ROOT / "figB_learning_curves.png", bbox_inches="tight")
    plt.close(fig)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def make_main_table(runs: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & QK Params (M) $\downarrow$ & Attn Params (M) $\downarrow$ & Final Top-1 (\%) $\uparrow$ & Best Top-1 (\%) $\uparrow$ & Final Top-5 (\%) $\uparrow$ & Final Loss $\downarrow$ \\",
        r"\midrule",
    ]
    baseline = next((run for run in runs if run["spec"].label == "Baseline"), None)
    baseline_top1 = baseline["metrics"]["final_eval_accuracy"] if baseline else None
    baseline_loss = baseline["metrics"]["final_eval_loss"] if baseline else None
    for run in runs:
        spec = run["spec"]
        metrics = run["metrics"]
        params = metrics.get("parameter_summary", {})
        delta_top1 = ""
        delta_loss = ""
        if baseline_top1 is not None and spec.label != "Baseline":
            delta_top1 = f" ({(metrics['final_eval_accuracy'] - baseline_top1) * 100:+.2f})"
        if baseline_loss is not None and spec.label != "Baseline":
            delta_loss = f" ({metrics['final_eval_loss'] - baseline_loss:+.4f})"
        lines.append(
            f"{spec.short} & {run['qk_params_total']/1e6:.2f} & {params.get('attention_params', 0)/1e6:.2f} & "
            f"{metrics['final_eval_accuracy']*100:.2f}{delta_top1} & {metrics['best_eval_accuracy']*100:.2f} & "
            f"{metrics['final_eval_top5_accuracy']*100:.2f} & {metrics['final_eval_loss']:.4f}{delta_loss} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(TABLE_ROOT / "table_total_comparison.tex", "\n".join(lines))


def normalize_layer_index(rows: list[dict]) -> list[int]:
    layers = [int(row["layer"]) for row in rows]
    return layers if min(layers) >= 1 else [layer + 1 for layer in layers]


def extract_diagnostics(run: dict) -> dict | None:
    metrics = run["metrics"]
    diag = metrics.get("bmb_diagnostics") or {}
    rows = diag.get("per_layer") or []
    if rows:
        return {
            "layers": normalize_layer_index(rows),
            "basis_rank": [float(row["effective_rank_B"]) for row in rows],
            "operator_rank": [float(row["effective_rank_M_mean"]) for row in rows],
            "diversity": [float(row["head_M_cosine_similarity_mean"]) for row in rows],
        }

    rank_json = REPORT_ROOT / "rank_analysis" / "all_ranks.json"
    if not rank_json.exists():
        return None
    all_ranks = load_json(rank_json)
    label = run["spec"].label
    if label not in all_ranks:
        return None
    rows = all_ranks[label]
    if not rows:
        return None
    required = ("effective_rank_basis", "effective_rank_qk_mean")
    if not all(key in rows[0] for key in required):
        return None
    diversity_key = next(
        (key for key in ("head_u_diversity", "head_q_diversity", "head_query_diversity") if key in rows[0]),
        None,
    )
    if diversity_key is None:
        return None
    return {
        "layers": normalize_layer_index(rows),
        "basis_rank": [float(row["effective_rank_basis"]) for row in rows],
        "operator_rank": [float(row["effective_rank_qk_mean"]) for row in rows],
        "diversity": [float(row[diversity_key]) for row in rows],
    }


def diagnostic_runs(runs: list[dict]) -> list[dict]:
    preferred = ["BBT $r$=64", "BMB-UV $r$=64,$s$=64", "BMB $r$=64", "BMB-UV $r$=32,$s$=32"]
    order = {label: idx for idx, label in enumerate(preferred)}
    chosen = []
    for run in runs:
        diag = extract_diagnostics(run)
        if diag is None:
            continue
        chosen.append({**run, "diagnostics": diag})
    chosen.sort(key=lambda run: order.get(run["spec"].label, 999))
    return chosen


def plot_rank_analysis(runs: list[dict]) -> None:
    diag_runs = diagnostic_runs(runs)
    if not diag_runs:
        print("[skip] rank diagnostics: no compatible diagnostics found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4), constrained_layout=True)
    panels = [
        ("basis_rank", "Shared Basis Effective Rank", "erank($\\mathbf{B}$)"),
        ("operator_rank", "Head Interaction Effective Rank", "mean erank($\\mathbf{M}$)"),
        ("diversity", "Head Interaction Diversity", "mean head cosine"),
    ]

    for ax, (key, title, ylabel) in zip(axes, panels):
        for run in diag_runs:
            spec = run["spec"]
            diag = run["diagnostics"]
            ax.plot(
                diag["layers"],
                diag[key],
                color=spec.color,
                linestyle=spec.linestyle,
                marker=spec.marker,
                linewidth=2.1,
                markersize=4.5,
                label=spec.short,
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 13))
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("Diagnostic Comparison of Shared Basis and Head Interactions", fontsize=14, y=1.03)
    fig.savefig(FIG_ROOT / "figD_rank_analysis.pdf", bbox_inches="tight")
    fig.savefig(FIG_ROOT / "figD_rank_analysis.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    setup_plotting()

    runs = collect_runs()
    if not runs:
        raise SystemExit("No available runs found. Please sync metrics.json into the repo or reports/raw tree.")

    print(f"[info] found {len(runs)} runs:")
    for run in runs:
        print(f"  - {run['spec'].label}: {run['dir']}")

    print("Plotting Figure A: Accuracy vs. QK Parameters ...")
    plot_accuracy_vs_qk_params(runs)
    print("Plotting Figure B: Learning Curves ...")
    plot_learning_curves(runs)
    print("Writing comparison table ...")
    make_main_table(runs)
    print("Plotting Figure D: Rank Analysis ...")
    plot_rank_analysis(runs)

    print(f"\nDone! Output in {REPORT_ROOT.resolve()}/")
    print(f"  Figures: {FIG_ROOT}")
    print(f"  Tables:  {TABLE_ROOT}")


if __name__ == "__main__":
    main()
