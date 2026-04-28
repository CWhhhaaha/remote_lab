#!/usr/bin/env python3
"""Draw total comparison figure for all ImageNet 30-epoch experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("reports/imagenet_total_comparison")
FIG_ROOT = ROOT / "figures"
TABLE_ROOT = ROOT / "tables"

# CORRECT QK parameters computed from actual source code (ViT-B/16, 12 layers)
# NOTE: PartialShared bug in summarize function was fixed here.
QK_PARAMS = {
    "Baseline": 14155776,              # 2*d*d*12
    "BMB $r$=64": 1228800,             # (d*r + (H+1)*r*r)*12
    "BBT $r$=64": 589824,              # d*r*12
    "BMB-UV $r$=64,$s$=64": 1769472,  # (d*r + 2*H*r*s)*12
    "FullyShared": 7077888,            # d*d*12
    "LowRank $r$=32": 7667712,         # 2*H*(d*r + r*d_h)*12
    "BMB-UV $r$=32,$s$=32": 589824,   # (d*r + 2*H*r*s)*12
    "PartialShared $r$=48": 8847360,   # (d*H*r_s + 2*d*H*r_p)*12
}

RUNS = {
    "Baseline": {
        "dir": "runs/imagenet1k_vit12_baseline_recipe_30ep_gpu5",
        "color": "#4C72B0",
        "marker": "o",
        "short": "Base",
    },
    "BMB $r$=64": {
        "dir": "runs/imagenet1k_vit12_bmb_recipe_r64_30ep_gpu7",
        "color": "#DD8452",
        "marker": "s",
        "short": "BMB",
    },
    "BBT $r$=64": {
        "dir": "runs/imagenet1k_vit12_bbt_recipe_r64_30ep_gpu0",
        "color": "#55A868",
        "marker": "^",
        "short": "BBT",
    },
    "BMB-UV $r$=64,$s$=64": {
        "dir": "runs/imagenet1k_vit12_bmbuv_recipe_r64_s64_30ep_gpu1",
        "color": "#C44E52",
        "marker": "v",
        "short": "BMB-UV64",
    },
    "FullyShared": {
        "dir": "runs/fs",
        "color": "#8172B2",
        "marker": "D",
        "short": "FullShr",
    },
    "LowRank $r$=32": {
        "dir": "runs/lr32",
        "color": "#937860",
        "marker": "P",
        "short": "LowR32",
    },
    "BMB-UV $r$=32,$s$=32": {
        "dir": "runs/bmbuv",
        "color": "#DA8BC3",
        "marker": "X",
        "short": "BMB-UV32",
    },
    "PartialShared $r$=48": {
        "dir": "runs/ps48",
        "color": "#8C8C8C",
        "marker": "*",
        "short": "PartShr",
    },
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
            "legend.fontsize": 8,
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


def load_run(run_dir: str) -> dict:
    d = Path(run_dir)
    metrics = load_json(d / "metrics.json")
    epoch_path = d / "analysis" / "epoch_metrics.json"
    epoch_metrics = load_json(epoch_path) if epoch_path.exists() else []
    return {"metrics": metrics, "epoch_metrics": epoch_metrics}


def metric_series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in rows:
        e = row.get("epoch")
        v = row.get(key)
        if e is None or e == 0 or v is None:
            continue
        xs.append(float(e))
        ys.append(float(v))
    return np.asarray(xs), np.asarray(ys)


def plot_accuracy_vs_qk_params() -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, meta in RUNS.items():
        run = load_run(meta["dir"])
        m = run["metrics"]
        x = QK_PARAMS[name] / 1e6
        y = m["final_eval_accuracy"] * 100.0
        ax.scatter(x, y, color=meta["color"], marker=meta["marker"], s=150, zorder=3)
        ax.annotate(meta["short"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("QK Path Parameters (M)")
    ax.set_ylabel("Final Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs. QK Parameter Size Trade-off (ImageNet-1K, 30 Epochs)")
    fig.savefig(FIG_ROOT / "figA_accuracy_vs_qk_params.pdf", bbox_inches="tight")
    fig.savefig(FIG_ROOT / "figA_accuracy_vs_qk_params.png", bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    panels = [
        ("eval_accuracy", "Validation Top-1", "Top-1 Accuracy (%)", lambda y: 100.0 * y),
        ("eval_top5_accuracy", "Validation Top-5", "Top-5 Accuracy (%)", lambda y: 100.0 * y),
        ("eval_loss", "Validation Loss", "Cross-Entropy Loss", lambda y: y),
        ("avg_total_loss", "Training Loss", "Training Loss", lambda y: y),
    ]
    for ax, (key, title, ylabel, transform) in zip(axes.flat, panels):
        for name, meta in RUNS.items():
            run = load_run(meta["dir"])
            xs, ys = metric_series(run["epoch_metrics"], key)
            if len(xs) > 0:
                ax.plot(xs, transform(ys), color=meta["color"], linewidth=1.8, label=meta["short"])
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


def make_main_table() -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & QK Params (M) $\downarrow$ & Attn Params (M) $\downarrow$ & Final Top-1 (\%) $\uparrow$ & Best Top-1 (\%) $\uparrow$ & Final Top-5 (\%) $\uparrow$ & Final Loss $\downarrow$ \\",
        r"\midrule",
    ]
    baseline = None
    for name, meta in RUNS.items():
        run = load_run(meta["dir"])
        m = run["metrics"]
        p = m["parameter_summary"]
        if name == "Baseline":
            baseline = (m["final_eval_accuracy"], m["final_eval_loss"])
        delta_top1 = ""
        delta_loss = ""
        if baseline and name != "Baseline":
            d = (m["final_eval_accuracy"] - baseline[0]) * 100.0
            delta_top1 = f" ({d:+.2f})"
            dloss = m["final_eval_loss"] - baseline[1]
            delta_loss = f" ({dloss:+.4f})"
        lines.append(
            f"{meta['short']} & {QK_PARAMS[name]/1e6:.2f} & {p['attention_params']/1e6:.2f} & "
            f"{m['final_eval_accuracy']*100:.2f}{delta_top1} & {m['best_eval_accuracy']*100:.2f} & "
            f"{m['final_eval_top5_accuracy']*100:.2f} & {m['final_eval_loss']:.4f}{delta_loss} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(TABLE_ROOT / "table_total_comparison.tex", "\n".join(lines))



    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    print("Plotting Figure A: Accuracy vs. QK Parameters ...")
    plot_accuracy_vs_qk_params()
    print("Plotting Figure B: Learning Curves ...")
    plot_learning_curves()
    print("Writing Figure C: LaTeX Tables ...")
    make_main_table()
    print("Plotting Figure D: Rank Analysis ...")


def plot_rank_analysis() -> None:
    rank_dir = Path("reports/imagenet_total_comparison/rank_analysis")
    if not (rank_dir / "all_ranks.json").exists():
        print("Rank analysis data not found. Skipping rank plot.")
        return

    with open(rank_dir / "all_ranks.json") as f:
        all_ranks = json.load(f)

    bmb_diagnostics = []
    bmb_metrics_path = Path("runs/imagenet1k_vit12_bmb_recipe_r64_30ep_gpu7/metrics.json")
    if bmb_metrics_path.exists():
        bmb_m = json.load(bmb_metrics_path.open())
        bmb_diagnostics = bmb_m.get("bmb_diagnostics", {}).get("per_layer", [])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    ax = axes[0]
    for name, meta in RUNS.items():
        if name not in all_ranks:
            continue
        layers = [r["layer"] for r in all_ranks[name]]
        eranks = [r.get("effective_rank_qk_mean", 0) for r in all_ranks[name]]
        ax.plot(layers, eranks, color=meta["color"], marker=meta["marker"],
                linewidth=2.0, markersize=5, label=meta["short"])
    if bmb_diagnostics:
        layers = [r["layer"] for r in bmb_diagnostics]
        eranks = [r.get("effective_rank_M_mean", 0) for r in bmb_diagnostics]
        ax.plot(layers, eranks, color="#DD8452", marker="s",
                linewidth=2.0, markersize=5, label="BMB")
    ax.set_title("Effective Rank of QK Kernel")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Effective Rank")
    ax.set_xticks(range(12))
    ax.legend(loc="best", frameon=False, ncol=2)

    ax = axes[1]
    for name, meta in RUNS.items():
        if name not in all_ranks:
            continue
        layers = [r["layer"] for r in all_ranks[name]]
        diversity = None
        for key in ("head_query_diversity", "head_q_diversity", "head_u_diversity"):
            if key in all_ranks[name][0]:
                diversity = [r[key] for r in all_ranks[name]]
                break
        if diversity is None:
            continue
        ax.plot(layers, diversity, color=meta["color"], marker=meta["marker"],
                linewidth=2.0, markersize=5, label=meta["short"])
    if bmb_diagnostics:
        layers = [r["layer"] for r in bmb_diagnostics]
        divs = [r.get("head_M_cosine_similarity_mean", 0) for r in bmb_diagnostics]
        ax.plot(layers, divs, color="#DD8452", marker="s",
                linewidth=2.0, markersize=5, label="BMB")
    ax.set_title("Head Diversity (Cosine Similarity)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_xticks(range(12))
    ax.legend(loc="best", frameon=False, ncol=2)

    fig.suptitle("Attention Rank and Diversity Analysis", fontsize=12, y=1.02)
    fig.savefig(FIG_ROOT / "figD_rank_analysis.pdf", bbox_inches="tight")
    fig.savefig(FIG_ROOT / "figD_rank_analysis.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    setup_plotting()

    print("Plotting Figure A: Accuracy vs. QK Parameters ...")
    plot_accuracy_vs_qk_params()

    print("Plotting Figure B: Learning Curves ...")
    plot_learning_curves()

    print("Writing Figure C: LaTeX Tables ...")
    make_main_table()

    print("Plotting Figure D: Rank Analysis ...")
    plot_rank_analysis()

    print(f"\nDone! Output in {ROOT.resolve()}/")
    print(f"  Figures: {FIG_ROOT}")
    print(f"  Tables:  {TABLE_ROOT}")


if __name__ == "__main__":
    main()
