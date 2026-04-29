#!/usr/bin/env python3
"""Create intermediate GPT-2/C4 comparison plots and tables from run folders."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze GPT-2/C4 smoke or partial runs.")
    p.add_argument("--runs-root", type=str, default="~/remote_lab/runs")
    p.add_argument("--pattern", type=str, default="gpt2_c4_*")
    p.add_argument("--output-dir", type=str, default="~/remote_lab/reports/gpt2_c4_comparison")
    return p.parse_args()


def load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def label_for_run(run_name: str, meta: dict[str, Any] | None) -> str:
    if meta is None:
        return run_name
    variant = meta.get("variant", run_name)
    kwargs = meta.get("variant_kwargs", {}) or {}
    if variant == "baseline":
        return "Baseline"
    if variant == "fullyshared":
        return "FullShr"
    if variant == "partialshared":
        return f"PartShr{kwargs.get('shared_dim', '')}"
    if variant == "lowrank":
        return f"LowRank{kwargs.get('rank', '')}"
    if variant == "bbt":
        return f"BBT r={kwargs.get('rank', '')}"
    if variant == "bmb":
        return f"BMB r={kwargs.get('rank', '')}"
    if variant == "bmbuv":
        return f"BMB-UV r={kwargs.get('rank', '')},s={kwargs.get('factor_rank', kwargs.get('rank', ''))}"
    return run_name


def latest_metric(rows: list[dict[str, Any]], key: str) -> Any:
    for row in reversed(rows):
        if key in row:
            return row[key]
    return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main() -> int:
    args = parse_args()
    runs_root = os.path.expanduser(args.runs_root)
    output_dir = os.path.expanduser(args.output_dir)
    ensure_dir(output_dir)

    run_dirs = sorted(glob.glob(os.path.join(runs_root, args.pattern)))
    if not run_dirs:
        raise SystemExit(f"No run directories matched {os.path.join(runs_root, args.pattern)}")

    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        meta = load_json(os.path.join(run_dir, "run_metadata.json"))
        summary = load_json(os.path.join(run_dir, "summary.json"))
        train_rows = load_jsonl(os.path.join(run_dir, "metrics_train.jsonl"))
        eval_rows = load_jsonl(os.path.join(run_dir, "metrics_eval.jsonl"))
        run_name = os.path.basename(run_dir)
        label = label_for_run(run_name, meta or summary)
        runs.append(
            {
                "run_dir": run_dir,
                "run_name": run_name,
                "label": label,
                "meta": meta or {},
                "summary": summary or {},
                "train_rows": train_rows,
                "eval_rows": eval_rows,
            }
        )

    # Train loss curve
    plt.figure(figsize=(10, 6))
    plotted = False
    for run in runs:
        xs = [r["step"] for r in run["train_rows"] if "loss" in r]
        ys = [r["loss"] for r in run["train_rows"] if "loss" in r]
        if xs and ys:
            plotted = True
            plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.0, label=run["label"])
    if plotted:
        plt.xlabel("Step")
        plt.ylabel("Train loss")
        plt.title("GPT-2/C4 Training Loss")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig_train_loss.png"), dpi=180)
    plt.close()

    # Eval loss curve
    plt.figure(figsize=(10, 6))
    plotted = False
    for run in runs:
        xs = [r["step"] for r in run["eval_rows"] if "eval_loss" in r]
        ys = [r["eval_loss"] for r in run["eval_rows"] if "eval_loss" in r]
        if xs and ys:
            plotted = True
            plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.0, label=run["label"])
    if plotted:
        plt.xlabel("Step")
        plt.ylabel("Eval loss")
        plt.title("GPT-2/C4 Eval Loss")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig_eval_loss.png"), dpi=180)
    plt.close()

    # Efficiency bar figure
    labels = [run["label"] for run in runs]
    step_sec = []
    tok_sec = []
    for run in runs:
        summary = run["summary"]
        eff = summary.get("efficiency_summary", {})
        step = eff.get("step_wall_clock_sec")
        tok = eff.get("train_tokens_per_sec")
        if step is None:
            step = latest_metric(run["train_rows"], "train_runtime")
            if step is not None:
                step = float(step) / max(1, latest_metric(run["train_rows"], "step") or 1)
        step_sec.append(step if step is not None else float("nan"))
        tok_sec.append(tok if tok is not None else float("nan"))

    if any(not math.isnan(x) for x in step_sec) or any(not math.isnan(x) for x in tok_sec):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(labels, step_sec)
        axes[0].set_ylabel("sec / step")
        axes[0].set_title("Step Wall-Clock")
        axes[0].tick_params(axis="x", rotation=35)
        axes[1].bar(labels, tok_sec)
        axes[1].set_ylabel("tokens / sec")
        axes[1].set_title("Training Throughput")
        axes[1].tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig_efficiency.png"), dpi=180)
        plt.close(fig)

    # Summary table
    rows: list[dict[str, Any]] = []
    for run in runs:
        meta = run["meta"]
        summary = run["summary"]
        att = (summary.get("attention_theory_summary") or meta.get("attention_theory_summary") or {})
        eff = summary.get("efficiency_summary", {})
        eval_loss = summary.get("eval_loss", latest_metric(run["eval_rows"], "eval_loss"))
        ppl = summary.get("perplexity")
        if ppl is None and eval_loss is not None:
            ppl = math.exp(float(eval_loss))
        rows.append(
            {
                "run": run["run_name"],
                "label": run["label"],
                "variant": (summary.get("variant") or meta.get("variant")),
                "eval_loss": eval_loss,
                "perplexity": ppl,
                "step_wall_clock_sec": eff.get("step_wall_clock_sec"),
                "train_tokens_per_sec": eff.get("train_tokens_per_sec"),
                "peak_cuda_memory_allocated_mb": eff.get("peak_cuda_memory_allocated_mb"),
                "total_qk_params": att.get("total_qk_params"),
                "total_attention_params": att.get("total_attention_params"),
                "per_example_qk_flops_total": att.get("per_example_qk_flops_total"),
                "per_example_attention_flops_total": att.get("per_example_attention_flops_total"),
            }
        )

    csv_path = os.path.join(output_dir, "table_gpt2_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = os.path.join(output_dir, "table_gpt2_comparison.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & Eval loss & PPL & Sec/step & Tok/s & QK params & Attn params\\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            def fmt(x, digits=3):
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    return "--"
                if isinstance(x, (int,)):
                    return f"{x:,}"
                return f"{float(x):.{digits}f}"

            f.write(
                f"{row['label']} & {fmt(row['eval_loss'])} & {fmt(row['perplexity'], 1)} & "
                f"{fmt(row['step_wall_clock_sec'], 2)} & {fmt(row['train_tokens_per_sec'], 1)} & "
                f"{fmt(row['total_qk_params'], 0)} & {fmt(row['total_attention_params'], 0)}\\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print("[done]")
    print(json.dumps({"runs_analyzed": len(runs), "output_dir": output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
