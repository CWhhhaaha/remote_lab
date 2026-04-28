#!/usr/bin/env python3
"""Find all ImageNet 30-epoch experiment results under runs/."""

import json
from pathlib import Path

runs_dir = Path("runs")
results = []

for run_dir in sorted(runs_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    metrics_path = run_dir / "metrics.json"
    epoch_path = run_dir / "analysis" / "epoch_metrics.json"
    if not metrics_path.exists():
        continue
    
    with open(metrics_path) as f:
        m = json.load(f)
    
    exp_name = m.get("experiment_name", run_dir.name)
    variant = m.get("attention_theory_summary", {}).get("attention_variant", "unknown")
    final_top1 = m.get("final_eval_accuracy", 0) * 100
    best_top1 = m.get("best_eval_accuracy", 0) * 100
    total_params = m.get("parameter_summary", {}).get("total_params", 0)
    attn_params = m.get("parameter_summary", {}).get("attention_params", 0)
    
    results.append({
        "run_dir": str(run_dir),
        "experiment_name": exp_name,
        "variant": variant,
        "final_top1": final_top1,
        "best_top1": best_top1,
        "total_params": total_params,
        "attention_params": attn_params,
        "has_epoch_metrics": epoch_path.exists(),
    })

print(f"Found {len(results)} experiments with metrics.json:\n")
print(f"{'Run Dir':<50} {'Variant':<25} {'Final Top-1':<12} {'Best Top-1':<12} {'Total Params':<14} {'Epoch Data'}")
print("-" * 130)
for r in results:
    print(f"{r['run_dir']:<50} {r['variant']:<25} {r['final_top1']:>10.2f}% {r['best_top1']:>10.2f}% {r['total_params']:>12,} {'YES' if r['has_epoch_metrics'] else 'NO'}")
