#!/usr/bin/env python3
"""Extract key metrics from the 4 new experiments for plotting."""

import json
from pathlib import Path

RUNS = {
    "baseline": "runs/imagenet1k_vit12_baseline_recipe_30ep_gpu7",
    "fullyshared": "runs/fs",
    "lowrank_r32": "runs/lr32",
    "bmbuv_r32s32": "runs/bmbuv",
    "partialshared_r48": "runs/ps48",
}

print("=" * 60)
for name, run_dir in RUNS.items():
    metrics_path = Path(run_dir) / "metrics.json"
    epoch_path = Path(run_dir) / "analysis" / "epoch_metrics.json"
    
    if not metrics_path.exists():
        print(f"{name}: metrics.json NOT FOUND at {metrics_path}")
        continue
    
    with open(metrics_path) as f:
        m = json.load(f)
    
    p = m.get("parameter_summary", {})
    attn_theory = m.get("attention_theory_summary", {})
    
    print(f"\n=== {name} ===")
    print(f"  final_top1:    {m.get('final_eval_accuracy', 0)*100:.2f}%")
    print(f"  best_top1:     {m.get('best_eval_accuracy', 0)*100:.2f}%")
    print(f"  final_top5:    {m.get('final_eval_top5_accuracy', 0)*100:.2f}%")
    print(f"  final_loss:    {m.get('final_eval_loss', 0):.4f}")
    print(f"  train_img/s:   {m.get('mean_train_images_per_sec', 0):.1f}")
    print(f"  eval_img/s:    {m.get('mean_eval_images_per_sec', 0):.1f}")
    print(f"  peak_mem_mb:   {m.get('peak_cuda_memory_allocated_mb', 0):.1f}")
    print(f"  total_params:  {p.get('total_params', 0)}")
    print(f"  attn_params:   {p.get('attention_params', 0)}")
    print(f"  qk_reduction%: {attn_theory.get('qk_weight_param_reduction_pct', 0):.2f}")
    
    if epoch_path.exists():
        with open(epoch_path) as f:
            epochs = json.load(f)
        # Print epoch 1, 5, 10, 20, 30 top1
        key_epochs = [1, 5, 10, 20, 30]
        accs = []
        for row in epochs:
            e = row.get("epoch")
            if e in key_epochs:
                accs.append(f"ep{e}:{row.get('eval_accuracy',0)*100:.1f}")
        print(f"  top1_trace:    {' | '.join(accs)}")

print("\n" + "=" * 60)
