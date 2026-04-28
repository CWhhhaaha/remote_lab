#!/usr/bin/env python3
"""Analyze effective rank and head diversity from trained model checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


def effective_rank(matrix: torch.Tensor, threshold: float = 0.01) -> int:
    try:
        _, s, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    except Exception:
        s = torch.linalg.svdvals(matrix.float())
    return int((s > threshold * s[0]).sum().item())


def head_diversity(mats: torch.Tensor) -> float:
    H = mats.shape[0]
    flat = mats.view(H, -1)
    norms = flat.norm(dim=1, keepdim=True)
    normed = flat / (norms + 1e-8)
    sim = normed @ normed.T
    mask = ~torch.eye(H, dtype=torch.bool)
    return float(sim[mask].mean().item())


def load_checkpoint(run_dir: str) -> dict:
    model_dir = Path(run_dir) / "model"
    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
            return load_file(str(safetensors_path))
        except ImportError:
            print("safetensors not installed, trying torch.load on .bin")
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def analyze_baseline(run_dir: str) -> list[dict]:
    ckpt = load_checkpoint(run_dir)
    results = []
    for i in range(12):
        q = ckpt[f"vit.encoder.layer.{i}.attention.attention.query.weight"]
        k = ckpt[f"vit.encoder.layer.{i}.attention.attention.key.weight"]
        q_heads = q.view(12, 64, 768)
        k_heads = k.view(12, 64, 768)
        eranks = []
        for h in range(12):
            kernel = q_heads[h].T @ k_heads[h]
            eranks.append(effective_rank(kernel))
        results.append({
            "layer": i,
            "effective_rank_qk_mean": float(np.mean(eranks)),
            "head_query_diversity": head_diversity(q_heads),
            "head_key_diversity": head_diversity(k_heads),
        })
    return results


def analyze_fullyshared(run_dir: str) -> list[dict]:
    ckpt = load_checkpoint(run_dir)
    results = []
    for i in range(12):
        w = ckpt[f"vit.encoder.layer.{i}.attention.attention.query_key.weight"]
        w_heads = w.view(12, 64, 768)
        eranks = []
        for h in range(12):
            kernel = w_heads[h].T @ w_heads[h]
            eranks.append(effective_rank(kernel))
        results.append({
            "layer": i,
            "effective_rank_qk_mean": float(np.mean(eranks)),
            "head_query_diversity": head_diversity(w_heads),
        })
    return results


def analyze_lowrank(run_dir: str) -> list[dict]:
    ckpt = load_checkpoint(run_dir)
    results = []
    for i in range(12):
        q_a = ckpt[f"vit.encoder.layer.{i}.attention.attention.q_a"]
        q_b = ckpt[f"vit.encoder.layer.{i}.attention.attention.q_b"]
        k_a = ckpt[f"vit.encoder.layer.{i}.attention.attention.k_a"]
        k_b = ckpt[f"vit.encoder.layer.{i}.attention.attention.k_b"]
        eranks = []
        for h in range(12):
            w_q = torch.matmul(q_a[h], q_b[h])
            w_k = torch.matmul(k_a[h], k_b[h])
            kernel = w_q @ w_k.T
            eranks.append(effective_rank(kernel))
        results.append({
            "layer": i,
            "effective_rank_qk_mean": float(np.mean(eranks)),
            "head_q_diversity": head_diversity(torch.matmul(q_a, q_b)),
            "head_k_diversity": head_diversity(torch.matmul(k_a, k_b)),
        })
    return results


def analyze_bmbuv(run_dir: str) -> list[dict]:
    ckpt = load_checkpoint(run_dir)
    results = []
    for i in range(12):
        basis = ckpt[f"vit.encoder.layer.{i}.attention.attention.basis.weight"]
        u = ckpt[f"vit.encoder.layer.{i}.attention.attention.u_factor"]
        v = ckpt[f"vit.encoder.layer.{i}.attention.attention.v_factor"]
        erank_basis = effective_rank(basis)
        u_div = head_diversity(u)
        v_div = head_diversity(v)
        eranks = []
        for h in range(12):
            mid = torch.matmul(u[h], v[h].T)
            kernel = basis.T @ mid @ basis
            eranks.append(effective_rank(kernel))
        results.append({
            "layer": i,
            "effective_rank_basis": erank_basis,
            "effective_rank_qk_mean": float(np.mean(eranks)),
            "head_u_diversity": u_div,
            "head_v_diversity": v_div,
        })
    return results


def analyze_partialshared(run_dir: str) -> list[dict]:
    ckpt = load_checkpoint(run_dir)
    results = []
    for i in range(12):
        share = ckpt[f"vit.encoder.layer.{i}.attention.attention.share.weight"]
        q_priv = ckpt[f"vit.encoder.layer.{i}.attention.attention.query_priv.weight"]
        k_priv = ckpt[f"vit.encoder.layer.{i}.attention.attention.key_priv.weight"]
        H, d = 12, 768
        r_s = share.shape[0] // H
        r_p = q_priv.shape[0] // H
        share_h = share.view(H, r_s, d)
        q_priv_h = q_priv.view(H, r_p, d)
        k_priv_h = k_priv.view(H, r_p, d)
        q_full = torch.cat([share_h, q_priv_h], dim=1)
        k_full = torch.cat([share_h, k_priv_h], dim=1)
        eranks = []
        for h in range(H):
            kernel = q_full[h].T @ k_full[h]
            eranks.append(effective_rank(kernel))
        results.append({
            "layer": i,
            "effective_rank_qk_mean": float(np.mean(eranks)),
            "head_q_diversity": head_diversity(q_full),
            "head_k_diversity": head_diversity(k_full),
            "shared_portion": r_s / (r_s + r_p),
        })
    return results


RUN_ANALYZERS = {
    "Baseline": analyze_baseline,
    "FullyShared": analyze_fullyshared,
    "LowRank r32": analyze_lowrank,
    "BMB-UV r32s32": analyze_bmbuv,
    "BMB-UV r64s64": analyze_bmbuv,
    "PartialShared r48": analyze_partialshared,
}

RUN_DIRS = {
    "Baseline": "runs/imagenet1k_vit12_baseline_recipe_30ep_gpu5",
    "FullyShared": "runs/fs",
    "LowRank r32": "runs/lr32",
    "BMB-UV r32s32": "runs/bmbuv",
    "BMB-UV r64s64": "runs/imagenet1k_vit12_bmbuv_recipe_r64_s64_30ep_gpu1",
    "PartialShared r48": "runs/ps48",
}


def main() -> None:
    base = Path(os.path.expanduser("~/remote_lab"))
    out_dir = base / "reports/imagenet_total_comparison/rank_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, analyzer in RUN_ANALYZERS.items():
        run_dir = base / RUN_DIRS[name]
        model_file = run_dir / "model" / "model.safetensors"
        print(f"Checking {name}: {model_file} ... exists={model_file.exists()}")
        if not model_file.exists():
            print(f"  Skipping {name}: {model_file} not found")
            continue
        print(f"  Analyzing {name} ...")
        try:
            results = analyzer(str(run_dir))
            all_results[name] = results
            safe_name = name.replace(" ", "_").replace("$", "").replace("=", "_")
            with open(out_dir / f"{safe_name}.json", "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"  Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()
    with open(out_dir / "all_ranks.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone! Results saved to {out_dir}")


if __name__ == "__main__":
    main()
