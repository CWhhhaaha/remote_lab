#!/usr/bin/env python3
"""Analyze effective rank and head diversity from trained model checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def effective_rank(matrix: torch.Tensor, threshold: float = 0.01) -> int:
    """Number of singular values above threshold * max_sv."""
    try:
        _, s, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    except Exception:
        # Fallback for older PyTorch
        s = torch.linalg.svdvals(matrix.float())
    return int((s > threshold * s[0]).sum().item())


def head_diversity(mats: torch.Tensor) -> float:
    """Mean pairwise cosine similarity of flattened matrices. [H, ...]"""
    H = mats.shape[0]
    flat = mats.view(H, -1)
    norms = flat.norm(dim=1, keepdim=True)
    normed = flat / (norms + 1e-8)
    sim = normed @ normed.T
    mask = ~torch.eye(H, dtype=torch.bool)
    return float(sim[mask].mean().item())


def analyze_baseline(run_dir: str) -> list[dict]:
    ckpt = torch.load(Path(run_dir) / "model" / "pytorch_model.bin", map_location="cpu")
    results = []
    for i in range(12):
        q = ckpt[f"vit.encoder.layer.{i}.attention.attention.query.weight"]  # [768, 768]
        k = ckpt[f"vit.encoder.layer.{i}.attention.attention.key.weight"]    # [768, 768]
        q_heads = q.view(12, 64, 768)
        k_heads = k.view(12, 64, 768)
        eranks = []
        for h in range(12):
            kernel = q_heads[h].T @ k_heads[h]  # [768, 768]
            eranks.append(effective_rank(kernel))
        results.append({
            "layer": i,
            "effective_rank_qk_mean": float(np.mean(eranks)),
            "head_query_diversity": head_diversity(q_heads),
            "head_key_diversity": head_diversity(k_heads),
        })
    return results


def analyze_fullyshared(run_dir: str) -> list[dict]:
    ckpt = torch.load(Path(run_dir) / "model" / "pytorch_model.bin", map_location="cpu")
    results = []
    for i in range(12):
        w = ckpt[f"vit.encoder.layer.{i}.attention.attention.query_key.weight"]  # [768, 768]
        w_heads = w.view(12, 64, 768)
        eranks = []
        for h in range(12):
            kernel = w_heads[h].T @ w_heads[h]  # symmetric
            eranks.append(effective_rank(kernel))
        results.append({
            "layer": i,
            "effective_rank_qk_mean": float(np.mean(eranks)),
            "head_query_diversity": head_diversity(w_heads),
        })
    return results


def analyze_lowrank(run_dir: str) -> list[dict]:
    ckpt = torch.load(Path(run_dir) / "model" / "pytorch_model.bin", map_location="cpu")
    results = []
    for i in range(12):
        q_a = ckpt[f"vit.encoder.layer.{i}.attention.attention.q_a"]  # [12, 768, 32]
        q_b = ckpt[f"vit.encoder.layer.{i}.attention.attention.q_b"]  # [12, 32, 64]
        k_a = ckpt[f"vit.encoder.layer.{i}.attention.attention.k_a"]  # [12, 768, 32]
        k_b = ckpt[f"vit.encoder.layer.{i}.attention.attention.k_b"]  # [12, 32, 64]
        eranks = []
        for h in range(12):
            w_q = torch.matmul(q_a[h], q_b[h])  # [768, 64]
            w_k = torch.matmul(k_a[h], k_b[h])  # [768, 64]
            kernel = w_q @ w_k.T  # [768, 768]
            eranks.append(effective_rank(kernel))
        results.append({
            "layer": i,
            "effective_rank_qk_mean": float(np.mean(eranks)),
            "head_q_diversity": head_diversity(torch.matmul(q_a, q_b)),
            "head_k_diversity": head_diversity(torch.matmul(k_a, k_b)),
        })
    return results


def analyze_bmbuv(run_dir: str) -> list[dict]:
    ckpt = torch.load(Path(run_dir) / "model" / "pytorch_model.bin", map_location="cpu")
    results = []
    for i in range(12):
        basis = ckpt[f"vit.encoder.layer.{i}.attention.attention.basis.weight"]  # [32, 768] or [64, 768]
        u = ckpt[f"vit.encoder.layer.{i}.attention.attention.u_factor"]  # [12, r, s]
        v = ckpt[f"vit.encoder.layer.{i}.attention.attention.v_factor"]  # [12, r, s]
        r = basis.shape[0]
        # Effective rank of basis
        erank_basis = effective_rank(basis)
        # Head diversity of U and V
        u_diversity = head_diversity(u)
        v_diversity = head_diversity(v)
        # Per-head QK kernel effective rank
        eranks = []
        for h in range(12):
            # QK kernel ≈ basis.T @ u[h] @ v[h].T @ basis
            mid = torch.matmul(u[h], v[h].T)  # [r, r]
            kernel = basis.T @ mid @ basis  # [768, 768]
            eranks.append(effective_rank(kernel))
        results.append({
            "layer": i,
            "effective_rank_basis": erank_basis,
            "effective_rank_qk_mean": float(np.mean(eranks)),
            "head_u_diversity": u_diversity,
            "head_v_diversity": v_diversity,
        })
    return results


def analyze_partialshared(run_dir: str) -> list[dict]:
    ckpt = torch.load(Path(run_dir) / "model" / "pytorch_model.bin", map_location="cpu")
    results = []
    for i in range(12):
        share = ckpt[f"vit.encoder.layer.{i}.attention.attention.share.weight"]         # [H*r_s, 768]
        q_priv = ckpt[f"vit.encoder.layer.{i}.attention.attention.query_priv.weight"]   # [H*r_p, 768]
        k_priv = ckpt[f"vit.encoder.layer.{i}.attention.attention.key_priv.weight"]     # [H*r_p, 768]
        # Reshape to per-head
        H, d = 12, 768
        r_s = share.shape[0] // H
        r_p = q_priv.shape[0] // H
        share_h = share.view(H, r_s, d)
        q_priv_h = q_priv.view(H, r_p, d)
        k_priv_h = k_priv.view(H, r_p, d)
        # Concatenate for full Q/K projection per head
        q_full = torch.cat([share_h, q_priv_h], dim=1)  # [H, d_k, d]
        k_full = torch.cat([share_h, k_priv_h], dim=1)  # [H, d_k, d]
        eranks = []
        for h in range(H):
            kernel = q_full[h].T @ k_full[h]  # [d, d]
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
    out_dir = Path("reports/imagenet_total_comparison/rank_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, analyzer in RUN_ANALYZERS.items():
        run_dir = RUN_DIRS[name]
        if not (Path(run_dir) / "model" / "pytorch_model.bin").exists():
            print(f"Skipping {name}: checkpoint not found")
            continue
        print(f"Analyzing {name} ...")
        try:
            results = analyzer(run_dir)
            all_results[name] = results
            with open(out_dir / f"{name.replace(' ', '_').replace('$', '').replace('=', '_')}.json", "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
    
    with open(out_dir / "all_ranks.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDone! Results saved to {out_dir}")


if __name__ == "__main__":
    main()
