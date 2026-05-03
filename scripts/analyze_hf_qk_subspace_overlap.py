#!/usr/bin/env python3
"""Analyze pairwise Q/K subspace overlap in a HF safetensors checkpoint.

This script directly tests whether per-head query subspaces or per-group key
subspaces exhibit strong overlap. For each requested layer, it:

1. extracts per-head Q blocks and per-group K blocks
2. computes an orthonormal basis for each block
3. measures pairwise subspace overlap using principal-angle statistics
4. optionally saves heatmaps and JSON summaries

The overlap score between two subspaces with orthonormal bases A and B is

    overlap(A, B) = ||A^T B||_F^2 / min(rank(A), rank(B)),

which lies in [0, 1] and equals the average squared cosine of the principal
angles between the two subspaces.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing config.json and model.safetensors.index.json",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0],
        help="Zero-based layer indices to analyze (default: 0)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for linear algebra (default: auto)",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.95,
        help=(
            "Per-block energy threshold for choosing the retained basis rank when "
            "--per-block-rank is not set (default: 0.95)."
        ),
    )
    parser.add_argument(
        "--per-block-rank",
        type=int,
        default=None,
        help="Optional fixed retained rank for each per-head/per-group subspace.",
    )
    parser.add_argument(
        "--q-key-template",
        default="model.layers.{layer}.self_attn.q_proj.weight",
        help="Template for q projection keys",
    )
    parser.add_argument(
        "--k-key-template",
        default="model.layers.{layer}.self_attn.k_proj.weight",
        help="Template for k projection keys",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory for saving overlap heatmaps (PNG) per layer.",
    )
    return parser.parse_args()


def choose_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    return name


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json", "r") as f:
        return json.load(f)


def load_weight_index(model_dir: Path) -> dict[str, str]:
    with open(model_dir / "model.safetensors.index.json", "r") as f:
        data = json.load(f)
    return data["weight_map"]


def load_named_tensors(
    model_dir: Path,
    weight_map: dict[str, str],
    tensor_names: Iterable[str],
) -> dict[str, object]:
    from safetensors import safe_open

    shards_to_keys: dict[str, list[str]] = defaultdict(list)
    for name in tensor_names:
        shard = weight_map.get(name)
        if shard is None:
            raise KeyError(f"Tensor not found in index: {name}")
        shards_to_keys[shard].append(name)

    loaded = {}
    for shard, keys in shards_to_keys.items():
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in keys:
                loaded[key] = f.get_tensor(key)
    return loaded


def split_projection_blocks(weight, num_blocks: int, head_dim: int, hidden_size: int):
    if tuple(weight.shape) != (num_blocks * head_dim, hidden_size):
        raise ValueError(
            f"Unexpected projection shape {tuple(weight.shape)}; "
            f"expected {(num_blocks * head_dim, hidden_size)}"
        )
    return [
        weight[idx * head_dim : (idx + 1) * head_dim, :].t().contiguous()
        for idx in range(num_blocks)
    ]


def basis_from_block(block, energy_threshold: float, per_block_rank: int | None):
    import torch

    block = block.float()
    U, s, _ = torch.linalg.svd(block, full_matrices=False)
    if per_block_rank is not None:
        rank = min(per_block_rank, U.shape[1])
    else:
        energy = torch.cumsum(s**2, dim=0) / (s**2).sum().clamp_min(1e-12)
        rank = int(torch.searchsorted(energy, torch.tensor(energy_threshold, device=energy.device)).item()) + 1
        rank = min(rank, U.shape[1])
    return U[:, :rank], s, rank


def pairwise_overlap(bases: list) -> tuple[list[list[float]], dict]:
    import torch

    n = len(bases)
    mat = torch.zeros((n, n), dtype=torch.float32, device=bases[0].device)
    angles_mean = torch.zeros((n, n), dtype=torch.float32, device=bases[0].device)
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            cosines = torch.linalg.svdvals(bases[i].transpose(0, 1) @ bases[j]).clamp(0.0, 1.0)
            denom = float(min(bases[i].shape[1], bases[j].shape[1]))
            score = float((cosines**2).sum().item() / max(denom, 1.0))
            mean_angle = float(torch.rad2deg(torch.arccos(cosines.clamp(-1.0, 1.0))).mean().item())
            mat[i, j] = mat[j, i] = score
            angles_mean[i, j] = angles_mean[j, i] = mean_angle

    off_diag = mat[~torch.eye(n, dtype=torch.bool, device=mat.device)]
    angle_off_diag = angles_mean[~torch.eye(n, dtype=torch.bool, device=angles_mean.device)]
    summary = {
        "mean_off_diagonal_overlap": float(off_diag.mean().item()) if off_diag.numel() else 1.0,
        "median_off_diagonal_overlap": float(off_diag.median().item()) if off_diag.numel() else 1.0,
        "max_off_diagonal_overlap": float(off_diag.max().item()) if off_diag.numel() else 1.0,
        "min_off_diagonal_overlap": float(off_diag.min().item()) if off_diag.numel() else 1.0,
        "mean_off_diagonal_principal_angle_deg": float(angle_off_diag.mean().item()) if angle_off_diag.numel() else 0.0,
    }
    return mat.cpu().tolist(), summary


def save_heatmap(matrix: list[list[float]], title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(matrix, dtype=float)
    plt.figure(figsize=(6.0, 5.0))
    im = plt.imshow(arr, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def analyze_layer(
    q_weight,
    k_weight,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    device: str,
    energy_threshold: float,
    per_block_rank: int | None,
) -> dict:
    q_blocks = split_projection_blocks(q_weight, num_attention_heads, head_dim, hidden_size)
    k_blocks = split_projection_blocks(k_weight, num_key_value_heads, head_dim, hidden_size)

    q_bases = []
    q_ranks = []
    for block in q_blocks:
        basis, _, rank = basis_from_block(block.to(device), energy_threshold, per_block_rank)
        q_bases.append(basis)
        q_ranks.append(rank)

    k_bases = []
    k_ranks = []
    for block in k_blocks:
        basis, _, rank = basis_from_block(block.to(device), energy_threshold, per_block_rank)
        k_bases.append(basis)
        k_ranks.append(rank)

    q_overlap, q_summary = pairwise_overlap(q_bases)
    k_overlap, k_summary = pairwise_overlap(k_bases)

    return {
        "q_retained_ranks": q_ranks,
        "k_retained_ranks": k_ranks,
        "q_overlap_matrix": q_overlap,
        "k_overlap_matrix": k_overlap,
        "q_summary": q_summary,
        "k_summary": k_summary,
    }


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    device = choose_device(args.device)

    config = load_config(model_dir)
    weight_map = load_weight_index(model_dir)

    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config.get("num_key_value_heads", num_attention_heads))
    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    head_dim = int(head_dim)

    layers = sorted(set(args.layers))
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer index {layer} out of range for {num_layers} layers.")

    tensor_names = []
    q_keys = {}
    k_keys = {}
    for layer in layers:
        q_key = args.q_key_template.format(layer=layer)
        k_key = args.k_key_template.format(layer=layer)
        q_keys[layer] = q_key
        k_keys[layer] = k_key
        tensor_names.extend([q_key, k_key])

    tensors = load_named_tensors(model_dir, weight_map, tensor_names)

    result = {
        "model_dir": str(model_dir),
        "model_config": {
            "architectures": config.get("architectures"),
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
        },
        "device": device,
        "energy_threshold": args.energy_threshold,
        "per_block_rank": args.per_block_rank,
        "layers": {},
    }

    print(f"Analyzing overlap in {model_dir}")
    print(
        f"hidden_size={hidden_size}, layers={num_layers}, "
        f"q_heads={num_attention_heads}, kv_heads={num_key_value_heads}, head_dim={head_dim}"
    )
    print(
        f"device={device}, energy_threshold={args.energy_threshold}, "
        f"per_block_rank={args.per_block_rank}, layers={layers}"
    )

    for layer in layers:
        q_weight = tensors[q_keys[layer]]
        k_weight = tensors[k_keys[layer]]
        print(f"\n[layer {layer}] q_shape={tuple(q_weight.shape)} k_shape={tuple(k_weight.shape)}")
        layer_result = analyze_layer(
            q_weight=q_weight,
            k_weight=k_weight,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            device=device,
            energy_threshold=args.energy_threshold,
            per_block_rank=args.per_block_rank,
        )
        result["layers"][str(layer)] = layer_result

        print(
            f"  Q overlap: mean={layer_result['q_summary']['mean_off_diagonal_overlap']:.4f}, "
            f"median={layer_result['q_summary']['median_off_diagonal_overlap']:.4f}, "
            f"max={layer_result['q_summary']['max_off_diagonal_overlap']:.4f}, "
            f"mean_angle={layer_result['q_summary']['mean_off_diagonal_principal_angle_deg']:.2f} deg"
        )
        print(
            f"  K overlap: mean={layer_result['k_summary']['mean_off_diagonal_overlap']:.4f}, "
            f"median={layer_result['k_summary']['median_off_diagonal_overlap']:.4f}, "
            f"max={layer_result['k_summary']['max_off_diagonal_overlap']:.4f}, "
            f"mean_angle={layer_result['k_summary']['mean_off_diagonal_principal_angle_deg']:.2f} deg"
        )

        if args.plot_dir is not None:
            plot_dir = args.plot_dir
            save_heatmap(
                layer_result["q_overlap_matrix"],
                f"Layer {layer} Q-head Subspace Overlap",
                plot_dir / f"layer{layer:02d}_q_overlap.png",
            )
            save_heatmap(
                layer_result["k_overlap_matrix"],
                f"Layer {layer} K-group Subspace Overlap",
                plot_dir / f"layer{layer:02d}_k_overlap.png",
            )
            print(f"  heatmaps saved under {plot_dir}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
