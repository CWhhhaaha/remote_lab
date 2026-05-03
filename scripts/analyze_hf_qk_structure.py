#!/usr/bin/env python3
"""Analyze Q/K structural sharing in a Hugging Face safetensors checkpoint.

This script is intended as a lightweight structural analysis tool for large
open-weight LLM checkpoints. Given one or more layers, it extracts the query
and key projection weights, constructs a joint Q/K matrix for each layer, and
reports:

1. top-r shared-subspace energy
2. entropy effective rank of the joint Q/K matrix
3. shared-basis reconstruction error for Q and K
4. shared-basis kernel reconstruction error under the model's GQA/MHA layout

The default key templates follow the common Hugging Face attention naming used
by LLaMA/Qwen-style architectures:

    model.layers.{layer}.self_attn.q_proj.weight
    model.layers.{layer}.self_attn.k_proj.weight

Typical usage:

    python scripts/analyze_hf_qk_structure.py \
      --model-dir /nfsdata2/openai/gpt-oss-20b \
      --layers 0 \
      --ranks 64 128 256 \
      --device cuda \
      --output /tmp/gpt_oss_20b_qk_layer0.json
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
        "--ranks",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Shared-basis ranks to evaluate (default: 64 128 256)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for linear algebra (default: auto)",
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
        "--basis-source",
        default="joint_svd",
        choices=["joint_svd", "kernel_aligned"],
        help=(
            "How to construct the shared basis B. "
            "'joint_svd' uses the top left singular vectors of [Q|K], "
            "while 'kernel_aligned' uses the top eigenvectors of an "
            "aggregate kernel covariance built from Q_h K_g(h)^T."
        ),
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
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with open(config_path, "r") as f:
        return json.load(f)


def load_weight_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors.index.json in {model_dir}")
    with open(index_path, "r") as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Index file does not contain a valid weight_map.")
    return weight_map


def load_named_tensors(
    model_dir: Path,
    weight_map: dict[str, str],
    tensor_names: Iterable[str],
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    shards_to_keys: dict[str, list[str]] = defaultdict(list)
    for name in tensor_names:
        shard = weight_map.get(name)
        if shard is None:
            raise KeyError(f"Tensor not found in index: {name}")
        shards_to_keys[shard].append(name)

    loaded: dict[str, torch.Tensor] = {}
    for shard, keys in shards_to_keys.items():
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                loaded[key] = f.get_tensor(key)
    return loaded


def entropy_effective_rank(singular_values: torch.Tensor, eps: float = 1e-12) -> float:
    import torch

    total = singular_values.sum().clamp_min(eps)
    probs = singular_values / total
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return float(torch.exp(entropy).item())


def relative_fro_error(reference: torch.Tensor, approx: torch.Tensor, eps: float = 1e-12) -> float:
    import torch

    denom = torch.norm(reference).clamp_min(eps)
    return float((torch.norm(reference - approx) / denom).item())


def split_projection_blocks(
    weight: torch.Tensor,
    num_blocks: int,
    head_dim: int,
    hidden_size: int,
) -> list[torch.Tensor]:
    if tuple(weight.shape) != (num_blocks * head_dim, hidden_size):
        raise ValueError(
            f"Unexpected projection shape {tuple(weight.shape)}; "
            f"expected {(num_blocks * head_dim, hidden_size)}"
        )
    blocks = []
    for idx in range(num_blocks):
        block = weight[idx * head_dim : (idx + 1) * head_dim, :].t().contiguous()
        blocks.append(block)
    return blocks


def analyze_layer(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    ranks: list[int],
    device: str,
    basis_source: str,
) -> dict:
    import torch

    q_blocks = split_projection_blocks(q_weight, num_attention_heads, head_dim, hidden_size)
    k_blocks = split_projection_blocks(k_weight, num_key_value_heads, head_dim, hidden_size)

    joint = torch.cat(q_blocks + k_blocks, dim=1).to(device=device, dtype=torch.float32)
    U, singular_values, _ = torch.linalg.svd(joint, full_matrices=False)

    total_energy = (singular_values**2).sum().clamp_min(1e-12)
    top_energy = {}
    for rank in ranks:
        capped = min(rank, singular_values.numel())
        ratio = float(((singular_values[:capped] ** 2).sum() / total_energy).item())
        top_energy[str(rank)] = ratio

    eff_rank = entropy_effective_rank(singular_values)

    q_blocks_dev = [block.to(device=device, dtype=torch.float32) for block in q_blocks]
    k_blocks_dev = [block.to(device=device, dtype=torch.float32) for block in k_blocks]
    query_per_kv_group = num_attention_heads // num_key_value_heads

    if basis_source == "joint_svd":
        basis_bank = U
        kernel_basis_meta = None
    elif basis_source == "kernel_aligned":
        aggregate = torch.zeros(
            (hidden_size, hidden_size),
            device=device,
            dtype=torch.float32,
        )
        for q_idx, q_block in enumerate(q_blocks_dev):
            kv_idx = q_idx // query_per_kv_group
            kernel_ref = q_block @ k_blocks_dev[kv_idx].transpose(0, 1)
            aggregate = aggregate + kernel_ref @ kernel_ref.transpose(0, 1)
            aggregate = aggregate + kernel_ref.transpose(0, 1) @ kernel_ref
        evals, evecs = torch.linalg.eigh(aggregate)
        order = torch.argsort(evals, descending=True)
        basis_bank = evecs[:, order]
        top_vals = evals[order].clamp_min(0)
        denom = top_vals.sum().clamp_min(1e-12)
        kernel_basis_meta = {
            "kernel_aligned_top_eigen_mass": {
                str(rank): float(top_vals[: min(rank, top_vals.numel())].sum().item() / denom.item())
                for rank in ranks
            }
        }
    else:
        raise ValueError(f"Unsupported basis_source: {basis_source}")

    reconstruction = {}
    for rank in ranks:
        capped = min(rank, basis_bank.shape[1])
        basis = basis_bank[:, :capped]

        q_errs = []
        q_hat = []
        for block in q_blocks_dev:
            approx = basis @ (basis.transpose(0, 1) @ block)
            q_hat.append(approx)
            q_errs.append(relative_fro_error(block, approx))

        k_errs = []
        k_hat = []
        for block in k_blocks_dev:
            approx = basis @ (basis.transpose(0, 1) @ block)
            k_hat.append(approx)
            k_errs.append(relative_fro_error(block, approx))

        kernel_errs = []
        for q_idx, q_block in enumerate(q_blocks_dev):
            kv_idx = q_idx // query_per_kv_group
            kernel_ref = q_block @ k_blocks_dev[kv_idx].transpose(0, 1)
            kernel_hat = q_hat[q_idx] @ k_hat[kv_idx].transpose(0, 1)
            kernel_errs.append(relative_fro_error(kernel_ref, kernel_hat))

        reconstruction[str(rank)] = {
            "mean_q_reconstruction_error": float(sum(q_errs) / len(q_errs)),
            "mean_k_reconstruction_error": float(sum(k_errs) / len(k_errs)),
            "mean_kernel_error": float(sum(kernel_errs) / len(kernel_errs)),
        }

    return {
        "joint_shape": list(joint.shape),
        "basis_source": basis_source,
        "top_r_shared_subspace_energy": top_energy,
        "effective_rank": eff_rank,
        "reconstruction": reconstruction,
        "kernel_aligned_basis_stats": kernel_basis_meta,
    }


def main() -> None:
    args = parse_args()
    import torch

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

    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_attention_heads={num_attention_heads} is not divisible by "
            f"num_key_value_heads={num_key_value_heads}"
        )

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
        "layers": {},
    }

    print(f"Analyzing {model_dir}")
    print(
        f"hidden_size={hidden_size}, layers={num_layers}, "
        f"q_heads={num_attention_heads}, kv_heads={num_key_value_heads}, head_dim={head_dim}"
    )
    print(f"device={device}, ranks={args.ranks}, layers={layers}")

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
            ranks=args.ranks,
            device=device,
            basis_source=args.basis_source,
        )
        result["layers"][str(layer)] = layer_result

        print("  joint_shape =", tuple(layer_result["joint_shape"]))
        for rank, energy in layer_result["top_r_shared_subspace_energy"].items():
            print(f"  top-{rank} energy = {energy:.6f}")
        print(f"  effective_rank = {layer_result['effective_rank']:.3f}")
        if layer_result["kernel_aligned_basis_stats"] is not None:
            for rank, mass in layer_result["kernel_aligned_basis_stats"]["kernel_aligned_top_eigen_mass"].items():
                print(f"  kernel-basis top-{rank} mass = {mass:.6f}")
        for rank, values in layer_result["reconstruction"].items():
            print(
                f"  r={rank}: "
                f"q_err={values['mean_q_reconstruction_error']:.6f}, "
                f"k_err={values['mean_k_reconstruction_error']:.6f}, "
                f"kernel_err={values['mean_kernel_error']:.6f}"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
