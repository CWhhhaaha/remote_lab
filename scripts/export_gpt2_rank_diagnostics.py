#!/usr/bin/env python3
"""Export GPT-2 layerwise rank diagnostics from latest checkpoints.

This script mirrors the ImageNet-side diagnostic definition by using
entropy effective rank for both the shared basis B and head-specific
interaction operators M.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


RUNS = {
    "BBT $r$=64": Path("/data/chenwei/remote_lab/runs/gpt2_c4_bbt_r64_50k"),
    "BMB $r$=64": Path("/data/chenwei/remote_lab/runs/gpt2_c4_bmb_r64_50k"),
    "BMB-UV $r$=32,$s$=32": Path("/data/chenwei/remote_lab/runs/gpt2_c4_bmbuv_r32s32_50k"),
    "BMB-UV $r$=64,$s$=64": Path("/data/chenwei/remote_lab/runs/gpt2_c4_bmbuv_r64s64_50k"),
}


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return ckpts[-1]


def entropy_effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> float:
    singular_values = torch.linalg.svdvals(matrix.float())
    total = singular_values.sum().clamp_min(eps)
    probs = singular_values / total
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return float(torch.exp(entropy).item())


def analyze_bbt(sd: dict) -> list[dict]:
    rows = []
    i = 0
    while f"transformer.h.{i}.attn.basis.weight" in sd:
        basis = sd[f"transformer.h.{i}.attn.basis.weight"]
        r = float(basis.shape[0])
        rows.append(
            {
                "layer": i + 1,
                "effective_rank_basis": entropy_effective_rank(basis),
                "effective_rank_qk_mean": r,
                "effective_rank_qk_min": r,
                "effective_rank_qk_max": r,
            }
        )
        i += 1
    return rows


def analyze_bmbuv(sd: dict) -> list[dict]:
    rows = []
    i = 0
    while f"transformer.h.{i}.attn.basis.weight" in sd:
        basis = sd[f"transformer.h.{i}.attn.basis.weight"]
        u = sd[f"transformer.h.{i}.attn.u_factor"]
        v = sd[f"transformer.h.{i}.attn.v_factor"]
        eranks = []
        for h in range(u.shape[0]):
            m = u[h] @ v[h].transpose(0, 1)
            eranks.append(entropy_effective_rank(m))
        rows.append(
            {
                "layer": i + 1,
                "effective_rank_basis": entropy_effective_rank(basis),
                "effective_rank_qk_mean": float(sum(eranks) / len(eranks)),
                "effective_rank_qk_min": float(min(eranks)),
                "effective_rank_qk_max": float(max(eranks)),
            }
        )
        i += 1
    return rows


def analyze_bmb(sd: dict) -> list[dict]:
    rows = []
    i = 0
    while f"transformer.h.{i}.attn.basis.weight" in sd:
        basis = sd[f"transformer.h.{i}.attn.basis.weight"]
        core = sd[f"transformer.h.{i}.attn.core"]
        head_residual = sd[f"transformer.h.{i}.attn.head_residual"]
        n_head = head_residual.shape[0]
        centered = head_residual - head_residual.mean(dim=0, keepdim=True)
        head_mats = core.unsqueeze(0) / float(n_head) + centered
        eranks = [entropy_effective_rank(head_mats[h]) for h in range(n_head)]
        rows.append(
            {
                "layer": i + 1,
                "effective_rank_basis": entropy_effective_rank(basis),
                "effective_rank_qk_mean": float(sum(eranks) / len(eranks)),
                "effective_rank_qk_min": float(min(eranks)),
                "effective_rank_qk_max": float(max(eranks)),
            }
        )
        i += 1
    return rows


ANALYZERS = {
    "BBT $r$=64": analyze_bbt,
    "BMB $r$=64": analyze_bmb,
    "BMB-UV $r$=32,$s$=32": analyze_bmbuv,
    "BMB-UV $r$=64,$s$=64": analyze_bmbuv,
}


def main() -> None:
    all_results = {}
    meta = {}

    for label, run_dir in RUNS.items():
        ckpt = latest_checkpoint(run_dir)
        sd = load_file(str(ckpt / "model.safetensors"))
        rows = ANALYZERS[label](sd)
        all_results[label] = rows
        meta[label] = {"checkpoint": str(ckpt), "num_layers": len(rows)}
        print(label, ckpt.name, f"{len(rows)} layers")

    out = {"meta": meta, "all_ranks": all_results}
    out_path = Path("/tmp/gpt2_rank_diagnostics_entropy_latest.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
