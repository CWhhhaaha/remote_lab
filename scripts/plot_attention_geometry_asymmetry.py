#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot layerwise asymmetry ratio trajectories from an official "
            "attention-geometry custom-model pickle."
        )
    )
    parser.add_argument(
        "input_pickle",
        type=Path,
        help="Path to a result pickle such as bert-small-encoder-jigsaw.pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output image file, e.g. outputs/plots/foo.png.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=float,
        default=None,
        help=(
            "If set, convert the x-axis from update steps to epochs using "
            "epoch = step / steps_per_epoch."
        ),
    )
    return parser.parse_args()


def load_result(path: Path) -> dict:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This script needs optional dependencies to unpickle the official "
            "results. Install them first with:\n"
            "  python -m pip install torch transformers matplotlib"
        ) from exc

    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict at pickle root, got {type(obj)!r}")
    return obj


def checkpoint_sort_key(name: str) -> int:
    return int(name.split("-")[-1])


def main() -> None:
    args = parse_args()
    result = load_result(args.input_pickle)

    checkpoints = sorted(result.keys(), key=checkpoint_sort_key)
    x_steps = np.asarray([checkpoint_sort_key(k) for k in checkpoints], dtype=float)
    symmetry = np.stack(
        [np.asarray(result[k][1], dtype=float) for k in checkpoints],
        axis=0,
    )
    asymmetry = 1.0 - symmetry
    num_layers = asymmetry.shape[1]

    if args.steps_per_epoch is not None:
        x_values = x_steps / args.steps_per_epoch
        x_label = "Epoch"
    else:
        x_values = x_steps
        x_label = "Training step"

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

    colors = plt.cm.cividis(np.linspace(0.12, 0.92, num_layers))
    for layer_idx in range(num_layers):
        ax.plot(
            x_values,
            asymmetry[:, layer_idx],
            label=f"Layer {layer_idx + 1}",
            linewidth=2.2,
            color=colors[layer_idx],
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Asymmetry ratio")
    ax.set_ylim(0.0, max(0.55, float(asymmetry.max()) + 0.02))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=True, ncol=2)
    ax.set_title(
        args.title or f"Layerwise asymmetry ratio: {args.input_pickle.stem}",
        pad=12,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"saved_plot={args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
