#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract layerwise asymmetry-ratio interval candidates from the "
            "official attention-geometry result pickle."
        )
    )
    parser.add_argument(
        "input_pickle",
        type=Path,
        help="Path to a custom-model result pickle such as bert-small-encoder-jigsaw.pkl.",
    )
    parser.add_argument(
        "--tail-windows",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="Checkpoint windows from the end of training used for interval summaries.",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="Optional absolute padding applied symmetrically to each interval boundary.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for saving the extracted summary.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict[str, list[Any]]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This script needs optional dependencies to unpickle the official "
            "results. Install them in your active environment first:\n"
            "  python -m pip install torch transformers"
        ) from exc

    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected pickle root to be dict, got {type(obj)!r}")
    return obj


def checkpoint_sort_key(name: str) -> int:
    try:
        return int(name.split("-")[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected checkpoint name: {name}") from exc


def to_serializable_number(value: float) -> float:
    return round(float(value), 6)


def compute_summary(asymmetry: np.ndarray, checkpoints: list[str], tail: int, pad: float) -> dict[str, Any]:
    tail = min(tail, asymmetry.shape[0])
    tail_values = asymmetry[-tail:]
    per_layer = []

    for layer_idx in range(tail_values.shape[1]):
        values = tail_values[:, layer_idx]
        q10 = np.quantile(values, 0.10)
        q25 = np.quantile(values, 0.25)
        q50 = np.quantile(values, 0.50)
        q75 = np.quantile(values, 0.75)
        q90 = np.quantile(values, 0.90)
        layer_summary = {
            "layer": layer_idx + 1,
            "mean": to_serializable_number(values.mean()),
            "std": to_serializable_number(values.std()),
            "min": to_serializable_number(values.min()),
            "max": to_serializable_number(values.max()),
            "q10": to_serializable_number(q10),
            "q25": to_serializable_number(q25),
            "q50": to_serializable_number(q50),
            "q75": to_serializable_number(q75),
            "q90": to_serializable_number(q90),
            "interval_q10_q90": [
                to_serializable_number(max(0.0, q10 - pad)),
                to_serializable_number(min(1.0, q90 + pad)),
            ],
            "interval_q25_q75": [
                to_serializable_number(max(0.0, q25 - pad)),
                to_serializable_number(min(1.0, q75 + pad)),
            ],
        }
        per_layer.append(layer_summary)

    return {
        "tail_window": tail,
        "start_checkpoint": checkpoints[-tail],
        "end_checkpoint": checkpoints[-1],
        "layers": per_layer,
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"source={report['source_pickle']}")
    print(f"checkpoints={report['checkpoint_count']}")
    print(f"layers={report['num_layers']}")
    print(f"final_checkpoint={report['final_checkpoint']}")
    print("final_asymmetry_ratio=" + json.dumps(report["final_asymmetry_ratio"]))
    print()

    for tail_report in report["tail_reports"]:
        print(
            f"[tail={tail_report['tail_window']}] "
            f"{tail_report['start_checkpoint']} -> {tail_report['end_checkpoint']}"
        )
        for layer in tail_report["layers"]:
            print(
                "  "
                f"L{layer['layer']}: "
                f"mean={layer['mean']:.6f} "
                f"q10_q90={layer['interval_q10_q90']} "
                f"q25_q75={layer['interval_q25_q75']}"
            )
        print()


def main() -> None:
    args = parse_args()
    result_dict = load_pickle(args.input_pickle)

    checkpoints = sorted(result_dict.keys(), key=checkpoint_sort_key)
    symmetry = np.stack(
        [np.asarray(result_dict[checkpoint][1], dtype=float) for checkpoint in checkpoints],
        axis=0,
    )
    asymmetry = 1.0 - symmetry
    config = result_dict[checkpoints[0]][0]

    report = {
        "source_pickle": str(args.input_pickle.resolve()),
        "checkpoint_count": len(checkpoints),
        "num_layers": int(config.num_hidden_layers),
        "parameter_count": int(result_dict[checkpoints[0]][3][0]),
        "first_checkpoint": checkpoints[0],
        "final_checkpoint": checkpoints[-1],
        "first_asymmetry_ratio": [to_serializable_number(v) for v in asymmetry[0]],
        "final_asymmetry_ratio": [to_serializable_number(v) for v in asymmetry[-1]],
        "tail_reports": [
            compute_summary(asymmetry, checkpoints, tail, args.pad)
            for tail in args.tail_windows
        ],
    }

    print_summary(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
