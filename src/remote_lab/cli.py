from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal remote_lab entrypoint for experiment runs."
    )
    parser.add_argument(
        "--config",
        default="configs/base.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/dev",
        help="Directory for this run's outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths without creating outputs.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_interval_config(
    project_root: Path,
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any] | None:
    regularization = config.get("regularization")
    if not isinstance(regularization, dict):
        return None

    interval_config = regularization.get("interval_config")
    if not interval_config:
        return None

    interval_path = Path(interval_config)
    if not interval_path.is_absolute():
        interval_path = (project_root / interval_path).resolve()

    if not interval_path.exists():
        raise FileNotFoundError(f"Interval config not found: {interval_path}")

    interval_data = load_json(interval_path)
    config["regularization"]["resolved_interval_config"] = str(interval_path)
    config["regularization"]["intervals"] = interval_data
    return interval_data


def summarize_interval_config(interval_data: dict[str, Any]) -> dict[str, Any]:
    layers = interval_data.get("layers", [])
    widths = [layer["rho_max"] - layer["rho_min"] for layer in layers]
    centers = [(layer["rho_max"] + layer["rho_min"]) / 2.0 for layer in layers]
    return {
        "name": interval_data.get("name"),
        "metric": interval_data.get("metric"),
        "num_layers": len(layers),
        "centers": [round(value, 6) for value in centers],
        "widths": [round(value, 6) for value in widths],
        "layers": layers,
    }


def resolve_dataset_paths(project_root: Path, config: dict[str, Any]) -> dict[str, str] | None:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return None

    resolved: dict[str, str] = {}
    for key, value in dataset.items():
        if isinstance(value, str) and (
            value.startswith("data/")
            or value.startswith("./data/")
            or value == "data"
        ):
            resolved[key] = str((project_root / value).resolve())

    if resolved:
        config["dataset"]["resolved_paths"] = resolved
    return resolved or None


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_path = (project_root / args.config).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_json(config_path)
    interval_data = resolve_interval_config(project_root, config, config_path)
    dataset_paths = resolve_dataset_paths(project_root, config)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("remote_lab run summary")
    print(f"project_root={project_root}")
    print(f"config={config_path}")
    print(f"output_dir={output_dir}")
    print(f"dry_run={args.dry_run}")
    if dataset_paths is not None:
        print("dataset_paths=")
        print(json.dumps(dataset_paths, indent=2, sort_keys=True))
    if interval_data is not None:
        print("interval_summary=")
        print(json.dumps(summarize_interval_config(interval_data), indent=2, sort_keys=True))
    print("config_contents=")
    print(json.dumps(config, indent=2, sort_keys=True))

    if args.dry_run:
        return

    task_type = str(config.get("task_type", "text_mlm"))
    if task_type == "vision_classification":
        from remote_lab.vision_training import train_vision_experiment

        train_fn = train_vision_experiment
    elif task_type == "vision_classification_recipe":
        from remote_lab.vision_training_recipe import train_vision_recipe_experiment

        train_fn = train_vision_recipe_experiment
    else:
        from remote_lab.training import train_experiment

        train_fn = train_experiment

    print("training_status=starting")
    summary = train_fn(config=config, output_dir=output_dir, project_root=project_root)
    print("training_status=completed")
    print("run_summary=")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
