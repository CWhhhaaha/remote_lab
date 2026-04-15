from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_path = (project_root / args.config).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("remote_lab run summary")
    print(f"project_root={project_root}")
    print(f"config={config_path}")
    print(f"output_dir={output_dir}")
    print(f"dry_run={args.dry_run}")
    print("config_contents=")
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
