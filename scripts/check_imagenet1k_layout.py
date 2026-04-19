from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an ImageNet-1k ImageFolder layout.")
    parser.add_argument(
        "--data-root",
        default="data/raw/imagenet1k",
        help="Root directory containing train/ and val/ subdirectories.",
    )
    return parser


def count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = (project_root / args.data_root).resolve()
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing val directory: {val_dir}")

    train_dataset = datasets.ImageFolder(root=train_dir)
    val_dataset = datasets.ImageFolder(root=val_dir)

    train_classes = train_dataset.classes
    val_classes = val_dataset.classes
    if train_classes != val_classes:
        raise ValueError("Train/val class folder lists do not match exactly.")

    print("imagenet1k_layout_summary")
    print(f"data_root={data_root}")
    print(f"train_dir={train_dir}")
    print(f"val_dir={val_dir}")
    print(f"num_classes={len(train_classes)}")
    print(f"train_images={len(train_dataset)}")
    print(f"val_images={len(val_dataset)}")
    print(f"train_files_scanned={count_files(train_dir)}")
    print(f"val_files_scanned={count_files(val_dir)}")
    print(f"classes_head={train_classes[:10]}")


if __name__ == "__main__":
    main()
