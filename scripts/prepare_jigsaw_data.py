#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import zipfile
from pathlib import Path


LABEL_NAMES = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle Jigsaw csv.zip files into train/test json.gz files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing train.csv.zip, test.csv.zip, and test_labels.csv.zip.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        required=True,
        help="Output path for the prepared train json.gz file.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        required=True,
        help="Output path for the prepared test json.gz file.",
    )
    return parser.parse_args()


def read_zipped_csv(zip_path: Path, csv_name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        with io.TextIOWrapper(archive.open(csv_name, "r"), encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def dict_label(row: dict[str, str]) -> int:
    values = [int(row[label]) for label in LABEL_NAMES]
    if any(value == -1 for value in values):
        return -1
    if any(value == 1 for value in values):
        return 1
    return 0


def build_train_dataset(rows: list[dict[str, str]]) -> dict:
    return {
        "name": "jigsaw",
        "train_samples": [
            {
                "id": f"jigsaw-{row['id']}",
                "text": row["comment_text"],
                "label": dict_label(row) == 1,
            }
            for row in rows
            if dict_label(row) != -1
        ],
        "dev_samples": None,
    }


def build_test_dataset(
    test_rows: list[dict[str, str]],
    test_label_rows: list[dict[str, str]],
) -> dict:
    label_map = {row["id"]: row for row in test_label_rows}
    return {
        "name": "jigsaw",
        "test_samples": [
            {
                "id": f"jigsaw-{row['id']}",
                "text": row["comment_text"],
                "label": dict_label(label_map[row["id"]]) == 1,
            }
            for row in test_rows
            if dict_label(label_map[row["id"]]) != -1
        ],
    }


def write_json_gz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    required_files = {
        "train": args.raw_dir / "train.csv.zip",
        "test": args.raw_dir / "test.csv.zip",
        "test_labels": args.raw_dir / "test_labels.csv.zip",
    }
    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required Jigsaw file for {name}: {path}")

    train_rows = read_zipped_csv(required_files["train"], "train.csv")
    test_rows = read_zipped_csv(required_files["test"], "test.csv")
    test_label_rows = read_zipped_csv(required_files["test_labels"], "test_labels.csv")

    train_payload = build_train_dataset(train_rows)
    test_payload = build_test_dataset(test_rows, test_label_rows)

    write_json_gz(args.train_output, train_payload)
    write_json_gz(args.test_output, test_payload)

    print(f"train_output={args.train_output.resolve()}")
    print(f"test_output={args.test_output.resolve()}")
    print(f"train_samples={len(train_payload['train_samples'])}")
    print(f"test_samples={len(test_payload['test_samples'])}")


if __name__ == "__main__":
    main()
