#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import torch
from transformers import BertTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline tokenize prepared Jigsaw json.gz files into torch caches."
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        required=True,
        help="Path to prepared train json.gz file.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        required=True,
        help="Path to prepared test json.gz file.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        required=True,
        help="Output path for tokenized train cache (.pt).",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        required=True,
        help="Output path for tokenized test cache (.pt).",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="bert-base-uncased",
        help="Hugging Face tokenizer name.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length after truncation.",
    )
    return parser.parse_args()


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def extract_texts(payload: dict[str, Any], split: str) -> list[str]:
    rows = payload.get(f"{split}_samples")
    if not isinstance(rows, list):
        raise ValueError(f"Expected {split}_samples list")
    return [row["text"] for row in rows if isinstance(row, dict) and row.get("text")]


def tokenize_texts(
    texts: list[str],
    tokenizer: BertTokenizerFast,
    max_length: int,
) -> list[dict[str, list[int]]]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )
    examples: list[dict[str, list[int]]] = []
    for idx in range(len(texts)):
        examples.append(
            {
                "input_ids": encoded["input_ids"][idx],
                "attention_mask": encoded["attention_mask"][idx],
                "special_tokens_mask": encoded["special_tokens_mask"][idx],
            }
        )
    return examples


def write_cache(
    output_path: Path,
    *,
    split: str,
    tokenizer_name: str,
    max_length: int,
    examples: list[dict[str, list[int]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "num_examples": len(examples),
        "examples": examples,
    }
    torch.save(payload, output_path)


def main() -> None:
    args = parse_args()
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)

    train_payload = load_json_gz(args.train_input)
    test_payload = load_json_gz(args.test_input)

    train_texts = extract_texts(train_payload, "train")
    test_texts = extract_texts(test_payload, "test")

    train_examples = tokenize_texts(train_texts, tokenizer, args.max_length)
    test_examples = tokenize_texts(test_texts, tokenizer, args.max_length)

    write_cache(
        args.train_output,
        split="train",
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        examples=train_examples,
    )
    write_cache(
        args.test_output,
        split="test",
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        examples=test_examples,
    )

    print(f"train_output={args.train_output.resolve()}")
    print(f"test_output={args.test_output.resolve()}")
    print(f"tokenizer_name={args.tokenizer_name}")
    print(f"max_length={args.max_length}")
    print(f"train_examples={len(train_examples)}")
    print(f"test_examples={len(test_examples)}")


if __name__ == "__main__":
    main()
