#!/usr/bin/env python3
"""Train GPT-2 small (124M) on C4 realnewslike with custom attention variants.

Example:
    python -m remote_lab.train_gpt2_c4 \
        --variant baseline --output-dir runs/gpt2_c4_baseline_50k

    python -m remote_lab.train_gpt2_c4 \
        --variant lowrank --rank 32 --output-dir runs/gpt2_c4_lowrank_r32_50k

    python -m remote_lab.train_gpt2_c4 \
        --variant bmbuv --rank 32 --factor-rank 32 \
        --output-dir runs/gpt2_c4_bmbuv_r32s32_50k
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from remote_lab.gpt2_attention_variants import replace_gpt2_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 on C4 with attention variants")
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=["baseline", "lowrank", "fullyshared", "bmbuv", "partialshared"],
        help="Attention variant to use",
    )
    parser.add_argument("--rank", type=int, default=32, help="Rank for lowrank/bmbuv")
    parser.add_argument("--factor-rank", type=int, default=None, help="Factor rank for bmbuv (defaults to rank)")
    parser.add_argument("--shared-dim", type=int, default=48, help="Shared qk dim for partialshared")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset-path", type=str, default="~/datasets/text/c4-realnewslike", help="Path to C4 dataset")
    parser.add_argument("--max-steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--per-device-batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Linear warmup steps")
    parser.add_argument("--seq-length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16 mixed precision")
    parser.add_argument("--no-bf16", action="store_false", dest="bf16", help="Disable bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Use gradient checkpointing")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    return parser.parse_args()


def load_or_download_c4(dataset_path: str):
    dataset_path = os.path.expanduser(dataset_path)
    if os.path.exists(dataset_path):
        print(f"Loading C4 from {dataset_path}")
        ds = load_from_disk(dataset_path)
    else:
        print(f"Dataset not found at {dataset_path}. Downloading from HuggingFace...")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=False)
        ds.save_to_disk(dataset_path)
        print(f"Saved to {dataset_path}")
    return ds


def tokenize_and_group(examples, tokenizer, seq_length: int):
    """Tokenize texts and group into fixed-length chunks."""
    # Concatenate all texts
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    # Drop remainder
    total_length = (total_length // seq_length) * seq_length
    result = {
        k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
        for k, t in concatenated.items()
    }
    return result


def compute_parameter_summary(model: GPT2LMHeadModel) -> dict:
    """Count total and attention-specific parameters."""
    total = sum(p.numel() for p in model.parameters())
    attn = 0
    for name, p in model.named_parameters():
        if "attn" in name:
            attn += p.numel()
    return {"total_params": total, "attention_params": attn, "non_attention_params": total - attn}


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load / download C4
    ds = load_or_download_c4(args.dataset_path)

    # C4 realnewslike only has 'train' split; create a small validation split
    ds = ds.train_test_split(test_size=0.005, seed=args.seed)  # 0.5% for val
    train_ds = ds["train"]
    val_ds = ds["test"]
    print(f"Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    # 3. Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=False, add_special_tokens=False)

    train_ds = train_ds.map(tokenize_fn, batched=True, num_proc=8, remove_columns=["text", "timestamp", "url"])
    val_ds = val_ds.map(tokenize_fn, batched=True, num_proc=8, remove_columns=["text", "timestamp", "url"])

    # 4. Group into chunks
    block_size = args.seq_length

    def group_fn(examples):
        return tokenize_and_group(examples, tokenizer, block_size)

    train_ds = train_ds.map(group_fn, batched=True, num_proc=8)
    val_ds = val_ds.map(group_fn, batched=True, num_proc=8)

    # 5. Build model
    config = GPT2Config()
    model = GPT2LMHeadModel(config)

    variant_kwargs = {}
    if args.variant == "lowrank":
        variant_kwargs = {"rank": args.rank}
    elif args.variant == "bmbuv":
        variant_kwargs = {"rank": args.rank, "factor_rank": args.factor_rank or args.rank}
    elif args.variant == "partialshared":
        variant_kwargs = {"shared_dim": args.shared_dim}

    if args.variant != "baseline":
        replace_gpt2_attention(model, args.variant, **variant_kwargs)
        print(f"Replaced attention with variant: {args.variant}, kwargs: {variant_kwargs}")

    param_info = compute_parameter_summary(model)
    print(f"Total params: {param_info['total_params']:,}")
    print(f"Attention params: {param_info['attention_params']:,}")

    # 6. Training arguments
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        seed=args.seed,
        report_to="none",  # disable wandb/tensorboard by default
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # 7. Train
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    # 8. Final eval
    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"])).item()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    print(f"Final perplexity: {perplexity:.2f}")

    # Save summary
    import json
    summary = {
        "variant": args.variant,
        "variant_kwargs": variant_kwargs,
        "param_info": param_info,
        "eval_loss": eval_results["eval_loss"],
        "perplexity": perplexity,
        "training_args": training_args.to_dict(),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
