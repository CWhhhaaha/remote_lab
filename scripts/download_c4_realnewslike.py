#!/usr/bin/env python3
"""Download C4 realnewslike subset to ~/datasets/text/"""

import os
from datasets import load_dataset

save_dir = os.path.expanduser("~/datasets/text/c4-realnewslike")
os.makedirs(os.path.dirname(save_dir), exist_ok=True)

print(f"Downloading C4 realnewslike to {save_dir} ...")
print("This is ~15GB and may take 30-60 minutes.")

ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=False)
ds.save_to_disk(save_dir)

print(f"Done! Saved to {save_dir}")
print(f"Dataset size: {len(ds)} examples")
