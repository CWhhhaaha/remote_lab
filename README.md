# remote_lab

This repository is set up for a simple workflow:

- write and commit code locally
- push to GitHub
- pull on the remote machine
- run experiments inside a project-local virtual environment

Private manuscript sources stay local and are excluded from version control.

## Repository layout

- `src/remote_lab/`: Python package and experiment entrypoints
- `configs/`: committed experiment configs
- `scripts/`: setup and run helpers for local or remote machines
- `data/`: local-only raw and prepared datasets for training
- `runs/`, `logs/`, `results/`: generated outputs kept out of git

## Local workflow

```bash
cd /Users/admin/Downloads/my_lab/remote_lab
bash scripts/setup_venv.sh
source .venv/bin/activate
python -m remote_lab.cli --dry-run
git add .
git commit -m "Describe your change"
git push
```

## Remote workflow

```bash
cd ~/remote_lab
git pull
bash scripts/setup_venv.sh
source .venv/bin/activate
bash scripts/run_experiment.sh --config configs/base.json --output-dir runs/first_run
```

## Data workflow

The repository expects datasets to live under a local-only `data/` tree that is ignored by git.

Recommended layout:

```bash
data/
  raw/
    jigsaw/
      train.csv.zip
      test.csv.zip
      test_labels.csv.zip
      sample_submission.csv.zip
  processed/
    train/
      jigsaw.json.gz
    test/
      jigsaw.json.gz
  cache/
    jigsaw/
      train_tokens.pt
      test_tokens.pt
```

Prepare the directory tree:

```bash
bash scripts/setup_data_dirs.sh
```

On the remote machine, download the Kaggle Jigsaw csv.zip files into `data/raw/jigsaw`:

```bash
source .venv/bin/activate
python -m pip install kaggle
bash scripts/download_jigsaw_kaggle.sh
```

This requires a working Kaggle credential setup on the remote machine.

Convert the raw archive into the prepared train/test files expected by the experiment config:

```bash
python scripts/prepare_jigsaw_data.py \
  --raw-dir data/raw/jigsaw \
  --train-output data/processed/train/jigsaw.json.gz \
  --test-output data/processed/test/jigsaw.json.gz
```

Then create the offline token caches used by training:

```bash
python scripts/tokenize_jigsaw.py \
  --train-input data/processed/train/jigsaw.json.gz \
  --test-input data/processed/test/jigsaw.json.gz \
  --train-output data/cache/jigsaw/train_tokens.pt \
  --test-output data/cache/jigsaw/test_tokens.pt \
  --max-length 128
```

Or use the helper script:

```bash
bash scripts/prepare_jigsaw_cache.sh
```

Check that the experiment config resolves the expected local dataset paths:

```bash
python -m remote_lab.cli \
  --config configs/experiments/jigsaw_bert_small_encoder_baseline_v1.json \
  --dry-run
```

## Notes

- Training now expects offline token caches under `data/cache/jigsaw/` and no longer tokenizes raw text inside the training loop.
- `scripts/setup_venv.sh` installs the package in editable mode, so imports stay consistent on both local and remote machines.
- The training code still downloads the tokenizer metadata on first use if it is not already cached locally by Hugging Face.
- In redirected logs, training prints one `[epoch_summary]` line after each epoch so you can monitor loss, regularization, timing, and layerwise ratios with `tail -f`.
- To launch baseline on GPU 2 and interval-reg on GPU 3 in one command, run `bash scripts/launch_jigsaw_pair.sh` from an activated environment. Override GPUs with `BASELINE_GPU=... INTERVAL_GPU=...`.
- To launch the symmetric-initialization run on GPU 4, run `bash scripts/launch_jigsaw_symm.sh`. Override the GPU with `SYMM_GPU=...`.
