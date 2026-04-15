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

## Notes

- `requirements.txt` is intentionally minimal right now. Add runtime dependencies there as the experiments become concrete.
- `scripts/setup_venv.sh` installs the package in editable mode, so imports stay consistent on both local and remote machines.
- The current CLI is a bootstrap entrypoint to verify that the environment, config path, and output directory wiring all work before the real training code is added.
