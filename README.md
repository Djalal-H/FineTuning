# MathScribe

Handwritten and printed math → LaTeX, with a fine-tuned vision-language model
(Qwen3.5-4B + LoRA, trained with [Unsloth](https://github.com/unslothai/unsloth)).

> **Status: Phase 0 (restructure).** The repo now has a reproducible SFT command.
> It has **no evaluated results yet**. See [PLAN.md](PLAN.md) for the roadmap.

## Results

There are no verified quality numbers yet. Every number about this project
comes from a script in this repo, and lives in [`reports/`](reports/):

- [`reports/phase0_reference.md`](reports/phase0_reference.md): the reference
  smoke run from the original notebook (training cost only, no quality metrics).
- The leaderboard (Phase 1) and the ablations (Phase 2) come later.

## Repo layout

```
configs/train/        YAML run configs (sft_smoke.yaml = the notebook run)
src/mathscribe/
  config.py           config loading + --set overrides
  data/               dataset loading and chat formatting
  train/sft.py        SFT entry point (Unsloth + TRL), Hub checkpoints, --resume
notebooks/
  Qwen_3_5_SFT.ipynb  original exploration notebook
  colab_runner.ipynb  thin remote runner (clone → setup → CLI)
scripts/setup_env.sh  one setup script for local and Colab
tests/                unit tests (no GPU needed)
reports/              generated reports
```

## Quickstart

### Local (code, tests, no training)

```bash
bash scripts/setup_env.sh          # uv venv + dev tools + pre-commit hooks
uv run pytest
uv run python -m mathscribe.train.sft --config configs/train/sft_smoke.yaml --dry-run
```

### Training (Colab or any CUDA GPU with ≥ 16 GB)

Open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) in Colab, set
`HUB_REPO`, and run all cells. The notebook runs this command:

```bash
python -m mathscribe.train.sft --config configs/train/sft_smoke.yaml \
  --set hub.push=true --set hub.repo_id=<hf-user>/mathscribe-sft-smoke
```

On a machine with the train extra installed (`bash scripts/setup_env.sh --train`),
use `uv run python -m mathscribe.train.sft ...` instead.

### Checkpoints and resume

- The trainer saves a checkpoint every `train.save_steps` steps.
- With `hub.push=true`, each checkpoint also goes to the Hub repo, in `last-checkpoint/`.
- Add `--resume` to continue a stopped run. The script picks the most advanced
  checkpoint from the local `output_dir` and the Hub.
- Do not change `max_steps` or the batch settings when you resume. The schedule
  in the checkpoint depends on them.

Override any config value with `--set section.key=value` (for example
`--set train.max_steps=30`).

## License

MIT.
