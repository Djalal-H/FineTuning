# CLAUDE.md

## Communication

Use ASD-STE100 Simplified Technical English (STE) in all communication with the user (chat replies, explanations, summaries, and questions).

Apply these STE rules:

- Use only approved STE words with their approved meanings. Use technical names and technical verbs from the project domain (for example: model, dataset, tokenizer, fine-tune, checkpoint) when necessary.
- Use one word for one meaning. Do not use synonyms for the same thing.
- Keep sentences short:
  - Procedural sentences (instructions): 20 words maximum.
  - Descriptive sentences: 25 words maximum.
- Write one instruction in each sentence. Start instructions with a verb in the imperative form (for example: "Run the script.").
- Use the active voice. Do not use the passive voice in instructions.
- Use only these verb tenses: simple present, simple past, simple future, and the imperative.
- Do not use "-ing" forms of verbs, except in technical names.
- Use articles ("the", "a", "an") and demonstratives ("this", "these") when possible.
- Write paragraphs with one topic each, with a maximum of six sentences.
- Write the most important information first. Write warnings and cautions before the related instruction.
- Use vertical lists for steps and for groups of items.

Code, commands, file paths, and identifiers are not part of STE. Write them exactly as they are.

## Progress

The roadmap is in `PLAN.md`. Update this section at the end of each phase.

### Phase 0: done (one check is open)

What was done:

- The repo has a `src/mathscribe/` package, managed with uv. It has ruff, pre-commit, and pytest (37 tests, no GPU needed).
- `mathscribe.train.sft` is the training CLI, with `configs/train/sft_smoke.yaml` (the notebook run). It has `--set key=value`, `--resume`, and `--dry-run`.
- Checkpoints go to the HF Hub (`last-checkpoint/`) every 5 steps. `--resume` uses the local checkpoint or the Hub checkpoint, whichever has more steps.
- Each run writes `run_info.json` (GPU, wall-clock time, peak memory, parameter counts, library versions).
- `notebooks/colab_runner.ipynb` is a thin runner. Its `sh()` function stops the notebook at the first failed command.
- The README has no unverified numbers. `reports/phase0_reference.md` holds the notebook reference run.

What was achieved:

- The smoke run trained on Colab from the CLI and pushed the adapter to the HF Hub.

Check done (2026-11-17, on Kaggle during the Phase 2 check run):

- A run of 10 steps, then a new session with `--resume` to step 20, continued from the Hub checkpoint. The W&B run and the cost carried over.

Facts to remember:

- TRL >= 1.x accepts only a `datasets.Dataset`. `load_latex_ocr()` uses `with_transform()`, so the rows become conversations only when the trainer reads them.
- `import unsloth` must come before `torch`, `trl`, and `transformers` (see the `# isort: split` marker in `sft.py`).
- On Colab, the install and the training both use the kernel's Python (`sys.executable`).
- Without `USE_DRIVE = True`, a Colab disconnect deletes the local files. The Hub keeps the latest checkpoint.
- The notebook uses text-then-image order in training, but image-then-text order in inference. Use the training order everywhere.

### Phase 1: done (evaluation harness and baselines, one item is open)

What was done:

- Frozen manifests in `data/testsets/`: `latex_ocr_test.json` (1000 rows of `test`), `latex_ocr_val.json` (1000 rows of `train`), `crohme2014.json` (all 986 rows). `reports/data_splits.md` has the hashes and the overlap with the training rows (0.9%, 2.0%, 3.0%).
- Train configs pin `data.revision`. `data.exclude` removes manifest rows (the val set) from training. `sft_smoke.yaml` keeps `exclude: []` (the notebook rows).
- One prompt function: `inference_messages()` and `convert_to_conversation()` both use `user_content()` (text, then image).
- `eval/normalize.py`, `eval/metrics.py` (exact, normalized, CER, BLEU, buckets), `eval/render.py` (compile rate, render match), `eval/bootstrap.py` (95% intervals, paired Δ vs baseline).
- `renderer/` is a node render server (MathJax + resvg). `scripts/setup_env.sh` installs it (`npm ci`) in every mode.
- `run_eval.py` backends: `unsloth`, `anthropic`, `pix2tex`, `got_ocr`, `jsonl` (replay). It writes `reports/results/<run>__<testset>_n<N>.json` and builds `reports/leaderboard.md`.
- Extras: `api` (anthropic), `baselines` (pix2tex, GOT-OCR). `baselines` conflicts with `train` in `[tool.uv]`.

Done after the phase:

- `qwen35_4b_zeroshot`, `sft_smoke`, and `qwen35_9b_zeroshot` have results on both test sets, with render metrics (`reports/results/`).

Open item:

- Run `claude_opus55.yaml` locally. It needs an Anthropic credential.

Facts to remember:

- The Qwen3.5 chat template trains on `<think>\n\n</think>\n\n` + answer. Inference must use `enable_thinking: false`. Without it, the prompt ends with `<think>\n` and the model starts to think.
- `--subset N` uses the first N manifest rows. The rows are in random order, so each prefix is a random sample. Each N has its own sha256 and its own leaderboard table.
- The `jsonl` backend scores a stored `predictions.jsonl` again, with no GPU. Use it after a metric change.
- The Colab runner (`TASK = "eval"`) runs every model in `EVAL_RUNS` on every test set in `EVAL_TESTSETS`, each in its own process. It skips runs whose result exists (`SKIP_DONE`), and downloads `eval_results.zip` with repo paths (`unzip -o` in the repo root).
- Render match snaps glyph positions to whole pixels and removes empty pixel columns, so spaces do not count. Change `RENDERER_VERSION` in `renderer/server.mjs` when the render output changes: the cache key and the paired Δ use it.
- MathJax renders 99.5% of the `test` references. Rows whose reference does not render are not in the render match denominator (`ref_render_rate`).
- pix2tex: `LatexOCR()` defaults to the CPU and to sampling (temperature 0.2), and it sets the root logger to FATAL. `Pix2TexPredictor` fixes all three.
- GOT-OCR needs about 5.3 GB for a batch of 2. It wraps formulas in a one-cell `array` and adds text after `\]`. The normalizer and `extract_latex` handle both, for every model.
- The shell `grep` is an alias for `ugrep` with recursive options. In pipelines, use `command grep`.
- Each result stores `git_dirty`. The leaderboard shows `+` after the commit for runs with uncommitted changes.

### Phase 2: in progress (proper SFT, ablations, and tracking)

What was done:

- W&B tracking (`tracking` config section). Each checkpoint stores the W&B run ID and the train time, so `--resume` continues the same W&B run.
- The $ cost from `configs/gpu_prices.yaml` (`mathscribe.cost`). Auto-generated model cards on the HF Hub (`mathscribe.train.model_card`).
- Validation during training: `eval_loss` on 200 val rows, early stopping, and best-checkpoint selection (`configs/train/sft_base.yaml`).
- The check run on Kaggle (20 steps, `--resume` from the Hub) passed.
- Ablation configs: `configs/train/ablation_base.yaml` (4,000 rows, 500 steps) and `configs/ablations.yaml` (6 runs, plus a `skipped` list).
- The ablation runner `mathscribe.ablations`: one worker for each GPU, train then val eval, `--resume-if-available`, `--dry-run`.
- `NanGuard` stops a run at the first loss that is not finite. It does not save or push the adapter after that.
- `notebooks/kaggle_runner.ipynb` defaults are the real ablations (all runs, 500 steps, 500 val rows, no tag).
- The 6 ablation runs on Kaggle (2026-11-19/20). The files are in `reports/results/abl_*` and `reports/runs/`.
- `python -m mathscribe.ablation_report` builds `reports/ablations.md` (paired Δ vs `abl_base`, cost, skipped runs).

Ablation results (val, 500 rows):

- No run is better than the base run (LoRA r=16): 68.0% normalized match, 82.0% render match.
- rsLoRA is 2.6 points lower on normalized match. This is the only significant difference.
- QLoRA 4-bit has the same quality, with 4.5 GB peak VRAM instead of about 9 GB. It is about 20% slower per step.

Next steps:

1. Select the settings for the full SFT run, from `reports/ablations.md`.
2. Do the full SFT run with `sft_base.yaml` on Kaggle (`TASK = "train"`).
3. Evaluate it on val and on both test sets. Also run `qwen35_4b_zeroshot` on the val set.
4. Put the SFT vs zero-shot delta and the ablation table in the README.

Facts to remember:

- Compute is free Colab or Kaggle T4 only. A T4 has no bf16, and float16 does not work for qwen3_5, so 16-bit runs use float32.
- On a Kaggle T4, a step takes about 7 s with validation every 100 steps (the ablations), and about 14 s with validation every 5 steps (the check run). The peak memory was about 9.3 GB of 14.6 GB.
- `abl_base` and `abl_rslora` continued from the Hub at step 500 in a new session. Their run files have no best checkpoint, and their peak memory is from the 3 s last session.
- The Kaggle zip has repo paths. Unzip it in the repo root (`unzip -o`). Do not move the unpacked folder into the repo.
- DoRA does not train here: the loss was `nan` from step 1 (Kaggle, 2026-11-17). It is in the `skipped` list.
- The Kaggle notebook clones `BRANCH` from GitHub. Push all changes before a Kaggle run.
- Early stopping has unit tests only. No GPU run triggered it.
