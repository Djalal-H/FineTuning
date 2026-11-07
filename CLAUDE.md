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

Open check:

- Stop a run after step 5, then run it again with `--resume`. Make sure that it continues from the Hub checkpoint.

Facts to remember:

- TRL >= 1.x accepts only a `datasets.Dataset`. `load_latex_ocr()` uses `with_transform()`, so the rows become conversations only when the trainer reads them.
- `import unsloth` must come before `torch`, `trl`, and `transformers` (see the `# isort: split` marker in `sft.py`).
- On Colab, the install and the training both use the kernel's Python (`sys.executable`).
- Without `USE_DRIVE = True`, a Colab disconnect deletes the local files. The Hub keeps the latest checkpoint.
- The notebook uses text-then-image order in training, but image-then-text order in inference. Use the training order everywhere.

### Phase 1: done (evaluation harness and baselines)

What was done:

- Frozen manifests in `data/testsets/`: `latex_ocr_test.json` (1000 rows of `test`), `latex_ocr_val.json` (1000 rows of `train`), `crohme2014.json` (all 986 rows). `reports/data_splits.md` has the hashes and the overlap with the training rows (0.9%, 2.0%, 3.0%).
- Train configs pin `data.revision`. `data.exclude` removes manifest rows (the val set) from training. `sft_smoke.yaml` keeps `exclude: []` (the notebook rows).
- One prompt function: `inference_messages()` and `convert_to_conversation()` both use `user_content()` (text, then image).
- `eval/normalize.py`, `eval/metrics.py` (exact, normalized, CER, BLEU, buckets), `eval/render.py` (compile rate, render match), `eval/bootstrap.py` (95% intervals, paired Δ vs baseline).
- `renderer/` is a node render server (MathJax + resvg). `scripts/setup_env.sh` installs it (`npm ci`) in every mode.
- `run_eval.py` backends: `unsloth`, `anthropic`, `pix2tex`, `got_ocr`, `jsonl` (replay). It writes `reports/results/<run>__<testset>_n<N>.json` and builds `reports/leaderboard.md`.
- Extras: `api` (anthropic), `baselines` (pix2tex, GOT-OCR). `baselines` conflicts with `train` in `[tool.uv]`.

Open items:

- Run on Colab, both test sets: `qwen35_4b_zeroshot`, `sft_smoke`, `qwen35_9b_zeroshot`. The old n=1000 Qwen results have no render metrics.
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
