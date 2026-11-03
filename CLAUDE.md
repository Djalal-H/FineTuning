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

### Phase 1: in progress (evaluation harness and baselines)

Steps:

1. Done. `data/testsets/latex_ocr_test.json` freezes 1000 rows of the `test` split (seed 2026, dataset revision `4da395c8`, sha256 `ca73bfe6…`).
2. Done. `inference_messages()` and `convert_to_conversation()` both use `user_content()` (text, then image).
3. Done. `eval/normalize.py` and `eval/metrics.py` (exact match, normalized match, CER, BLEU, difficulty buckets), with unit tests.
4. Code done. `python -m mathscribe.eval.run_eval --subset N`. Open: run the zero-shot base model and the smoke adapter on Colab (`TASK = "eval"`).
5. Next. Add the LaTeX renderer (`render.py`) to `scripts/setup_env.sh`. Then add compile rate and render match.
6. Next. Add the other baselines (pix2tex, a larger Qwen-VL, an API model) and CROHME.
7. Done. `run_eval` writes `reports/results/<run>__<testset>_n<N>.json` and builds `reports/leaderboard.md`.

Facts to remember:

- The Qwen3.5 chat template trains on `<think>\n\n</think>\n\n` + answer. Inference must use `enable_thinking: false`. Without it, the prompt ends with `<think>\n` and the model starts to think.
- `--subset N` uses the first N manifest rows. The rows are in random order, so each prefix is a random sample. Each N has its own sha256 and its own leaderboard table.
- The `jsonl` backend scores a stored `predictions.jsonl` again, with no GPU. Use it after a metric change.
- The `UnslothPredictor` (batched generation, left padding) has no GPU test yet. The first Colab run is its test.
