# MathScribe: Roadmap from Notebook to Production-Grade Project

## Context

Right now the repo has only `Qwen_3_5_SFT.ipynb` and `README.md`. The notebook is essentially the Unsloth vision tutorial: Qwen3.5-4B + LoRA (r=16, vision and language layers) on `unsloth/LaTeX_OCR`, `max_steps=15`, no eval split, no metrics, no baselines. The README reports numbers (94.5% accuracy, 2.1x speed, 6.4 GB) that nothing in the repo backs up.

**Goal:** turn it into **MathScribe** (handwritten/printed math → LaTeX), a project that shows:

- rigorous evaluation,
- advanced fine-tuning (ablations, GRPO with a compile-and-render reward, DPO),
- full MLOps (tracking, CI eval gate, quantized vLLM serving, monitoring, a feedback loop).

## Guiding principles

- Every number in the README is produced by a script in this repo and logged to the experiment tracker.
- Frozen test sets, versioned data, reproducible seeded configs.
- Notebooks are only for exploration. All real logic lives in `src/mathscribe/`.

## Compute strategy

The local GPU has only 6 GB of VRAM, so all training and full-scale evaluation runs remotely. The code does not depend on the machine: every job is a CLI command with a config file.

| Task                                                   | Local (6 GB) | Colab (T4 free / L4–A100 Pro)            | Modal / RunPod (hourly) |
| ------------------------------------------------------ | ------------ | ----------------------------------------- | ----------------------- |
| Code, unit tests, metrics, `render.py`, rewards, API | ✅           | —                                        | —                      |
| Data pipeline, synthetic data engine                   | ✅ (CPU)     | —                                        | —                      |
| pix2tex baseline, eval on a small subset               | ✅           | —                                        | —                      |
| GGUF inference (llama.cpp)                             | ✅           | —                                        | —                      |
| SFT, DPO, ablations                                    | ❌           | ✅                                        | optional                |
| GRPO                                                   | ❌           | ⚠️ T4 is slow; L4/A100 recommended      | ✅ recommended          |
| Full eval of the 4B model on all test sets             | slow         | ✅                                        | ✅                      |
| CI eval gate, scheduled retrain, serving               | ❌           | ❌ (needs a person to start each session) | ✅                      |

Rules:

- **Colab is a thin runner.** `notebooks/colab_runner.ipynb` only mounts Drive, clones the repo, runs `uv pip install -e ".[train]"`, logs in to HF/W&B, and runs the same CLI command used locally. No logic goes in that notebook.
- **Sessions are disposable.** Colab can stop at any time. Push checkpoints to the HF Hub (or Drive) every N steps, and make all train scripts accept `--resume`.
- **Same environment everywhere.** The renderer (`texlive` or node + KaTeX) installs with a single setup script that runs both locally and on Colab, because the GRPO reward and the eval need it.
- **Cost is measured in GPU hours.** Every tracker run logs the GPU type and wall-clock time. The $ cost is derived from public hourly prices.

## Target repo layout

```
mathscribe/
├── pyproject.toml            # uv-managed; extras: train, serve, dev
├── configs/                  # Hydra/YAML: model/, data/, train/, eval/
├── src/mathscribe/
│   ├── data/                 # loaders, conversation formatting, synthetic engine, augmentation
│   ├── train/                # sft.py, grpo.py, dpo.py
│   ├── eval/                 # metrics.py, render.py, normalize.py, run_eval.py
│   ├── rewards/              # compile/render rewards for GRPO
│   ├── export/               # merge LoRA, AWQ, GGUF, push to HF Hub
│   └── serve/                # FastAPI app, vLLM client, retry-on-compile-fail
├── notebooks/                # Qwen_3_5_SFT.ipynb (exploration) + colab_runner.ipynb (thin remote runner)
├── scripts/                  # setup_env.sh (deps + LaTeX renderer, local and Colab)
├── modal_app/                # Modal functions: eval-gate, scheduled retrain, serving
├── tests/                    # unit tests for metrics, normalization, rewards, API
├── docker/                   # Dockerfile.serve, docker-compose (vLLM + API + Prometheus/Grafana)
├── .github/workflows/        # lint+test, eval-gate, build-and-push image
├── demo/                     # Gradio HF Space (draw on a canvas → LaTeX + render)
└── reports/                  # leaderboard.md, ablations.md, load_test.md (auto-generated)
```

---

## Phase 0: Honest baseline and restructure (≈2 days)

- [X] Move the notebook to `notebooks/`. Add `pyproject.toml` (uv), ruff, pre-commit and pytest.
- [X] Remove the unverifiable metrics from the README and point to `reports/` instead.
- [X] Port data formatting (`convert_to_conversation`, instruction string) and model/LoRA loading from the notebook into `src/mathscribe/data/` and `src/mathscribe/train/sft.py`.
- [X] Add `notebooks/colab_runner.ipynb` and `scripts/setup_env.sh` (see [Compute strategy](#compute-strategy)).
- [X] Add checkpoint push to the HF Hub every N steps, and a `--resume` flag in `sft.py`.

**Done when:** `uv run python -m mathscribe.train.sft --config configs/train/sft_smoke.yaml` reproduces the notebook run from the command line, launched from `colab_runner.ipynb`, and a run that is stopped in the middle resumes from the last Hub checkpoint.

## Phase 1: Evaluation harness and baselines (≈1 week) ⭐ foundation

- [ ] Fixed train/val/test splits of `unsloth/LaTeX_OCR` (seeded, with hashes recorded). Add **CROHME** as a handwritten test set.
- [ ] `normalize.py`: canonical form for LaTeX (whitespace, `\left`/`\right`, redundant braces).
- [ ] `render.py`: LaTeX → PNG (KaTeX/MathJax via node, or `pdflatex` + `pdf2image`), with a timeout and caching.
- [ ] `metrics.py`: exact match, normalized match, CER, BLEU, **compile rate**, **render match** (SSIM or pixel difference between rendered images).
- [ ] Difficulty buckets (length, matrices, nested fractions) with a score for each bucket.
- [ ] Baselines: zero-shot Qwen3.5-4B, a larger Qwen-VL, pix2tex, Texify/GOT-OCR, and one API model. Run pix2tex and the API model locally. Run the VLM baselines on Colab.
- [ ] `run_eval.py --subset N` for fast local smoke checks on a small frozen subset.
- [ ] `reports/leaderboard.md`, generated automatically by `run_eval.py`.

**Done when:** one command evaluates any checkpoint or baseline and writes the leaderboard, and the metrics have unit tests.

## Phase 2: Proper SFT, ablations and tracking (≈1 week)

Decisions (2026-11-07):

- **Compute: free Colab (T4) only.** A T4 needs about 17 s per step (batch 8, float32: Unsloth says float16 does not work for qwen3_5, and the T4 has no bf16). One epoch of the full split is about 8,600 steps (about 40 h), so the full run uses a fixed subset.
- **Tracker: W&B.** No server, free, and TRL supports it.
- **DVC moves to Phase 3.** The data has versions already (pinned HF revision + frozen manifests with sha256). Phase 3 makes local synthetic data, and DVC is useful there.

Items:

- [X] W&B logging for train and eval runs (`tracking` config section). The W&B run ID and the train time go into each checkpoint, so `--resume` continues the same W&B run and the cost counts all sessions.
- [X] $ cost from public GPU prices (`configs/gpu_prices.yaml`, `mathscribe.cost`).
- [X] HF Hub as the model registry with auto-generated model cards (`mathscribe.train.model_card`).
- [X] Validation during training: `eval_loss` on 200 val rows, early stopping, best-checkpoint selection (`configs/train/sft_base.yaml`).
- [X] Short check run (20 steps, on Kaggle: `notebooks/kaggle_runner.ipynb`) for validation, best-checkpoint selection, W&B, the model card and `--resume` from the Hub. About 14 s per step on a T4 (validation included), peak 9.3 GB of 14.6 GB. Early stopping has unit tests only: the check run did not trigger it.
- [X] Ablation configs: `configs/train/ablation_base.yaml` (4,000 rows, 500 steps, no early stopping) and `configs/ablations.yaml` (7 runs, one setting each). `lora.use_dora` is a config key now. Unsloth passes `use_dora` to PEFT. **DoRA does not train** here: the Kaggle smoke test (2026-11-17, T4, 16-bit, vision layers on) gave a `nan` loss from step 1. It is in the `skipped` list of `configs/ablations.yaml`. The runner stops a run at the first non-finite loss (`NanGuard`), so a bad setting cannot use hours of quota.
- [X] Ablation runner `mathscribe.ablations` (`python -m mathscribe.ablations --dry-run` checks every config with no GPU). Each run trains, then evaluates on val. Jobs share a queue, one worker for each GPU, so two Kaggle T4 GPUs run two ablations at the same time. It skips finished steps, and `--resume-if-available` continues from the local or Hub checkpoint after a stop. `notebooks/kaggle_runner.ipynb` (`TASK = "ablation"`) starts it. The notebook defaults are a smoke test (10 steps, tag `smoke`).
- [X] One-factor-at-a-time ablations on a fixed subset (4,000 rows, 500 steps), compared on the **val** set only (500 rows): base LoRA r=16; rsLoRA; r ∈ {8, 64}; vision layers off; QLoRA 4-bit vs 16-bit (float32 on a T4). DoRA is skipped (`nan` loss). Done on Kaggle (2026-11-19/20, two T4 GPUs): about 7 s per step, about 1 GPU hour each. Result: no run is better than the base run. rsLoRA is 2.6 points lower on normalized match (significant). QLoRA 4-bit has the same quality with 4.5 GB peak VRAM instead of about 9 GB.
- [ ] Full SFT run (`sft_base.yaml`: 12,000 rows, 1 epoch, 1,500 steps). **Not done:** the phase closed without it. The config is ready. The Phase 2 adapter is `abl_base` (`djaloul/mathscribe-abl-base`).
- [X] `reports/ablations.md`, generated from the committed result JSONs and `run_info.json` files, with a W&B link for each row (no W&B key needed to build it): `python -m mathscribe.ablation_report`. It shows the paired Δ vs the base run.

**Done when:** the README shows a real SFT vs zero-shot delta and the ablation table, all linked to tracker runs.

**Closed (2026-11-20).** The README shows the ablation table and the `sft_smoke` vs zero-shot delta on both test sets. The full SFT run was not done, to save time.

## Phase 3: Advanced training: GRPO, DPO, synthetic data (≈1–2 weeks) ⭐ headline

- [ ] `rewards/`: compile reward (0/1), render-similarity reward (SSIM against the target render), length/format penalties.
- [ ] GRPO from the best SFT checkpoint (`abl_base`, see Phase 2) (TRL/Unsloth GRPOTrainer), tracking reward curves. GRPO generates several samples for each image, so it needs much more memory and time than SFT. Run it on Colab Pro (L4/A100) or a rented GPU (Modal/RunPod), with QLoRA and a small group size. Fallback: run GRPO on a 2B model.
- [ ] DPO on hard negatives: sample N outputs and pair correct ones with ones that compile but render wrong.
- [ ] DVC for the local datasets (moved from Phase 2): the synthetic data and its renders.
- [ ] Synthetic data engine: LaTeX from arXiv sources → render with varied fonts → augment (blur, rotation, paper texture, ink strokes), with a curriculum ordered by difficulty.
- [ ] Optional: distill to a small VLM (0.8–2B or SmolVLM) for CPU/WebGPU inference.

**Done when:** the leaderboard shows the gains from SFT → SFT+GRPO → SFT+GRPO+DPO on both the printed and the CROHME test sets.

## Phase 4: Export and serving (≈1 week)

- [ ] `export/`: merge the LoRA; export AWQ (GPU) and GGUF (llama.cpp/Ollama). Re-evaluate each export and report the quality lost to quantization. Run the merge and the AWQ export on Colab. Test the GGUF locally on the 6 GB GPU.
- [ ] vLLM OpenAI-compatible server. Optional multi-LoRA serving (printed / handwritten / chemistry adapters).
- [ ] FastAPI layer: image upload, request batching, **compile check with a sampling retry**, caching, structured logs.
- [ ] Docker + docker-compose; deploy to Modal, RunPod or Cloud Run.
- [ ] Load test (locust/k6) → `reports/load_test.md`: p50/p95 latency, throughput, **$ per 1k images**.

**Done when:** the public endpoint and HF Space demo are live and the load-test report is committed.

## Phase 5: MLOps loop (≈1 week)

- [ ] CI: lint and unit tests on every PR. An **eval-gate** workflow runs a frozen subset and blocks promotion if render match drops below a threshold. GitHub Actions runners have no GPU, so the workflow calls a Modal function (`modal_app/`) that runs the eval on a GPU and returns the metrics. Fallback: run the GGUF model on CPU with a small frozen subset (about 100 images).
- [ ] Build and push the image on tag. Registry promotion (staging → prod) depends on the eval-gate result.
- [ ] Monitoring with Prometheus + Grafana: latency, error rate, **live compile-success rate** (a quality signal that needs no labels), input drift (image statistics).
- [ ] Feedback loop: demo users correct outputs → corrections stored as DPO pairs → scheduled retrain → eval gate → promote. Colab cannot run on a schedule, so the scheduled retrain is a Modal cron job. Fallback: start the retrain manually on Colab, and keep the eval gate and the promotion automatic.

**Done when:** the full loop (feedback → retrain → gated promotion → redeploy) has run end to end at least once.

## Phase 6: Polish and product (stretch)

- [ ] A screenshot-to-LaTeX hotkey tool, or an Obsidian/VS Code plugin that calls the API.
- [ ] An in-browser distilled model via transformers.js/WebGPU.
- [ ] README rewrite: architecture diagram, leaderboard, ablations, demo GIF, plus a short tech blog post.

---

## Target resume line

> Fine-tuned Qwen3.5-4B for math OCR with SFT + GRPO using a compile-and-render reward. It beats [baseline] by X points render match at 1/10 the size. Served with vLLM (p95 Y ms, $Z per 1k images), with CI eval gating, drift monitoring and a user-feedback retraining loop.
