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

- [ ] W&B or MLflow logging, DVC for datasets, HF Hub as the model registry with auto-generated model cards.
- [ ] Full SFT run: epochs, evaluation on val, early stopping, best-checkpoint selection.
- [ ] One-factor-at-a-time ablations (about 6 runs, not a 36-run full grid) from one base config: LoRA vs DoRA vs rsLoRA; r ∈ {8, 16, 64}; vision layers on/off; QLoRA 4-bit vs bf16. Run them on a fixed training subset to fit the Colab budget. Log quality, peak VRAM, GPU type, GPU hours and the derived $ cost.
- [ ] `reports/ablations.md`, generated from tracker runs.

**Done when:** the README shows a real SFT vs zero-shot delta and the ablation table, all linked to tracker runs.

## Phase 3: Advanced training: GRPO, DPO, synthetic data (≈1–2 weeks) ⭐ headline

- [ ] `rewards/`: compile reward (0/1), render-similarity reward (SSIM against the target render), length/format penalties.
- [ ] GRPO from the best SFT checkpoint (TRL/Unsloth GRPOTrainer), tracking reward curves. GRPO generates several samples for each image, so it needs much more memory and time than SFT. Run it on Colab Pro (L4/A100) or a rented GPU (Modal/RunPod), with QLoRA and a small group size. Fallback: run GRPO on a 2B model.
- [ ] DPO on hard negatives: sample N outputs and pair correct ones with ones that compile but render wrong.
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
