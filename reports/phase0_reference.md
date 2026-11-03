# Phase 0 reference run

This is the target for the Phase 0 "done" check: `configs/train/sft_smoke.yaml`
must reproduce this run from the command line.

**Source:** the saved cell outputs of
[`notebooks/Qwen_3_5_SFT.ipynb`](../notebooks/Qwen_3_5_SFT.ipynb) (cells 3, 24, 25 and 26).
These numbers measure training cost only. They do not measure model quality.

## Environment

| Item | Value |
|---|---|
| GPU | Tesla T4, 14.563 GB |
| Unsloth | 2026.4.5 |
| Transformers | 5.5.0 |
| Torch | 2.10.0+cu128 (CUDA toolkit 12.8) |
| Precision | float32 (Unsloth: "float16 precision for qwen3_5 won't work") |

## Run

| Item | Value |
|---|---|
| Model | `unsloth/Qwen3.5-4B`, 16-bit LoRA (`load_in_4bit=False`) |
| LoRA | r=16, alpha=16, dropout=0, vision + language layers, `random_state=2003` |
| Data | `unsloth/LaTeX_OCR` train split, 68,686 rows |
| Batch | 2 × 4 grad. accum. = 8 |
| Steps | 15 (`max_steps`), warmup 5, lr 2e-4 linear, `adamw_8bit`, seed 3407 |
| Trainable params | 38,756,352 of 4,578,021,888 (0.85 %) |
| Train runtime | 253.8 s (4.23 min) |
| Peak reserved memory | 9.705 GB |

## Training loss per step

| Step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Loss | 0.6997 | 0.8615 | 0.6621 | 0.4513 | 0.3637 | 0.3422 | 0.2323 | 0.1318 |

| Step | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|
| Loss | 0.0887 | 0.0971 | 0.0603 | 0.0812 | 0.0759 | 0.0302 | 0.0420 |

## How to compare

The CLI writes `run_info.json` to the `output_dir`. Compare these fields with the table above:

- `trainable_params` and `total_params`: must be equal.
- The per-step loss in the training log: must follow the table above closely
  (bit-exact equality is not expected on a GPU).
- After a resume, the loss from the resume step onward must also follow the table.
- `peak_reserved_memory_gb` and `wall_clock_s`: must be in the same range on a T4.
