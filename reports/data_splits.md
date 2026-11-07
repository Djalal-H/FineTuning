# Data splits

All sets are frozen manifests in [`data/testsets/`](../data/testsets/). A manifest records
the dataset, the commit sha of the dataset repo, the split, the seed, and the selected rows
in a seeded random order. Each row has a hash of its image bytes and its text. The eval
checks every row against its hash (`python -m mathscribe.eval.testsets verify <manifest>`).

| Set | Manifest | Source | Rows | Seed | Rows sha256 | Use |
|---|---|---|---|---|---|---|
| `latex_ocr_test` | [`latex_ocr_test.json`](../data/testsets/latex_ocr_test.json) | `unsloth/LaTeX_OCR` `test` @ `4da395c8` | 1000 of 7,632 | 2026 | `ca73bfe6fc809f65` | printed test set (leaderboard) |
| `latex_ocr_val` | [`latex_ocr_val.json`](../data/testsets/latex_ocr_val.json) | `unsloth/LaTeX_OCR` `train` @ `4da395c8` | 1000 of 68,686 | 2026 | `669beb01e2e35545` | validation (Phase 2: eval during training, checkpoint selection) |
| `crohme2014` | [`crohme2014.json`](../data/testsets/crohme2014.json) | `Neeze/CROHME-full` `2014` @ `6d03ee8f` | all 986 | 2026 | `a8d04ecfa9a793f1` | handwritten test set (leaderboard) |

## Training rows

The training set is the `unsloth/LaTeX_OCR` `train` split at the same revision, minus the
`latex_ocr_val` rows: 67,686 rows. A train config gets this with:

```yaml
data:
  revision: 4da395c8a0253f4f30983cf08f2480f9bafbd561
  exclude: [data/testsets/latex_ocr_val.json]
```

`configs/train/sft_smoke.yaml` keeps `exclude: []`, because it reproduces the notebook run
(the full `train` split).

## Overlap with the training rows

The number of formulas whose canonical form ([`normalize.py`](../src/mathscribe/eval/normalize.py))
also occurs in the training rows. A model can know these formulas from training.

| Set | Compared with | Overlap | Command |
|---|---|---|---|
| `latex_ocr_test` | `unsloth/LaTeX_OCR` `train` minus val (67,686 rows) | 9 of 1000 (0.9%) | `python -m mathscribe.eval.testsets overlap data/testsets/latex_ocr_test.json --split train --exclude data/testsets/latex_ocr_val.json` |
| `latex_ocr_val` | `unsloth/LaTeX_OCR` `train` minus val (67,686 rows) | 20 of 1000 (2.0%) | `python -m mathscribe.eval.testsets overlap data/testsets/latex_ocr_val.json --split train --exclude data/testsets/latex_ocr_val.json` |
| `crohme2014` | `Neeze/CROHME-full` `train` (8,834 rows) | 30 of 986 (3.0%) | `python -m mathscribe.eval.testsets overlap data/testsets/crohme2014.json --split train` |

The CROHME overlap comes mostly from short formulas (for example `2 p`). The project does
not train on CROHME yet.

## Renderer coverage of the references

The MathJax renderer ([`render.py`](../src/mathscribe/eval/render.py)) renders 7,597 of the
7,632 `unsloth/LaTeX_OCR` `test` references (99.5%) and all 986 CROHME 2014 references. Most
failures are references that also fail in LaTeX after the im2latex tokenization (for example
`\hspace { - 0 . 1 i n }`). The eval leaves these rows out of the render match denominator and
reports the share as `ref_render_rate`.

## Notes

- The CROHME images are white ink on a black background (1366 × 768). Every model gets the
  images as they are.
- CROHME is under the CROHME competition terms (research use). The `Neeze/CROHME-full` card
  states no license.
