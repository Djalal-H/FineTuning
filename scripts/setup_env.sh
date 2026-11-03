#!/usr/bin/env bash
# Set up the MathScribe environment. The same script runs locally and on Colab.
#
#   bash scripts/setup_env.sh            # local: uv venv + dev tools + pre-commit hooks
#   bash scripts/setup_env.sh --train    # local: also the GPU training stack
#   bash scripts/setup_env.sh --colab    # Colab: train stack into the system Python
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE=local
WITH_TRAIN=0
for arg in "$@"; do
  case "$arg" in
    --colab) MODE=colab; WITH_TRAIN=1 ;;
    --train) WITH_TRAIN=1 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv --version
}

setup_renderer() {
  # The LaTeX renderer (KaTeX via node, or pdflatex) is needed by the eval and
  # the GRPO reward. It comes in Phase 1 (see PLAN.md).
  echo ">> LaTeX renderer: not installed yet (Phase 1)"
}

ensure_uv

if [[ "$MODE" == colab ]]; then
  # Colab already has CUDA torch in the system Python. Install into it, no venv.
  echo ">> Installing mathscribe[train] into the system Python"
  uv pip install --system -e ".[train]"
else
  extras=()
  [[ "$WITH_TRAIN" == 1 ]] && extras+=(--extra train)
  echo ">> uv sync ${extras[*]:-}"
  uv sync "${extras[@]}"
  if [[ -d .git ]]; then
    uv run pre-commit install
  fi
fi

setup_renderer
echo ">> Done ($MODE)"
