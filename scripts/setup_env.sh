#!/usr/bin/env bash
# Set up the MathScribe environment. The same script runs locally and on Colab.
#
#   bash scripts/setup_env.sh            # local: uv venv + dev tools + pre-commit hooks
#   bash scripts/setup_env.sh --train    # local: also the GPU training stack
#   bash scripts/setup_env.sh --api      # local: also the Claude API baseline (anthropic)
#   bash scripts/setup_env.sh --baselines  # local: also pix2tex and GOT-OCR (torch)
#   bash scripts/setup_env.sh --colab    # Colab: train stack into the system Python
#
# Every mode also installs the LaTeX renderer (node packages in renderer/).
# --train and --baselines cannot go together (pix2tex pins old packages).
# With --colab, set PYTHON to choose the interpreter (default: python3 on PATH).
# The runner notebook sets it to the kernel's own interpreter (sys.executable).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE=local
WITH_TRAIN=0
EXTRAS=()
for arg in "$@"; do
  case "$arg" in
    --colab) MODE=colab; WITH_TRAIN=1 ;;
    --train) WITH_TRAIN=1 ;;
    --api) EXTRAS+=(--extra api) ;;
    --baselines) EXTRAS+=(--extra baselines) ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
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

node_major() {
  node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0
}

setup_renderer() {
  # The LaTeX renderer (renderer/: MathJax + resvg in node) gives the eval its compile
  # rate and render match. The GRPO reward (Phase 3) uses it too.
  if ! command -v node >/dev/null 2>&1 || [[ "$(node_major)" -lt 18 ]]; then
    if [[ "$MODE" == colab ]]; then
      echo ">> Installing Node.js 20 (the renderer needs node >= 18)"
      curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
      apt-get install -y nodejs
    else
      echo "!! The LaTeX renderer needs Node.js >= 18. Install it, then run this script again." >&2
      exit 1
    fi
  fi
  echo ">> LaTeX renderer: npm ci in renderer/ (node $(node --version))"
  (cd renderer && npm ci --no-audit --no-fund --loglevel=error)
}

ensure_uv

if [[ "$MODE" == colab ]]; then
  # Colab already has CUDA torch in the system Python. Install into it, no venv.
  # Name the interpreter explicitly, so the install and the run use the same Python.
  PY="${PYTHON:-$(command -v python3)}"
  echo ">> Installing mathscribe[train] into $PY"
  uv pip install --system --python "$PY" -e ".[train]"
  echo ">> Checking the install"
  "$PY" - <<'EOF'
import importlib.util
import sys

import mathscribe

missing = [m for m in ("unsloth", "trl", "datasets") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"Missing packages after install: {missing}")
print(f"mathscribe {mathscribe.__version__} from {mathscribe.__file__} ({sys.executable})")
EOF
else
  extras=("${EXTRAS[@]}")
  [[ "$WITH_TRAIN" == 1 ]] && extras+=(--extra train)
  echo ">> uv sync ${extras[*]:-}"
  uv sync "${extras[@]}"
  if [[ -d .git ]]; then
    uv run pre-commit install
  fi
fi

setup_renderer
echo ">> Done ($MODE)"
