"""LaTeX -> PNG rendering, and the comparison of two renders.

The renderer is a persistent node process (``renderer/server.mjs``: MathJax + resvg).
Install it with ``bash scripts/setup_env.sh`` (it runs ``npm ci`` in ``renderer/``).

- **Compile**: MathJax parses the LaTeX with no error and no undefined macro.
  MathJax is not pdflatex, so this is an approximation of "it compiles".
- **Render match**: the im2latex image exact match "-ws" (Deng et al., 2017). Binarize
  both renders, crop them to the ink, and remove the empty pixel columns. The renders
  match if the remaining column sequences are equal. Horizontal spacing does not count.
- **Render similarity**: 1 - (column edit distance / length of the longer sequence).
  It is 1.0 for a match, and near 0 for unrelated renders.

Both sides go through the same renderer, so the comparison does not depend on the
font or the style of the dataset images.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mathscribe.eval.metrics import levenshtein

log = logging.getLogger("mathscribe.render")

RENDERER_DIR = Path(__file__).resolve().parents[3] / "renderer"
INK_THRESHOLD = 128  # gray value below this is ink


class RendererUnavailable(RuntimeError):
    """node or the renderer packages are missing. Run ``bash scripts/setup_env.sh``."""


@dataclass(frozen=True)
class RenderResult:
    ok: bool
    png: bytes | None = None
    error: str | None = None


class Renderer:
    """A client for ``renderer/server.mjs``. Use it as a context manager."""

    def __init__(
        self,
        renderer_dir: str | Path = RENDERER_DIR,
        timeout_s: float = 10.0,
        cache_dir: str | Path | None = None,
    ):
        self.dir = Path(renderer_dir)
        self.timeout_s = timeout_s
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory: dict[str, RenderResult] = {}
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._next_id = 0
        self.version = ""
        self._start()

    # ---------------------------------------------------------------- process
    def _start(self) -> None:
        node = shutil.which("node")
        if node is None:
            raise RendererUnavailable("node is not installed. Run: bash scripts/setup_env.sh")
        if not (self.dir / "node_modules" / "mathjax-full").is_dir():
            raise RendererUnavailable(
                f"{self.dir}/node_modules is missing. Run: bash scripts/setup_env.sh"
            )
        self._proc = subprocess.Popen(
            [node, "server.mjs"],
            cwd=self.dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._lines = queue.Queue()
        reader = threading.Thread(
            target=self._read_stdout, args=(self._proc, self._lines), daemon=True
        )
        reader.start()
        reply = self._request({"version": True})
        if reply is None or not reply.get("ok"):
            raise RendererUnavailable(f"the renderer did not start: {reply}")
        self.version = reply["version"]

    @staticmethod
    def _read_stdout(proc: subprocess.Popen[str], lines: queue.Queue[str | None]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # end of stream

    def _request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Send one request, wait for its reply. None on a timeout or a crash."""
        assert self._proc is not None and self._proc.stdin is not None
        self._next_id += 1
        request_id = self._next_id
        try:
            self._proc.stdin.write(json.dumps({"id": request_id, **payload}) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return None
        while True:
            try:
                line = self._lines.get(timeout=self.timeout_s)
            except queue.Empty:
                return None
            if line is None:
                return None
            reply = json.loads(line)
            if reply.get("id") == request_id:
                return reply

    def _restart(self) -> None:
        self.close()
        self._start()

    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    def __enter__(self) -> Renderer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- cache
    def _key(self, latex: str) -> str:
        return hashlib.sha256(f"{self.version}\0{latex}".encode()).hexdigest()

    def _disk_get(self, key: str) -> RenderResult | None:
        if self.cache_dir is None:
            return None
        base = self.cache_dir / key[:2] / key
        if base.with_suffix(".png").is_file():
            return RenderResult(ok=True, png=base.with_suffix(".png").read_bytes())
        if base.with_suffix(".err").is_file():
            return RenderResult(ok=False, error=base.with_suffix(".err").read_text())
        return None

    def _disk_put(self, key: str, result: RenderResult) -> None:
        if self.cache_dir is None or result.error == "timeout":
            return
        base = self.cache_dir / key[:2] / key
        base.parent.mkdir(parents=True, exist_ok=True)
        if result.ok and result.png is not None:
            base.with_suffix(".png").write_bytes(result.png)
        else:
            base.with_suffix(".err").write_text(result.error or "")

    # ---------------------------------------------------------------- render
    def render(self, latex: str) -> RenderResult:
        """Render one formula. A timeout restarts the node process and gives ok=False."""
        if not latex.strip():
            return RenderResult(ok=False, error="empty")
        key = self._key(latex)
        cached = self._memory.get(key) or self._disk_get(key)
        if cached is not None:
            self._memory[key] = cached
            return cached
        reply = self._request({"latex": latex})
        if reply is None:
            log.warning("Render timeout (%.0f s), restarting the renderer", self.timeout_s)
            self._restart()
            result = RenderResult(ok=False, error="timeout")
        elif reply.get("ok"):
            result = RenderResult(ok=True, png=base64.b64decode(reply["png"]))
        else:
            result = RenderResult(ok=False, error=reply.get("error", "unknown error"))
        self._memory[key] = result
        self._disk_put(key, result)
        return result


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def ink_columns(png: bytes) -> list[bytes]:
    """The non-empty pixel columns of the cropped, binarized render (top to bottom)."""
    import numpy as np
    from PIL import Image

    gray = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    ink = gray < INK_THRESHOLD
    rows = np.flatnonzero(ink.any(axis=1))
    if rows.size == 0:
        return []
    ink = ink[rows[0] : rows[-1] + 1]
    return [np.packbits(col).tobytes() for col in ink.T if col.any()]


def compare_renders(png_a: bytes, png_b: bytes) -> tuple[bool, float]:
    """(render match, render similarity) of two renders."""
    a, b = ink_columns(png_a), ink_columns(png_b)
    if a == b:
        return True, 1.0
    longest = max(len(a), len(b))
    return False, 1.0 - levenshtein(a, b) / longest


def render_scores(renderer: Renderer, prediction: str, reference: str) -> dict[str, Any]:
    """Per-row render scores. ``ref_renders`` is False if the reference does not render:
    then the row has no render match and no render similarity."""
    pred = renderer.render(prediction)
    ref = renderer.render(reference)
    out: dict[str, Any] = {
        "compiles": pred.ok,
        "compile_error": pred.error,
        "ref_renders": ref.ok,
        "render_match": None,
        "render_similarity": None,
    }
    if ref.ok:
        if pred.ok:
            assert pred.png is not None and ref.png is not None
            match, similarity = compare_renders(pred.png, ref.png)
        else:
            match, similarity = False, 0.0
        out["render_match"] = match
        out["render_similarity"] = similarity
    return out
