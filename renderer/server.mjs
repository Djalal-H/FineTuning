// LaTeX math -> PNG render server for the MathScribe eval.
//
// Protocol: one JSON object per line on stdin, one JSON object per line on stdout.
//   request:  {"id": 1, "latex": "\\frac{a}{b}"}
//   response: {"id": 1, "ok": true, "png": "<base64>", "width": 40, "height": 52}
//             {"id": 1, "ok": false, "error": "Undefined control sequence \\foo"}
//
// MathJax parses the TeX. Any parse error or undefined macro is a failure ("does not
// compile"): the noerrors and noundefined packages are off, and formatError throws.
// resvg turns the SVG into a PNG. It uses no system fonts, so the output is the same
// on each machine.
//
// The Python client is src/mathscribe/eval/render.py. Keep RENDERER_VERSION in sync:
// the client puts it in its cache key.

import { createRequire } from "node:module";
import readline from "node:readline";

const require = createRequire(import.meta.url);
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
require("mathjax-full/js/input/tex/AllPackages.js");
const { Resvg } = require("@resvg/resvg-js");

export const RENDERER_VERSION = "mathjax-3.2.2+resvg-2.6.2+snap4";

// TeX packages: roughly what pdflatex with amsmath and amssymb accepts.
const PACKAGES = ["base", "configmacros", "ams", "boldsymbol", "newcommand", "textmacros", "color", "cancel", "upgreek"];

// Old LaTeX commands that MathJax does not define. They change no glyph shape that the
// comparison can see, or they have a close MathJax form.
const MACROS = {
  boldmath: "",
  unboldmath: "",
  protect: "",
  mathstrut: "\\vphantom{(}",
  hfill: "",
  hss: "",
  relax: "",
  noalign: ["", 1],
  makebox: ["{#2}", 2, ""],
  raisebox: ["{#2}", 2],
  lefteqn: ["{#1}", 1],
  b: ["\\underline{#1}", 1],
  cdotp: "\\cdot",
  ldotp: ".",
  textup: ["\\text{#1}", 1],
  textnormal: ["\\text{#1}", 1],
  mbox: ["\\text{#1}", 1],
  // plain TeX and text-mode symbols in the im2latex references
  dag: "\\dagger",
  ddag: "\\ddagger",
  sp: "^",
  sb: "_",
  slash: "/",
  footnotesize: "",
  sl: "\\it",
  ensuremath: ["{#1}", 1],
  d: ["\\underset{.}{#1}", 1],
  l: "\\unicode{x142}",
  L: "\\unicode{x141}",
  o: "\\unicode{xF8}",
  O: "\\unicode{xD8}",
  i: "\\imath",
  AA: "\\unicode{xC5}",
  pounds: "\\unicode{xA3}",
};

// The PNG scale. MathJax units are 1/1000 em, and its TeX font has an x-height of 0.442 em.
const EX_PX = 10; // pixels per ex
const UNIT_PX = EX_PX / 442; // pixels per MathJax unit

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({
  packages: [...PACKAGES, "unicode"],
  macros: MACROS,
  formatError: (_jax, err) => {
    throw err;
  },
});
const svgOut = new SVG({ fontCache: "local" });
const doc = mathjax.document("", { InputJax: tex, OutputJax: svgOut });

// ---------------------------------------------------------------------------
// Pixel snap. MathJax puts glyphs at fractional pixel positions. Then a glyph that moves
// by a space (\,) gets different anti-aliased pixels, and the comparison sees a
// different image. The snap moves each group and each rule (fraction bar) to a whole
// pixel, so the same glyph sequence gives the same pixels after any horizontal space.
// ---------------------------------------------------------------------------
function parseTransform(t) {
  let tx = 0;
  let ty = 0;
  let sx = 1;
  let sy = 1;
  let scaled = false;
  for (const [, name, args] of t.matchAll(/(\w+)\(([^)]*)\)/g)) {
    const v = args.split(/[\s,]+/).filter(Boolean).map(Number);
    if (name === "translate" && !scaled) {
      tx = v[0];
      ty = v[1] ?? 0;
    } else if (name === "scale") {
      sx = v[0];
      sy = v[1] ?? v[0];
      scaled = true;
    } else {
      return null; // rotate, matrix, or translate after scale: leave as is
    }
  }
  return { tx, ty, sx, sy };
}

function snapTree(node, st, grid) {
  for (const child of adaptor.childNodes(node)) {
    const kind = adaptor.kind(child);
    if (kind === "#text" || kind === "#comment" || kind === "defs" || kind === "svg") continue;
    let next = st;
    if (kind === "rect") {
      const x = +adaptor.getAttribute(child, "x") || 0;
      const y = +adaptor.getAttribute(child, "y") || 0;
      const w = +adaptor.getAttribute(child, "width") || 0;
      const h = +adaptor.getAttribute(child, "height") || 0;
      const xs = [grid.x(st.ax + st.sx * x), grid.x(st.ax + st.sx * (x + w))];
      const ys = [grid.y(st.ay + st.sy * y), grid.y(st.ay + st.sy * (y + h))];
      const lx = xs.map((v) => (v - st.ax) / st.sx);
      const ly = ys.map((v) => (v - st.ay) / st.sy);
      adaptor.setAttribute(child, "x", String(Math.min(...lx)));
      adaptor.setAttribute(child, "y", String(Math.min(...ly)));
      adaptor.setAttribute(child, "width", String(Math.abs(lx[1] - lx[0])));
      adaptor.setAttribute(child, "height", String(Math.max(Math.abs(ly[1] - ly[0]), 1 / UNIT_PX / Math.abs(st.sy))));
      continue;
    }
    const t = adaptor.getAttribute(child, "transform");
    if (t) {
      const p = parseTransform(t);
      if (p === null) continue;
      const ax = grid.x(st.ax + st.sx * p.tx);
      const ay = grid.y(st.ay + st.sy * p.ty);
      let out = `translate(${(ax - st.ax) / st.sx},${(ay - st.ay) / st.sy})`;
      if (p.sx !== 1 || p.sy !== 1) out += ` scale(${p.sx},${p.sy})`;
      adaptor.setAttribute(child, "transform", out);
      next = { ax, ay, sx: st.sx * p.sx, sy: st.sy * p.sy };
    }
    snapTree(child, next, grid);
  }
}

// The first <svg> with a viewBox (equation tags, e.g. from \nonumber, add a wrapper).
function findSvg(node) {
  for (const child of adaptor.childNodes(node)) {
    if (adaptor.kind(child) === "svg" && adaptor.getAttribute(child, "viewBox")) return child;
    if (adaptor.kind(child) !== "#text") {
      const found = findSvg(child);
      if (found) return found;
    }
  }
  return null;
}

function toSvg(latex) {
  const container = doc.convert(latex, { display: true, em: 16, ex: 8, containerWidth: 1280 });
  const svg = findSvg(container);
  if (svg === null) throw new Error("MathJax produced no SVG");
  const [minX, minY, w, h] = adaptor.getAttribute(svg, "viewBox").split(/\s+/).map(Number);
  const grid = {
    x: (v) => minX + Math.round((v - minX) * UNIT_PX) / UNIT_PX,
    y: (v) => minY + Math.round((v - minY) * UNIT_PX) / UNIT_PX,
  };
  snapTree(svg, { ax: 0, ay: 0, sx: 1, sy: 1 }, grid);
  // Whole-pixel canvas, and a viewBox of the same size: the scale is exactly UNIT_PX.
  const wPx = Math.ceil(w * UNIT_PX);
  const hPx = Math.ceil(h * UNIT_PX);
  adaptor.setAttribute(svg, "viewBox", `${minX} ${minY} ${wPx / UNIT_PX} ${hPx / UNIT_PX}`);
  adaptor.setAttribute(svg, "width", `${wPx}px`);
  adaptor.setAttribute(svg, "height", `${hPx}px`);
  let out = adaptor.outerHTML(svg).replaceAll("currentColor", "#000000");
  if (!out.includes('xmlns="http://www.w3.org/2000/svg"')) {
    out = out.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  if (!out.includes("xmlns:xlink=")) {
    out = out.replace("<svg", '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
  }
  return out;
}

function render(latex) {
  const svg = toSvg(latex);
  const resvg = new Resvg(svg, { background: "#ffffff", font: { loadSystemFonts: false } });
  const image = resvg.render();
  return { png: Buffer.from(image.asPng()).toString("base64"), width: image.width, height: image.height };
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on("line", (line) => {
  if (!line.trim()) return;
  let id = null;
  let out;
  try {
    const req = JSON.parse(line);
    id = req.id;
    if (req.version) {
      out = { id, ok: true, version: RENDERER_VERSION };
    } else {
      out = { id, ok: true, ...render(String(req.latex)) };
    }
  } catch (err) {
    out = { id, ok: false, error: String(err && err.message ? err.message : err) };
  }
  process.stdout.write(JSON.stringify(out) + "\n");
});
