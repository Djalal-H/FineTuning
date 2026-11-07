"""LaTeX tokenization, output extraction, and a canonical form for comparison.

The canonical form (``normalize``) is only for comparison. It is not always valid LaTeX.
It is the output of these rules, in this order:

1. Tokenize: control words (``\\frac``), control symbols (``\\{``, ``\\,``, ``\\\\``), and
   single characters. Whitespace only separates tokens.
2. Drop delimiter sizing (``\\left``, ``\\right``, ``\\middle``, ``\\big`` ... ``\\Bigg`` and
   their ``l``/``r``/``m`` forms). The null delimiter ``.`` after them goes too.
3. Drop spacing (``\\,``, ``\\quad``, ``~`` ...) and style switches (``\\displaystyle`` ...).
4. Replace synonyms with one spelling (``\\le`` -> ``\\leq``, ``\\dfrac`` -> ``\\frac`` ...).
5. If the braces are balanced:
   - drop ``\\hspace``, ``\\vspace`` and ``\\label`` with their argument,
   - unwrap redundant braces: a group with exactly one item (``{ x }`` -> ``x``,
     ``{ { a b } }`` -> ``{ a b }``). Empty groups (``{ }``) stay,
   - in an environment (``\\begin ... \\end``): a cell that is exactly one brace group
     loses the braces (``& { a b } &`` -> ``& a b &``, ``& { } &`` -> ``& &``), and a
     ``\\\\`` immediately before ``\\end`` goes. The column spec of ``array`` stays apart,
   - an ``array``, ``matrix``, ``aligned``, ``gathered`` or ``split`` with exactly one cell
     (no ``&`` and no ``\\\\``) becomes its content: ``\\begin{array}{l} x \\end{array}`` -> ``x``,
   - put a subscript before a superscript (``x ^ b _ a`` -> ``x _ a ^ b``).
   If the braces are not balanced, step 5 does not run. Then a broken output
   cannot become equal to a reference only because of brace rules.
6. Join the tokens with one space.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\\.|\S", re.DOTALL)

_SIZES = ["big", "Big", "bigg", "Bigg"]
SIZING = frozenset(
    ["\\left", "\\right", "\\middle"]
    + [f"\\{s}{suffix}" for s in _SIZES for suffix in ("", "l", "r", "m")]
)
SPACING = frozenset(
    [
        "\\,",
        "\\:",
        "\\;",
        "\\!",
        "\\ ",
        "~",
        "\\quad",
        "\\qquad",
        "\\enspace",
        "\\thinspace",
        "\\negthinspace",
        "\\medspace",
        "\\thickspace",
    ]
)
STYLE = frozenset(
    ["\\displaystyle", "\\textstyle", "\\scriptstyle", "\\scriptscriptstyle", "\\nonumber"]
)
SYNONYMS = {
    "\\le": "\\leq",
    "\\ge": "\\geq",
    "\\ne": "\\neq",
    "\\to": "\\rightarrow",
    "\\gets": "\\leftarrow",
    "\\dfrac": "\\frac",
    "\\tfrac": "\\frac",
    "\\vert": "|",
    "\\lvert": "|",
    "\\rvert": "|",
    "\\Vert": "\\|",
    "\\lVert": "\\|",
    "\\rVert": "\\|",
    "\\lbrace": "\\{",
    "\\rbrace": "\\}",
    "\\lbrack": "[",
    "\\rbrack": "]",
    "\\land": "\\wedge",
    "\\lor": "\\vee",
    "\\lnot": "\\neg",
}
# Commands that are dropped together with their (brace) argument.
DROP_WITH_ARG = frozenset(["\\hspace", "\\vspace", "\\label"])


class Group(list):
    """A ``{ ... }`` group in the parse tree. Items are tokens (str) or groups."""


class UnbalancedBraces(ValueError):
    """The token sequence has a ``}`` without a ``{``, or a ``{`` without a ``}``."""


# --------------------------------------------------------------------------- #
# Tokens and tree
# --------------------------------------------------------------------------- #
def tokenize(latex: str) -> list[str]:
    return TOKEN_RE.findall(latex)


def parse(tokens: list[str]) -> list[str | Group]:
    """Build the brace tree. Raise ``UnbalancedBraces`` if the braces do not match."""
    stack: list[list[str | Group]] = [[]]
    for tok in tokens:
        if tok == "{":
            group = Group()
            stack[-1].append(group)
            stack.append(group)
        elif tok == "}":
            if len(stack) == 1:
                raise UnbalancedBraces("'}' without '{'")
            stack.pop()
        else:
            stack[-1].append(tok)
    if len(stack) != 1:
        raise UnbalancedBraces("'{' without '}'")
    return stack[0]


def flatten(items: list[str | Group]) -> list[str]:
    out: list[str] = []
    for item in items:
        if isinstance(item, Group):
            out += ["{", *flatten(item), "}"]
        else:
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _token_rules(tokens: list[str]) -> list[str]:
    out: list[str] = []
    skip_dot = False
    for tok in tokens:
        if skip_dot:
            skip_dot = False
            if tok == ".":
                continue
        if tok in SIZING:
            skip_dot = True
            continue
        if tok in SPACING or tok in STYLE:
            continue
        out.append(SYNONYMS.get(tok, tok))
    return out


def _tree_rules(items: list[str | Group]) -> list[str | Group]:
    # Drop commands with their argument (an optional "*", then one item).
    kept: list[str | Group] = []
    i = 0
    while i < len(items):
        item = items[i]
        if isinstance(item, str) and item in DROP_WITH_ARG:
            i += 1
            if i < len(items) and items[i] == "*":
                i += 1
            i += 1
            continue
        kept.append(item)
        i += 1

    # Normalize the children, then unwrap groups with exactly one item.
    out: list[str | Group] = []
    for item in kept:
        if isinstance(item, Group):
            inner = _tree_rules(item)
            if len(inner) == 1:
                out.append(inner[0])
            else:
                out.append(Group(inner))
        else:
            out.append(item)

    out = _unwrap_single_cell_envs(_cell_rules(out))

    # "^ B _ A" -> "_ A ^ B".
    i = 0
    while i + 3 < len(out):
        if out[i] == "^" and out[i + 2] == "_":
            out[i : i + 4] = [out[i + 2], out[i + 3], out[i], out[i + 1]]
            i += 4
        else:
            i += 1
    return out


ENV_WITH_COLSPEC = frozenset(["array", "tabular", "subarray"])
ROW_BREAK = "\\\\"


def _env_name(item: str | Group | None) -> str | None:
    if isinstance(item, Group) and all(isinstance(t, str) for t in item):
        return "".join(item)  # type: ignore[arg-type]
    return item if isinstance(item, str) else None


def _cell_rules(items: list[str | Group]) -> list[str | Group]:
    """Cell braces and the last row break in environments (see the module docstring)."""
    out: list[str | Group] = []
    cells: list[list[str | Group]] = []  # the open cell of each open environment

    def emit(item: str | Group) -> None:
        (cells[-1] if cells else out).append(item)

    def close_cell() -> None:
        cell = cells.pop()
        if len(cell) == 1 and isinstance(cell[0], Group):
            cell = list(cell[0])
        (cells[-1] if cells else out).extend(cell)

    i = 0
    while i < len(items):
        item = items[i]
        if item == "\\begin":
            emit(item)
            name = items[i + 1] if i + 1 < len(items) else None
            i += 1
            if name is not None:
                emit(name)
                i += 1
                if _env_name(name) in ENV_WITH_COLSPEC and i < len(items):
                    emit(items[i])  # the column spec
                    i += 1
            cells.append([])
            continue
        if cells and item == ROW_BREAK and i + 1 < len(items) and items[i + 1] == "\\end":
            i += 1  # a row break before \end adds no row
            continue
        if cells and item in ("&", ROW_BREAK):
            close_cell()
            emit(item)
            cells.append([])
        elif cells and item == "\\end":
            close_cell()
            emit(item)
        else:
            emit(item)
        i += 1
    while cells:  # an environment with no \end
        close_cell()
    return out


SINGLE_CELL_ENVS = frozenset(["array", "subarray", "matrix", "aligned", "gathered", "split"])


def _matching_end(items: list[str | Group], begin: int) -> int | None:
    """The index of the ``\\end`` that closes the ``\\begin`` at ``begin``."""
    depth = 0
    for j in range(begin, len(items)):
        if items[j] == "\\begin":
            depth += 1
        elif items[j] == "\\end":
            depth -= 1
            if depth == 0:
                return j
    return None


def _has_top_level_separator(body: list[str | Group]) -> bool:
    depth = 0
    for item in body:
        if item == "\\begin":
            depth += 1
        elif item == "\\end":
            depth -= 1
        elif depth == 0 and item in ("&", ROW_BREAK):
            return True
    return False


def _unwrap_single_cell_envs(items: list[str | Group]) -> list[str | Group]:
    """A one-cell environment is its content (see the module docstring)."""
    out: list[str | Group] = []
    i = 0
    while i < len(items):
        if items[i] == "\\begin" and i + 1 < len(items):
            name = _env_name(items[i + 1])
            end = _matching_end(items, i)
            if (
                name in SINGLE_CELL_ENVS
                and end is not None
                and end + 1 < len(items)
                and _env_name(items[end + 1]) == name
            ):
                start = i + 2 + (1 if name in ENV_WITH_COLSPEC else 0)
                body = items[start:end]
                if not _has_top_level_separator(body):
                    out.extend(_unwrap_single_cell_envs(body))
                    i = end + 2
                    continue
        out.append(items[i])
        i += 1
    return out


def normalize_tokens(latex: str) -> list[str]:
    tokens = _token_rules(tokenize(latex))
    try:
        tree = parse(tokens)
    except UnbalancedBraces:
        return tokens
    return flatten(_tree_rules(tree))


def normalize(latex: str) -> str:
    """Canonical form of a LaTeX string (see the module docstring)."""
    return " ".join(normalize_tokens(latex))


# --------------------------------------------------------------------------- #
# Structure (for difficulty buckets)
# --------------------------------------------------------------------------- #
FRAC_COMMANDS = frozenset(["\\frac", "\\cfrac", "\\binom"])


def frac_depth(items: list[str | Group]) -> int:
    """Maximum nesting of ``\\frac`` (a fraction inside a fraction argument = 2)."""
    depth = 0
    i = 0
    while i < len(items):
        item = items[i]
        if isinstance(item, Group):
            depth = max(depth, frac_depth(item))
        elif item in FRAC_COMMANDS:
            args = items[i + 1 : i + 3]
            inner = max((frac_depth(a) for a in args if isinstance(a, Group)), default=0)
            depth = max(depth, 1 + inner)
            i += 2
        i += 1
    return depth


# --------------------------------------------------------------------------- #
# Model output -> LaTeX
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```[a-zA-Z]*[ \t]*\n?(.*?)```", re.DOTALL)
# A math wrapper at the start of the output, up to its FIRST closing delimiter. Math
# cannot contain its own closing delimiter, so text after it is not part of the formula.
_WRAPPERS = [
    (re.compile(r"^\$\$(.*?)\$\$", re.DOTALL), 1),
    (re.compile(r"^\$(.*?)\$", re.DOTALL), 1),
    (re.compile(r"^\\\[(.*?)\\\]", re.DOTALL), 1),
    (re.compile(r"^\\\((.*?)\\\)", re.DOTALL), 1),
    (re.compile(r"^\\begin\{(equation\*?|displaymath)\}(.*?)\\end\{\1\}", re.DOTALL), 2),
]


def extract_latex(raw: str) -> str:
    """Get the LaTeX from a raw model output. The same rules apply to every model.

    - Text up to the last ``</think>`` goes. An open ``<think>`` with no end gives "".
    - The content of the first Markdown code fence, if there is one.
    - One math wrapper at the start: ``$$..$$``, ``$..$``, ``\\[..\\]``, ``\\(..\\)``,
      ``equation``, up to its first closing delimiter. Text after it goes.
    """
    text = raw
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return ""
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    text = text.strip()
    for pattern, group in _WRAPPERS:
        match = pattern.match(text)
        if match:
            text = match.group(group).strip()
            break
    return text
