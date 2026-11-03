import pytest

from mathscribe.eval.normalize import (
    Group,
    UnbalancedBraces,
    extract_latex,
    frac_depth,
    normalize,
    parse,
    tokenize,
)


def test_tokenize_splits_commands_symbols_and_characters():
    assert tokenize(r"\frac{ab}{c}\,x\\y\{") == [
        "\\frac", "{", "a", "b", "}", "{", "c", "}", "\\,", "x", "\\\\", "y", "\\{",
    ]  # fmt: skip


def test_tokenize_ignores_whitespace_but_keeps_control_space():
    assert tokenize("a  b\n c") == ["a", "b", "c"]
    assert tokenize(r"a\ b") == ["a", "\\ ", "b"]


def test_parse_builds_groups():
    assert parse(tokenize("a { b { c } }")) == ["a", Group(["b", Group(["c"])])]


@pytest.mark.parametrize("latex", ["{ a", "a }", "} {"])
def test_parse_rejects_unbalanced_braces(latex):
    with pytest.raises(UnbalancedBraces):
        parse(tokenize(latex))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # whitespace
        (r"\frac{a}{b}", r"\frac { a } { b }"),
        # \left / \right and the null delimiter
        (r"\left( x \right)", "( x )"),
        (r"\left. x \right|", "x |"),
        (r"\Bigl[ x \Bigr]", "[ x ]"),
        # redundant braces
        ("x ^ { 2 }", "x ^ 2"),
        ("{ { a b } }", "{ a b }"),
        (r"{ \alpha }", r"\alpha"),
        # spacing and style
        (r"a \, b \quad c ~ d", "a b c d"),
        (r"\displaystyle \sum", r"\sum"),
        (r"a \hspace { 2 pt } b", "a b"),
        (r"a \hspace* { 1 em } b", "a b"),
        (r"x \label { eq : 1 }", "x"),
        # synonyms
        (r"a \le b \ne c", r"a \leq b \neq c"),
        (r"\dfrac 1 2", r"\frac 1 2"),
        (r"\vert x \vert", "| x |"),
        # subscript before superscript
        ("x ^ { b } _ { a }", "x _ a ^ b"),
    ],
)
def test_equivalent_forms_have_one_canonical_form(a, b):
    assert normalize(a) == normalize(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("x ^ { a b }", "x ^ a b"),  # multi-token group is not redundant
        (r"{ \cal R } x", r"\cal R x"),  # braces limit the font switch
        ("{ } ^ { 1 4 } C", "^ { 1 4 } C"),  # empty groups stay
        ("a b", "a c"),
    ],
)
def test_different_forms_stay_different(a, b):
    assert normalize(a) != normalize(b)


def test_normalize_is_idempotent():
    latex = r"\left( { \frac { { x } ^ { 2 } } { y } } \right) \, \le \dfrac{1}{2}"
    once = normalize(latex)
    assert normalize(once) == once


def test_unbalanced_input_skips_the_brace_rules():
    # "{ a" must not become "a", or a broken output could match the reference "a".
    assert normalize("{ a") == "{ a"
    assert normalize(r"{ \left( a") == "{ ( a"


@pytest.mark.parametrize(
    ("latex", "depth"),
    [
        ("x", 0),
        (r"\frac a b", 1),
        (r"\frac { \frac a b } c", 2),
        (r"\frac a { 1 + \frac { \frac 1 2 } c }", 3),
        (r"\frac a b + \frac c d", 1),
        (r"{ x + \frac { \frac a b } c }", 2),
    ],
)
def test_frac_depth(latex, depth):
    assert frac_depth(parse(tokenize(latex))) == depth


@pytest.mark.parametrize(
    ("raw", "latex"),
    [
        (r"\frac { a } { b }", r"\frac { a } { b }"),
        ("  x + 1 \n", "x + 1"),
        ("$$x^2$$", "x^2"),
        ("$x^2$", "x^2"),
        (r"\[ x^2 \]", "x^2"),
        (r"\( x^2 \)", "x^2"),
        (r"\begin{equation} x \end{equation}", "x"),
        (r"\begin{equation*} x \end{equation*}", "x"),
        ("Here it is:\n```latex\n\\alpha + 1\n```\nDone.", r"\alpha + 1"),
        ("```\n$$ y $$\n```", "y"),
        ("<think>\nlet me see\n</think>\n\n$z$", "z"),
        ("<think>\n\n</think>\n\nz", "z"),
        ("<think>\nstill thinking", ""),
        # matrix environments are content, not wrappers
        (r"\begin{array}{c} a \end{array}", r"\begin{array}{c} a \end{array}"),
    ],
)
def test_extract_latex(raw, latex):
    assert extract_latex(raw) == latex
