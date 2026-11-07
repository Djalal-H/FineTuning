"""Renderer tests. They need node and renderer/node_modules (bash scripts/setup_env.sh)."""

import pytest

from mathscribe.eval.render import (
    Renderer,
    RendererUnavailable,
    compare_renders,
    ink_columns,
    render_scores,
)

pytest.importorskip("numpy")
pytest.importorskip("PIL")


@pytest.fixture(scope="module")
def renderer(tmp_path_factory):
    try:
        r = Renderer(cache_dir=tmp_path_factory.mktemp("render_cache"))
    except RendererUnavailable as e:
        pytest.skip(str(e))
    yield r
    r.close()


def _match(renderer, a, b):
    return compare_renders(renderer.render(a).png, renderer.render(b).png)


def test_version_is_reported(renderer):
    assert renderer.version.startswith("mathjax-")


def test_render_gives_a_png(renderer):
    result = renderer.render(r"\frac{a}{b}")
    assert result.ok
    assert result.png.startswith(b"\x89PNG")


@pytest.mark.parametrize(
    ("latex", "error"),
    [
        (r"\foo x", "Undefined control sequence"),
        (r"\frac{a", "Missing close brace"),
        ("  ", "empty"),
    ],
)
def test_broken_latex_does_not_compile(renderer, latex, error):
    result = renderer.render(latex)
    assert not result.ok
    assert error in result.error


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (r"\frac { a } { b } + x ^ { 2 }", r"\frac{a}{b}+x^2"),
        (r"\frac{a}{b}+x^2", r"\frac{a}{b}\,+x^2"),  # spaces do not count (pixel snap)
        (r"a+b+c+d", r"a + b \; + c \quad + d"),
        (r"x_{a}^{b}", r"x^{b}_{a}"),
        (
            r"\begin{array}{cc} a & b \\ c & d \end{array}",
            r"\begin{array}{cc} {a} & {b} \\ c & d \\ \end{array}",
        ),
    ],
)
def test_equivalent_latex_renders_match(renderer, a, b):
    assert _match(renderer, a, b) == (True, 1.0)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (r"\frac{a}{b}+x^2", r"\frac{a}{c}+x^2"),
        (r"x + y .", r"x + y"),  # a missing period counts
        (r"\sin x", r"sin x"),
        (r"x", r"x^\prime"),
    ],
)
def test_different_latex_renders_do_not_match(renderer, a, b):
    match, similarity = _match(renderer, a, b)
    assert not match
    assert 0.0 <= similarity < 1.0


def test_similarity_is_higher_for_a_closer_formula(renderer):
    _, near = _match(renderer, r"a + b + c + d", r"a + b + c + e")
    _, far = _match(renderer, r"a + b + c + d", r"\int_0^1 f(t) dt")
    assert near > far


def test_old_macros_from_the_references_render(renderer):
    for latex in [r"\dag", r"x \sp 2", r"\boldmath x", r"\mbox{if} x", r"a \nonumber"]:
        assert renderer.render(latex).ok, latex


def test_cache_returns_the_same_result(renderer):
    first = renderer.render(r"\sqrt{2}")
    assert renderer.render(r"\sqrt{2}") is first


def test_disk_cache_survives_a_new_renderer(tmp_path):
    try:
        with Renderer(cache_dir=tmp_path) as r1:
            png = r1.render(r"\gamma").png
    except RendererUnavailable as e:
        pytest.skip(str(e))
    assert any(tmp_path.rglob("*.png"))
    with Renderer(cache_dir=tmp_path) as r2:
        r2._request = lambda payload: pytest.fail("must use the cache")
        assert r2.render(r"\gamma").png == png


def test_timeout_restarts_the_renderer(renderer, monkeypatch):
    monkeypatch.setattr(renderer, "_request", lambda payload: None)
    restarts = []
    monkeypatch.setattr(renderer, "_restart", lambda: restarts.append(1))
    result = renderer.render(r"\text{never cached " + "x" * 10 + "}")
    assert (result.ok, result.error) == (False, "timeout")
    assert restarts == [1]


def test_render_scores(renderer):
    s = render_scores(renderer, r"\frac{a}{b}", r"\frac { a } { b }")
    assert s["compiles"] and s["ref_renders"] and s["render_match"]
    s = render_scores(renderer, r"\frac{a", r"\frac { a } { b }")
    assert not s["compiles"]
    assert (s["render_match"], s["render_similarity"]) == (False, 0.0)
    # A reference that does not render gives no render scores for the row.
    s = render_scores(renderer, "x", r"\hspace { 1 p t } x")
    assert not s["ref_renders"]
    assert s["render_match"] is None and s["render_similarity"] is None


def test_ink_columns_of_a_blank_image_are_empty():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (10, 10), 255).save(buf, format="PNG")
    assert ink_columns(buf.getvalue()) == []
