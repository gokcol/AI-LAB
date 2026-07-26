"""Every page must render without raising, and every simulation must actually draw.

Why this exists: the GUI had no test at all. Two real bugs shipped that this catches —
a page that raised `StreamlitDuplicateElementKey` the moment a dialog opened, and a page
that raised `StreamlitPageNotFoundError` from a link to another module's navigation.
Both were found by a human clicking, which does not scale to 45 pages.

This is a smoke test, deliberately. It proves a page runs and produces the figures and
widgets it is supposed to; it does NOT prove the numbers are right. Numeric claims are
checked in test_core_math.py, test_e01_neuron.py and test_engine_gradcheck.py.

Slow by nature (each page is a full script run, and several train a model), so it is
marked `slow` and skipped by default:

    pytest                      # fast suite, skips these
    pytest -m slow              # render every page
    pytest -m slow -k activations
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUI = ROOT / "gui"
VIEWS = GUI / "views"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GUI))

os.environ.setdefault("MPLBACKEND", "Agg")

# Pages needing an env flag to appear at all; rendered here so they cannot rot unnoticed.
NEEDS_ENV = {"sandbox.py": ("AILAB_ENABLE_SANDBOX", "1")}

PAGES = sorted(p.name for p in VIEWS.glob("*.py") if not p.name.startswith("_"))


def _run(path: pathlib.Path, tmp_path: pathlib.Path, timeout: float = 300.0):
    """Render one page in isolation.

    st.page_link is stubbed out. It resolves against the *active navigation*, which only
    exists when a page is reached through app.py, so a standalone render of dashboard /
    the_chain / ml_overview dies inside Streamlit with KeyError: 'url_pathname' — a
    property of the harness, not of the page. AppTest cannot drive st.navigation, so the
    choice is to stub the call or to leave three pages untested; stubbing keeps the other
    ~200 lines of each page under test. The thing stubbing would hide — a link pointing at
    a page that is not registered — is caught statically instead, in
    test_page_link_targets_exist, which is a stronger check than a render anyway.
    """
    from streamlit.elements.widgets.button import ButtonMixin
    from streamlit.testing.v1 import AppTest

    os.environ["AILAB_FEEDBACK_DIR"] = str(tmp_path)
    # Patch the innermost implementation. Neither `st.page_link` nor
    # `DeltaGenerator.page_link` is enough: pages call it on columns, and gui/i18n.py's
    # localisation shim captured `st.page_link` as a BOUND method at import time, so
    # rebinding the module attribute or the class attribute leaves that copy untouched.
    # Every route reaches ButtonMixin._page_link, so that is the seam.
    real = ButtonMixin._page_link
    ButtonMixin._page_link = lambda self, *a, **k: None
    try:
        at = AppTest.from_file(str(path), default_timeout=timeout)
        return at.run()
    finally:
        ButtonMixin._page_link = real


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, tmp_path, monkeypatch):
    for var, val in [NEEDS_ENV.get(page, ())] if page in NEEDS_ENV else []:
        monkeypatch.setenv(var, val)
    at = _run(VIEWS / page, tmp_path)
    if at.exception:
        messages = "\n".join(str(getattr(e, "value", e)) for e in at.exception)
        pytest.fail(f"{page} raised while rendering:\n{messages}")


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_page_is_not_blank(page, tmp_path, monkeypatch):
    """A page that renders nothing is broken even though it did not raise — this is what
    a silently-swallowed exception or a wrong `if` guard looks like from outside."""
    for var, val in [NEEDS_ENV.get(page, ())] if page in NEEDS_ENV else []:
        monkeypatch.setenv(var, val)
    at = _run(VIEWS / page, tmp_path)
    produced = (len(at.get("markdown")) + len(at.get("caption")) + len(at.get("title"))
                + len(at.get("header")) + len(at.get("subheader")))
    # tests.py is a button and a results pane: genuinely sparse until you click Run.
    floor = 2 if page == "tests.py" else 3
    assert produced >= floor, f"{page} rendered almost nothing ({produced} text elements)"


@pytest.mark.slow
def test_app_entrypoint_renders(tmp_path):
    at = _run(GUI / "app.py", tmp_path)
    assert not at.exception, at.exception
    assert at.get("tab"), "home page rendered no tabs"


def test_every_view_is_registered_or_deliberately_not():
    """A page file nobody can reach is dead weight; catching it here is cheaper than
    noticing months later. Update EXPECTED_UNREGISTERED when that is intentional."""
    EXPECTED_UNREGISTERED: set[str] = set()
    app = (GUI / "app.py").read_text()
    unregistered = {p for p in PAGES if f'views/{p}' not in app}
    assert unregistered == EXPECTED_UNREGISTERED, (
        f"view files not referenced by app.py: {sorted(unregistered - EXPECTED_UNREGISTERED)}")


def _runtime_strings(path: pathlib.Path) -> list[str]:
    """Every string as it will exist AT RUNTIME, not as it appears in the source.

    Scanning raw source is wrong and produces pure noise: Python's implicit concatenation
    and line-continuations put quotes and newlines in the middle of what is really one
    flat string, so a perfectly good `$x$` split across two source lines looks broken and
    a valid inline SVG looks malformed. `ast` folds concatenation for us; f-strings are
    reconstructed with a placeholder per interpolation so the literal text around them is
    still checked. Docstrings are skipped — they are prose for developers, never rendered,
    and they legitimately contain things like nginx's `$proxy_add_x_forwarded_for`.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    out: list[str] = []

    def visit(node):
        if isinstance(node, ast.JoinedStr):          # f-string: rebuild, do not recurse
            out.append("".join(
                v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "0"
                for v in node.values))
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                out.append(node.value)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return out


def _latex_call_args(path: pathlib.Path) -> list[str]:
    """The LaTeX passed to st.latex(...).

    These carry no `$` delimiters, so a scan that only looks for `$...$` spans misses
    every one of them — and there are ~80. A planted `\\left( \\frac{a}{b` sailed
    through the suite until this was added.
    """
    import ast

    def flatten(node) -> str | None:
        """Best-effort runtime text of a string expression.

        Most st.latex calls in this lab are concatenations of raw strings with computed
        values spliced in, so handling only bare literals covered 29 of 83. Interpolated
        values become "0": their content is a number, and what needs checking is the
        literal LaTeX around them.
        """
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else "0"
        if isinstance(node, ast.JoinedStr):
            parts = [flatten(v) for v in node.values]
            return "".join(p if p is not None else "0" for p in parts)
        if isinstance(node, ast.FormattedValue):
            return "0"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # An unresolvable operand becomes "0" rather than voiding the whole
            # expression: a spliced-in value contributes no braces, and the literal
            # LaTeX either side of it is exactly what needs checking.
            return (flatten(node.left) or "0") + (flatten(node.right) or "0")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return flatten(node.left)          # "...%s..." % (...)
        if isinstance(node, ast.Call):
            return "0"                         # str(x), f(x): a value, not markup
        return None                            # a variable: cannot resolve statically

    out: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "latex") or not node.args:
            continue
        text = flatten(node.args[0])
        if text is not None:
            out.append(text)
    return out


def _sources():
    return sorted(VIEWS.glob("*.py")) + sorted(GUI.glob("*.py"))


def test_no_multiline_inline_math():
    """`$...$` spanning a newline renders as raw red text in Streamlit and cascades into
    the rest of the page. This has bitten the lab twice; it is cheap to forbid."""
    import re

    offenders = []
    for f in _sources():
        for text in _runtime_strings(f):
            body = re.sub(r"\$\$[\s\S]+?\$\$", "", text)   # display math may span lines
            i = 0
            while (a := body.find("$", i)) >= 0:
                if a and body[a - 1] == "\\":
                    i = a + 1
                    continue
                b = body.find("$", a + 1)
                if b < 0:
                    break
                if "\n" in body[a:b]:
                    offenders.append(f"{f.name}: {body[a:b][:70]!r}")
                i = b + 1
    assert not offenders, "multi-line inline math:\n" + "\n".join(offenders)


def test_latex_delimiters_balanced():
    """Unbalanced braces or \\left without \\right make KaTeX drop the whole span."""
    import re

    bad, checked = [], 0
    for f in _sources():
        spans_from_latex_calls = _latex_call_args(f)      # st.latex(...) has no $ markers
        for text in _runtime_strings(f):
            spans = [m.group(1) for m in re.finditer(r"\$\$([\s\S]+?)\$\$", text)]
            body = re.sub(r"\$\$[\s\S]+?\$\$", "", text)
            spans += [m.group(1) for m in re.finditer(r"(?<![\\\\$])\$([^$\n]+?)\$(?!\$)", body)]
            for tex in spans:
                if tex.count("{") != tex.count("}"):
                    bad.append(f"{f.name}: unbalanced braces in {tex[:60]!r}")
                if len(re.findall(r"\\left(?![a-zA-Z])", tex)) != \
                   len(re.findall(r"\\right(?![a-zA-Z])", tex)):
                    bad.append(f"{f.name}: \\left/\\right mismatch in {tex[:60]!r}")
                checked += 1
        for tex in spans_from_latex_calls:
            checked += 1
            if tex.count("{") != tex.count("}"):
                bad.append(f"{f.name}: unbalanced braces in st.latex {tex[:60]!r}")
            if len(re.findall(r"\\left(?![a-zA-Z])", tex)) != \
               len(re.findall(r"\\right(?![a-zA-Z])", tex)):
                bad.append(f"{f.name}: \\left/\\right mismatch in st.latex {tex[:60]!r}")
    assert checked > 500, f"only {checked} math spans found — the extractor is broken"
    assert not bad, "\n".join(bad)


def test_svg_diagrams_are_well_formed():
    """An inline SVG with a stray tag silently renders as nothing."""
    import re
    import xml.etree.ElementTree as ET

    bad, seen = [], 0
    for f in _sources():
        for text in _runtime_strings(f):
            for m in re.finditer(r"<svg[\s\S]*?</svg>", text):
                seen += 1
                try:
                    ET.fromstring(m.group(0))
                except ET.ParseError as e:
                    bad.append(f"{f.name}: {e}")
    assert seen > 20, f"only found {seen} SVGs — the extractor is broken, not the diagrams"
    assert not bad, "\n".join(bad)


def test_page_link_targets_exist_and_are_registered():
    """Every st.page_link must point at a real view that app.py registers.

    This is the static form of the bug that once shipped: a link from Home to an ANN-only
    page raised StreamlitPageNotFoundError whenever the ML or Math track was selected,
    because st.page_link can only resolve pages in the ACTIVE navigation. Checking the
    target exists and is registered catches the typo class at zero cost.
    """
    import ast
    import re

    app = (GUI / "app.py").read_text(encoding="utf-8")
    missing = []
    for f in _sources():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "page_link" and node.args):
                continue
            arg = node.args[0]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue                       # computed target: covered by the render test
            target = arg.value
            if not re.match(r"^views/[\w]+\.py$", target):
                continue
            if not (VIEWS / pathlib.Path(target).name).exists():
                missing.append(f"{f.name}: links to {target}, which does not exist")
            elif target not in app:
                missing.append(f"{f.name}: links to {target}, which app.py never registers")
    assert not missing, "\n".join(missing)


def test_sidebar_controls_are_never_hidden_by_theme_css():
    """The sidebar's expand button lives inside [data-testid="stToolbar"].

    Hiding that container with display:none made collapsing the sidebar a ONE-WAY DOOR:
    the collapse button travels off-screen with the sidebar, the expand button only mounts
    while collapsed, and a child of a display:none parent cannot be un-hidden. Hide the
    branded ITEMS (stToolbarActions, stMainMenu, stAppDeployButton, stStatusWidget) and
    never the containers.
    """
    import re

    css = (GUI / "ui.py").read_text(encoding="utf-8")
    # Strip CSS comments first: the comment explaining THIS rule names the very selectors
    # being checked for, which made the guard fail on correct CSS.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    forbidden = ("stToolbar\"", "stAppToolbar", "stHeader\"", "stSidebar\"",
                 "stSidebarCollapseButton", "stExpandSidebarButton", "stSidebarContent")
    # find each display:none rule and read the selector list preceding it
    for m in re.finditer(r"([^{}]+)\{[^{}]*display:\s*none[^{}]*\}", css):
        selector = m.group(1)
        for bad in forbidden:
            if bad in selector:
                raise AssertionError(
                    f"theme CSS hides {bad} in a display:none rule — this breaks the sidebar "
                    f"toggle.\nselector was: {selector.strip()[:200]}")
    assert "stToolbarActions" in css, "expected the branded toolbar items to still be hidden"
