"""Home — the landing page (default page of the app).

Explains what the lab is, how to use it, and the terms of use; shows live stats,
a version history, and the protected feedback form (gui/feedback.py).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import os

import streamlit as st

import feedback

GUI = pathlib.Path(__file__).resolve().parents[1]
ROOT = GUI.parent

VERSION = "1.0"
VERSION_HISTORY = [
    ("26.07.2026", "1.0", "Initial release — the ANN track (single neuron → a small GPT), "
                          "classical ML (M0–M8), Math foundations (X1–X6), worked numeric "
                          "examples, an activation-functions deep dive, live playgrounds "
                          "throughout, and this feedback form."),
]

# --------------------------------------------------------------------------- #
# Graphics
# --------------------------------------------------------------------------- #
_HERO_SVG = '''<div style="text-align:center;margin:0.2rem 0 0.8rem"><svg viewBox="0 0 760 240" style="width:100%;max-width:860px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AI Lab banner: the journey from a single neuron, through layers and attention, to a small GPT."><defs><linearGradient id="hmbg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#EAF1FB"/><stop offset="0.55" stop-color="#E6F4EC"/><stop offset="1" stop-color="#FDF3E3"/></linearGradient><marker id="hmar" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#7A8A99"/></marker></defs><rect x="1" y="1" width="758" height="238" rx="18" fill="url(#hmbg)" stroke="#E2E2DA"/><g fill="#C9D8EA" opacity="0.8"><circle cx="70" cy="40" r="2.5"/><circle cx="700" cy="200" r="2.5"/><circle cx="640" cy="36" r="2"/><circle cx="120" cy="205" r="2"/><circle cx="380" cy="24" r="2"/></g><text x="380" y="64" text-anchor="middle" font-family="sans-serif" font-size="34" font-weight="800" fill="#172033">🧠 AI Lab</text><text x="380" y="90" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#5A6B61">neural networks · machine learning · the mathematics behind them</text><text x="380" y="109" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#8A9AA8">study notes you can run — a single neuron all the way to a small GPT</text><g font-family="sans-serif"><circle cx="120" cy="172" r="26" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="2.2"/><text x="120" y="178" text-anchor="middle" font-size="15" fill="#0C447C">Σ</text><g fill="#1D9E75"><circle cx="268" cy="147" r="7"/><circle cx="268" cy="165" r="7"/><circle cx="268" cy="183" r="7"/><circle cx="292" cy="156" r="7"/><circle cx="292" cy="174" r="7"/></g><rect x="404" y="143" width="104" height="44" rx="9" fill="#FBEAF0" stroke="#C0507A" stroke-width="2"/><text x="456" y="163" text-anchor="middle" font-size="11.5" fill="#8A2351">attention</text><text x="456" y="178" text-anchor="middle" font-size="9" fill="#B06A87">Q·Kᵀ softmax</text><rect x="600" y="140" width="110" height="50" rx="10" fill="#D8EFD7" stroke="#3DA147" stroke-width="2.2"/><text x="655" y="161" text-anchor="middle" font-size="13" font-weight="700" fill="#1D5E2A">tiny GPT</text><text x="655" y="177" text-anchor="middle" font-size="9" fill="#4E7D58">you can train it</text><g stroke="#7A8A99" stroke-width="2.2" fill="none"><line x1="150" y1="165" x2="255" y2="165" marker-end="url(#hmar)"/><line x1="303" y1="165" x2="400" y2="165" marker-end="url(#hmar)"/><line x1="510" y1="165" x2="596" y2="165" marker-end="url(#hmar)"/></g><g font-size="9.5" fill="#8A9AA8" text-anchor="middle"><text x="120" y="207">one neuron</text><text x="280" y="207">layers</text><text x="456" y="207">transformers</text><text x="655" y="207">language models</text></g></g><text x="380" y="228" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#8A9AA8">three tracks — ANN · ML · Math — every stop interactive: theory, playground, self-check, worked solutions</text></svg></div>'''

_LOOP_SVG = '''<div style="text-align:center;margin:0.3rem 0"><svg viewBox="0 0 560 210" style="width:100%;max-width:560px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The learning loop used on every page: predict, then play with the interactive demo, then check yourself with the quiz, then rebuild the idea from memory."><defs><marker id="lpar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#7A8A99"/></marker></defs><rect x="1" y="1" width="558" height="208" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g font-family="sans-serif" font-size="12.5" text-anchor="middle" font-weight="600"><rect x="40" y="30" width="120" height="44" rx="9" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.8"/><text x="100" y="50" fill="#0C447C">🔮 Predict</text><text x="100" y="65" font-size="9" font-weight="400" fill="#6B8AB0">commit to a guess</text><rect x="400" y="30" width="120" height="44" rx="9" fill="#DCEFE2" stroke="#1D9E75" stroke-width="1.8"/><text x="460" y="50" fill="#0E5E45">🎛 Play</text><text x="460" y="65" font-size="9" font-weight="400" fill="#5E8E76">move every slider</text><rect x="400" y="136" width="120" height="44" rx="9" fill="#FBEAF0" stroke="#C0507A" stroke-width="1.8"/><text x="460" y="156" fill="#8A2351">❓ Check</text><text x="460" y="171" font-size="9" font-weight="400" fill="#B06A87">quiz + solutions</text><rect x="40" y="136" width="120" height="44" rx="9" fill="#FBEAD6" stroke="#9A6A2A" stroke-width="1.8"/><text x="100" y="156" fill="#5A3E14">🛠 Rebuild</text><text x="100" y="171" font-size="9" font-weight="400" fill="#9A7B4F">from memory, in code</text></g><g stroke="#7A8A99" stroke-width="2" fill="none"><line x1="164" y1="52" x2="396" y2="52" marker-end="url(#lpar)"/><line x1="460" y1="78" x2="460" y2="132" marker-end="url(#lpar)"/><line x1="396" y1="158" x2="164" y2="158" marker-end="url(#lpar)"/><line x1="100" y1="132" x2="100" y2="78" marker-end="url(#lpar)"/></g><text x="280" y="112" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95">the learning loop — on every page</text></svg></div>'''


_TIMELINE_SVG = '''<div style="text-align:center;margin:0.2rem 0 0.8rem"><svg viewBox="0 0 760 150" style="width:100%;max-width:760px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Career timeline: 1987 first code on IBM mainframes in Assembler; 1993 to 1994 a PhD-level course on the mathematics of neural networks; 1999 a PhD in Aeronautics and computational engineering; 1999 to 2006 leading university IT services while teaching; 2016 founding an information-security consultancy and audit practice; 2025 using agentic AI professionally."><rect x="1" y="1" width="758" height="148" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="70" y1="70" x2="700" y2="70" stroke="#C9D3DC" stroke-width="3"/><g><circle cx="100" cy="70" r="7" fill="#5B8FC2"/><circle cx="205" cy="70" r="9" fill="#1D9E75" stroke="#0E5E45" stroke-width="2"/><circle cx="280" cy="70" r="7" fill="#5B8FC2"/><circle cx="385" cy="70" r="7" fill="#5B8FC2"/><circle cx="535" cy="70" r="7" fill="#9A6A2A"/><circle cx="670" cy="70" r="9" fill="#C0507A" stroke="#8A2351" stroke-width="2"/></g><g font-family="sans-serif" font-size="10.5" font-weight="700" text-anchor="middle" fill="#33312E"><text x="100" y="40">1987</text><text x="205" y="34" fill="#0E5E45">1993–94</text><text x="280" y="40">1999</text><text x="385" y="118">1999–2006</text><text x="535" y="118">2016</text><text x="670" y="118" fill="#8A2351">2025</text></g><g font-family="sans-serif" font-size="8.8" text-anchor="middle" fill="#6B6A66"><text x="100" y="52">first code</text><text x="205" y="46" fill="#1D7A5E">maths of neural nets</text><text x="280" y="52">PhD, Aeronautics</text><text x="385" y="130">university IT + teaching</text><text x="535" y="130">infosec consultancy</text><text x="670" y="130" fill="#B0567E">agentic AI</text></g><g stroke="#C9D3DC" stroke-width="1"><line x1="385" y1="79" x2="385" y2="106"/><line x1="535" y1="79" x2="535" y2="106"/><line x1="670" y1="81" x2="670" y2="106"/><line x1="100" y1="61" x2="100" y2="56"/><line x1="205" y1="59" x2="205" y2="50"/><line x1="280" y1="61" x2="280" y2="56"/></g><text x="380" y="14" text-anchor="middle" font-family="sans-serif" font-size="9.5" fill="#9C9B95">from punched mainframe decks to LLM agents</text></svg></div>'''


@st.cache_data(show_spinner=False)
def _stats():
    views = len(list((GUI / "views").glob("*.py")))
    exps = len(list(ROOT.glob("experiments/tier*/e*/run.py")))
    files = list(GUI.glob("*.py")) + list((GUI / "views").glob("*.py"))
    svgs = quiz = 0
    for f in files:
        try:
            src = f.read_text()
        except OSError:
            continue
        svgs += src.count("<svg")
        quiz += src.count("Question(")
    return views, exps, svgs, quiz


def _switch_module(mod: str):
    st.session_state["module"] = mod


def _goto(mod: str, page: str):
    """Jump to a page that lives in another module's navigation.

    st.page_link can only target pages registered in the ACTIVE nav, and Home is shown
    in all three tracks — so a direct link to an ANN page raises while ML/Math is
    selected. Instead we switch the module and let app.py perform the navigation on the
    next run, once the nav has been rebuilt."""
    st.session_state["module"] = mod
    st.session_state["_goto_page"] = page


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.markdown(_HERO_SVG, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Navigation sits immediately under the banner, so the first thing a visitor meets is
# a choice rather than a wall of prose. Everything that used to run down the page --
# the welcome, the disclaimer and the counters -- now opens inside "Welcome", which is
# the tab Streamlit selects by default: nothing became harder to find, and the tabs
# above the fold say at a glance that there is more here than one long page.
# Short labels keep the row usable on a phone, where the strip scrolls horizontally.
# --------------------------------------------------------------------------- #
(tab_welcome, tab_start, tab_method, tab_run,
 tab_terms, tab_fb, tab_about) = st.tabs(
    ["👋 Welcome", "🚀 Start", "🎓 Method", "🖥 Run it",
     "📜 Terms", "💬 Feedback", "👤 About"]
)

with tab_welcome:
    st.markdown(
        "**Welcome.** These are **my own study notes** — personal notes I curated for myself while "
        "relearning how modern AI actually works — **neural networks** from a single neuron to a "
        "small GPT, **classical machine learning**, and the **mathematics** underneath both. "
        "I learn by "
        "**doing**, so every note here comes with an interactive playground, worked numeric "
        "examples, self-check questions with answers, and code that runs. I am publishing them "
        "as-is, exactly as I use them, in case they are useful to someone else on the same road."
    )

    st.warning(
        "**Please read this first — what this is, and what it is not.**\n\n"
        "This is a **personal study notebook**, curated by **Orhan Gökçöl** while learning, and "
        "written with heavy use of **Claude** (Anthropic) as an AI assistant. It is **not** a "
        "textbook, not a course, not peer-reviewed, and not authoritative teaching material. It "
        "reflects one learner's understanding at one point in time — it may be incomplete, "
        "simplified, or in places simply **wrong**. Please treat every claim as a starting point "
        "to verify against the primary sources listed on each page, never as a citation.\n\n"
        "**There is no guarantee — explicit or implied — that anything here is correct, "
        "complete, or error-free. The author accepts no liability whatsoever for any "
        "misinformation, error or omission, nor for any loss or damage arising from the use "
        "of this material. You use it entirely at your own risk.**",
        icon=":material/menu_book:")

    st.caption("Source, issues and corrections: "
               "[github.com/gokcol/AI-LAB](https://github.com/gokcol/AI-LAB) — corrections are "
               "genuinely welcome; that is how study notes get better.")

    views, exps, svgs, quiz = _stats()
    _CARDS = [("s-blue", "🧭", f"{views}", "interactive pages"),
              ("s-green", "🔬", f"{exps}", "code experiments"),
              ("s-amber", "📈", f"{svgs}+", "diagrams"),
              ("s-plum", "❓", f"{quiz}+", "self-check questions")]
    st.markdown(
        '<div class="ailab-stats">'
        + "".join(f'<div class="ailab-stat {c}"><div class="ic">{i}</div>'
                  f'<div class="num">{n}</div><div class="lab">{l}</div></div>'
                  for c, i, n, l in _CARDS)
        + "</div>",
        unsafe_allow_html=True,
    )


with tab_start:
    # --- three tracks ----------------------------------------------------------- #
    st.markdown("### 🗺 Three tracks — pick your path")
    t = st.columns(3)
    with t[0]:
        with st.container(border=True):
            st.markdown("#### 🧠 ANN\n**The spine.** A single neuron → logic gates → training "
                        "→ CNN/RNN → attention → a **tiny GPT you can train**. Five levels, "
                        "basics to frontier.")
            st.button("Start with the big picture →", key="home_go_chain",
                      on_click=_goto, args=("ANN", "views/the_chain.py"),
                      use_container_width=True)
            st.button("See the 5-level roadmap →", key="home_go_dash",
                      on_click=_goto, args=("ANN", "views/dashboard.py"),
                      use_container_width=True)
    with t[1]:
        with st.container(border=True):
            st.markdown("#### 🧮 ML\n**Classical machine learning.** Regression, classification, "
                        "trees, SVMs, clustering, model selection, and doing it for real in "
                        "scikit-learn (M0–M8).")
            st.button("Open the ML track →", key="home_go_ml", on_click=_switch_module,
                      args=("ML",), use_container_width=True)
    with t[2]:
        with st.container(border=True):
            st.markdown("#### 📐 Math\n**The foundations, as needed.** Vectors & matrices, "
                        "calculus & gradients, probability, optimization, information theory, "
                        "numerics (X1–X6).")
            st.button("Open the Math track →", key="home_go_math", on_click=_switch_module,
                      args=("Math",), use_container_width=True)


    # --- how to use it ---------------------------------------------------------- #
    with st.container(border=True):
        st.markdown("### 🔁 How to use this lab")
        lc = st.columns([0.55, 0.45])
        with lc[0]:
            st.markdown(_LOOP_SVG, unsafe_allow_html=True)
        with lc[1]:
            st.markdown(
                "Every page follows the same loop:\n\n"
                "1. **Predict** — each page opens with a 🔮 question; commit to a guess first.\n"
                "2. **Play** — move the sliders; every number on screen recomputes live.\n"
                "3. **Check** — instant-feedback quizzes and ✅ worked solutions for every task.\n"
                "4. **Rebuild** — the *Study Coach* and code experiments help you reconstruct "
                "ideas from memory — the only proof you've learned them.\n\n"
                "Work top-to-bottom through the levels, or jump straight to what you're curious "
                "about — the math track is there whenever a page leans on it."
            )
            st.button("Study Coach — a guided routine →", key="home_go_coach",
                      on_click=_goto, args=("ANN", "views/study_coach.py"))


with tab_method:
    with st.container(border=True):
        st.markdown("### 🎓 How I actually studied this")
        st.info(
            "**I did not read my way to understanding — I built my way there.** These notes are the "
            "residue of that process, and they are laid out the same way I worked:\n\n"
            "**1 · Interactive labs before prose.** I learn a concept by *moving it*. So almost every "
            "idea here has a playground where the numbers recompute as you drag a slider — because "
            "watching a decision boundary tilt taught me more in a minute than a page of text did.\n\n"
            "**2 · A starting point, not a finished lecture.** For each topic I wrote down just "
            "enough to begin — the core definition, a worked number, one diagram — and let the "
            "exercises carry the rest. Treat every page as a launch pad, not a destination.\n\n"
            "**3 · Pen and paper, always.** The derivations were done by hand before they were typed. "
            "If I could not reproduce a gradient on paper, I did not understand it yet — and that is "
            "why the worked solutions show the arithmetic rather than just the answer.\n\n"
            "**4 · I read around everything, like a student.** No single source is enough. I cross-read "
            "textbooks, papers, lectures and other people's explanations until the ideas agreed — the "
            "**References** tab on each page lists what I leaned on, and you should go there too.\n\n"
            "*If you use these notes the same way — play first, derive on paper, then read widely — "
            "they will work far better than reading them straight through.*",
            icon=":material/school:")


with tab_run:
    # --- run it locally --------------------------------------------------------- #
    with st.container(border=True):
        st.markdown("### 🖥 Best experience: run it on your own machine")
        _local = os.environ.get("AILAB_ENABLE_SANDBOX") == "1"
        if _local:
            st.success(
                "**You are running the lab locally — everything is unlocked**, including the 🐍 "
                "Sandbox in the sidebar. The notes below describe what visitors to the hosted site "
                "are missing.", icon=":material/check_circle:")
        else:
            st.info(
                "**This site is the read-only edition.** The lab was built to be *run*, not just "
                "read — and a few of the best parts only exist locally. Cloning it takes about two "
                "minutes.", icon=":material/rocket_launch:")

        rc = st.columns(2)
        with rc[0]:
            st.markdown(
                "**What you additionally get locally**\n\n"
                "- 🐍 **The Python Sandbox** — a live scratchpad with `numpy` and the lab's own "
                "`core` preloaded, so you can test every idea immediately. **Not available on this "
                "site.**\n"
                "- 🔬 **The code experiments** — `e01`–`e21` actually execute, including the real "
                "**PyTorch nanoGPT** you can train on your own GPU/Apple-Silicon.\n"
                "- ✅ **The test suite** — `pytest` proves the autograd engine's gradients are "
                "correct.\n"
                "- ✏️ **Edit anything** — change a lesson, add a playground, break something on "
                "purpose (the fastest way to learn).\n"
                "- ⚡ **Full speed, no limits**, and nothing leaves your machine."
            )
        with rc[1]:
            st.markdown("**Prerequisites**")
            st.markdown(
                "- **Python 3.10+** (developed on 3.14) and **git**\n"
                "- ~**1.5 GB** disk for the virtual environment\n"
                "- Any OS — macOS, Linux, or Windows\n"
                "- *Optional:* **PyTorch** (`requirements-dl.txt`) only for the nanoGPT "
                "experiment; everything else is NumPy + scikit-learn"
            )
            st.markdown("**Install and run**")
            st.code(
                "git clone https://github.com/gokcol/AI-LAB.git\n"
                "cd AI-LAB\n"
                "python3 -m venv .venv\n"
                "source .venv/bin/activate        # Windows: .venv\\Scripts\\activate\n"
                "pip install -e '.[gui]'\n"
                "./start.sh                       # → http://localhost:8501",
                language="bash")
            st.caption("`./stop.sh` stops it. On Windows, run "
                       "`streamlit run gui/app.py` instead of `./start.sh`.")


TERMS_DIR = GUI / "content"


def _pick_terms_lang():
    """Callback for the Terms language switch."""
    st.session_state["reader_lang"] = (
        "tr" if st.session_state.get("_terms_pick") == "Türkçe" else "en")


def terms_lang() -> str:
    """Language for the Terms. Shared with the consent notice on the feedback form, so a
    reader who switches here switches both (see feedback.reader_lang)."""
    return feedback.reader_lang()


@st.cache_data(show_spinner=False)
def _terms_text(lang: str) -> tuple[str, str]:
    """(short version, full terms) for one language, with the live values filled in."""
    raw = (TERMS_DIR / f"terms.{lang}.md").read_text(encoding="utf-8")
    raw = raw.replace("{retention}", str(feedback.RETENTION_DAYS)) \
             .replace("{region}", feedback.DATA_LOCATION)
    short, _, full = raw.partition("<!--FULL-->")
    return short.strip(), full.strip()


def _terms_body():
    """Rendered in the Terms tab and in the footer dialog -- one source, two languages."""
    lang = terms_lang()
    # A two-item switch, right-aligned above the text. It appears only here, so the rest
    # of the UI is untouched for every visitor.
    #
    # Driven by on_change rather than by comparing the widget's return value: a widget
    # that is given BOTH a default and a session-state key fights itself -- on the rerun
    # the stored value wins, flips the language straight back, and the switch appears
    # dead. Seed the key once, then let the callback own the change.
    want_label = "Türkçe" if lang == "tr" else "English"
    if "_terms_pick" not in st.session_state:
        st.session_state["_terms_pick"] = want_label
    sw = st.columns([0.62, 0.38])
    with sw[1]:
        st.segmented_control(
            "", ["English", "Türkçe"], key="_terms_pick",
            label_visibility="collapsed", on_change=_pick_terms_lang,
        )

    short, full = _terms_text(lang)
    st.markdown(short)
    with st.expander("Tüm koşulları okuyun" if lang == "tr" else "Read the full terms"):
        st.markdown(full)


@st.dialog("📜 Terms of use & privacy", width="large")
def _terms_dialog():
    _terms_body()


with tab_terms:
    with st.container(border=True):
        _terms_body()


with tab_fb:
    # --- feedback --------------------------------------------------------------- #
    st.markdown("### 💬 Feedback")
    st.markdown("Found a mistake? Want a topic covered? A quick note helps the lab improve — "
                "name and email are optional.")
    feedback.render_form()


with tab_about:
    # --- about the author ------------------------------------------------------- #
    with st.container(border=True):
        st.markdown("### 👤 About the author — whose notes these are")
        st.markdown(_TIMELINE_SVG, unsafe_allow_html=True)
        st.markdown(
            "**Orhan Gökçöl** has been writing software since **1987** — starting on IBM mainframes "
            "in Assembler and REXX, then Fortran for scientific computing, analysing and visualizing "
            "the interplanetary magnetic field for his MSc. During those graduate years he completed "
            "a **PhD-level course on the mathematical foundations of artificial neural networks "
            "(1993–94)** — the very mathematics this lab is built on. He earned a **PhD in "
            "Aeronautics** at Istanbul Technical University in 1999, where computational engineering "
            "led him to model industrial processes across manufacturing, retail, finance, electronics "
            "and education, writing his own simulation and visualization codes.\n\n"
            "Through the 1990s he built ERP-style business systems for SMEs and the enterprise Linux "
            "infrastructure they ran on, and helped bring the internet to industry and the public. "
            "From **1999–2006** he led a university's IT services and its Cyber Technologies Research "
            "Center while teaching Computer and Mechatronics Engineering — object-oriented "
            "programming, data structures, computer graphics, web programming — and, with "
            "psychologists and educators, built career-counselling tools used by **millions of "
            "students**. He later developed graduate IT programmes at Bahçeşehir University, and in "
            "**2016** founded his own consultancy, training and audit practice, where he works as a "
            "senior auditor and trainer in information security and IT resilience "
            "(**ISO/IEC 27001, ISO/SAE 21434, IEC 62443, TISAX**).\n\n"
            "Since **2021** he has mentored new graduates and students in software engineering, and "
            "since **2025** he has used large language models and agentic tools professionally for "
            "software development — the subject of a forthcoming book. Across that arc he has written "
            "software in Assembler, BASIC, REXX, Fortran, Pascal, C, C++, Java, C#/.NET, Perl, PHP "
            "and Python: *from punched mainframe decks to LLM agents*.\n\n"
            "**These notes are a return to those 1993–94 foundations** — relearning neural networks "
            "from the single neuron up, thirty years on, now that they run the world. That is all "
            "this site is: one engineer's study notebook, kept in public."
        )
        ac = st.columns([0.42, 0.58])
        ac[0].link_button("👤 LinkedIn — /in/gokcol", "https://www.linkedin.com/in/gokcol",
                          use_container_width=True)
        ac[1].link_button("⭐ This lab's source on GitHub", "https://github.com/gokcol/AI-LAB",
                          use_container_width=True)
        st.markdown(
            "> *“What I cannot create, I do not understand.”* — **Richard Feynman**, blackboard note "
            "(1988).  \n"
            "That line is why this lab exists: everything here is rebuilt from scratch, in code you "
            "can run."
        )


    # --- version history -------------------------------------------------------- #
    with st.container(border=True):
        vh = st.columns([0.72, 0.28])
        vh[0].markdown("### 🗓 Version history")
        vh[1].link_button("⭐ Source on GitHub", "https://github.com/gokcol/AI-LAB",
                          use_container_width=True)
        for date, ver, notes in VERSION_HISTORY:
            st.markdown(f"**v{ver}** · {date} — {notes}")


st.divider()
st.caption(f"AI Lab v{VERSION} · personal study notes by Orhan Gökçöl, written with Claude · "
           "source: [github.com/gokcol/AI-LAB](https://github.com/gokcol/AI-LAB) · "
           "hosted at **ai-lab.gokcol.online** 🧠")
# The disclaimer now opens inside the Welcome tab (the default one), so it is still the
# first thing a visitor reads — but a reader who lands on another tab must not be able to
# miss it entirely. This line is outside the tabs and therefore always on screen.
fc = st.columns([0.78, 0.22])
fc[0].caption("⚠️ Study notes, not a textbook: **no guarantee of accuracy and no liability** "
              "for any error or omission — verify everything before relying on it.")
# Streamlit has no API for selecting a tab programmatically, so "see the Terms" cannot be
# a link to the tab. A dialog reading the very same _terms_body() is the honest fix: one
# source of truth, reachable from anywhere on the page, and it works on a phone.
if fc[1].button("📜 Read the Terms", type="tertiary", use_container_width=True):
    _terms_dialog()
