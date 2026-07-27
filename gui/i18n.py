"""Tiny EN/TR translation helper for the Streamlit GUI.

Keep UI chrome here. Longer lesson/experiment content should live in localized
Markdown files and be read through `lab.read_localized`.
"""

from __future__ import annotations

import functools

import streamlit as st

LANGUAGES = {
    "en": "English",
    "tr": "Türkçe",
}

DEFAULT_LANG = "en"

# The training journey is now maintained in both tracks, so expose the existing
# Turkish catalog and language selector again.
MULTILINGUAL = True

TEXT = {
    "app.brand": {"en": "AI Lab", "tr": "AI Lab"},
    "app.language": {"en": "Language", "tr": "Dil"},
    "app.module": {"en": "Module", "tr": "Modül"},
    "app.module_help": {
        "en": "ANN = neural networks (single neuron → GPT) · ML = classical machine learning · Math = the maths behind both.",
        "tr": "ANN = sinir ağları (tek nörondan GPT'ye) · ML = klasik makine öğrenmesi · Math = ikisinin arkasındaki matematik.",
    },
    "app.footer": {
        "en": "**Orhan Gökçöl's** personal AI lab · free to use",
        "tr": "**Orhan Gökçöl'ün** kişisel AI laboratuvarı · kullanımı serbest",
    },
    "track.ann": {
        "en": "Neural networks — a single neuron → a GPT.",
        "tr": "Sinir ağları — tek nörondan GPT'ye.",
    },
    "track.ml": {
        "en": "Classical machine learning (M0–M8).",
        "tr": "Klasik makine öğrenmesi (M0-M8).",
    },
    "track.math": {
        "en": "The foundations — dip in as needed.",
        "tr": "Temeller — ihtiyaç oldukça geri dön.",
    },
    "section.overview": {"en": "Overview", "tr": "Genel Bakış"},
    "section.level1": {"en": "Level 1 · The neuron", "tr": "Seviye 1 · Nöron"},
    "section.level2": {"en": "Level 2 · Training a net", "tr": "Seviye 2 · Ağı eğitmek"},
    "section.level3": {"en": "Level 3 · Architectures", "tr": "Seviye 3 · Mimariler"},
    "section.level4": {"en": "Level 4 · Attention & Transformers", "tr": "Seviye 4 · Attention ve Transformer"},
    "section.level5": {"en": "Level 5 · LLMs in practice", "tr": "Seviye 5 · Pratikte LLM'ler"},
    "section.lab": {"en": "Lab", "tr": "Laboratuvar"},
    "section.supervised": {"en": "Supervised (M1–M4)", "tr": "Denetimli (M1-M4)"},
    "section.unsupervised": {"en": "Unsupervised (M5)", "tr": "Denetimsiz (M5)"},
    "section.practice": {"en": "Practice (M6–M8)", "tr": "Pratik (M6-M8)"},
    "section.algebra": {"en": "Algebra & calculus", "tr": "Cebir ve kalkülüs"},
    "section.probability": {"en": "Probability & optimization", "tr": "Olasılık ve optimizasyon"},
    "section.information": {"en": "Information & numerics", "tr": "Bilgi kuramı ve sayısal hesap"},
    "section.tools": {"en": "Tools", "tr": "Araçlar"},
    "page.home": {"en": "Home", "tr": "Ana sayfa"},
    "page.dashboard": {"en": "Dashboard", "tr": "Panel"},
    "page.study_coach": {"en": "Study Coach", "tr": "Çalışma Koçu"},
    "page.big_picture": {"en": "The big picture", "tr": "Büyük resim"},
    "page.neuron_playground": {"en": "Neuron playground", "tr": "Nöron oyun alanı"},
    "page.two_neurons": {"en": "Two neurons", "tr": "İki nöron"},
    "page.neurons_compute": {"en": "Neurons → computer", "tr": "Nöronlar → bilgisayar"},
    "page.worked_numbers": {"en": "Worked examples (numbers)", "tr": "Sayısal örnekler"},
    "page.worked_numbers2": {"en": "Worked examples II (bigger)", "tr": "Sayısal örnekler II"},
    "page.activations": {"en": "Activation functions", "tr": "Aktivasyon fonksiyonları"},
    "page.backprop": {"en": "Backprop", "tr": "Backprop"},
    "page.mlp": {"en": "MLP (train it)", "tr": "MLP (eğit)"},
    "page.optimizers": {"en": "Optimizers", "tr": "Optimizasyon algoritmaları"},
    "page.deep_nets": {"en": "Deep nets (2D)", "tr": "Derin ağlar (2D)"},
    "page.regularization": {"en": "Regularization", "tr": "Regularizasyon"},
    "page.cnn": {"en": "CNN (images)", "tr": "CNN (görüntüler)"},
    "page.rnn": {"en": "RNN (sequences)", "tr": "RNN (diziler)"},
    "page.tokenization": {"en": "Tokenization", "tr": "Tokenizasyon"},
    "page.attention": {"en": "Attention (LLMs)", "tr": "Attention (LLM'ler)"},
    "page.tiny_gpt": {"en": "Tiny GPT", "tr": "Küçük GPT"},
    "page.interpretability": {"en": "Interpretability",
                              "tr": "Yorumlanabilirlik"},
    "page.generative": {"en": "Generative family",
                        "tr": "Üretken model ailesi"},
    "page.sampling": {"en": "Decoding (sampling)", "tr": "Decoding (sampling)"},
    "page.embeddings": {"en": "Embeddings & RAG", "tr": "Embedding ve RAG"},
    "page.posttraining": {"en": "Post-training (RLHF)", "tr": "Post-training (RLHF)"},
    "page.experiments": {"en": "Experiments", "tr": "Deneyler"},
    "page.infobase": {"en": "Infobase", "tr": "Bilgi tabanı"},
    "page.tests": {"en": "Tests", "tr": "Testler"},
    "page.sandbox": {"en": "Sandbox", "tr": "Sandbox"},
    "page.ml_overview": {"en": "ML overview", "tr": "ML genel bakış"},
    "page.ml_foundations": {"en": "M0 · Foundations", "tr": "M0 · Temeller"},
    "page.ml_regression": {"en": "M1 · Regression", "tr": "M1 · Regresyon"},
    "page.ml_classification": {"en": "M2 · Classification", "tr": "M2 · Sınıflandırma"},
    "page.ml_trees": {"en": "M3 · Trees & ensembles", "tr": "M3 · Ağaçlar ve ensemble"},
    "page.ml_svm": {"en": "M4 · SVM & kernels", "tr": "M4 · SVM ve kernel"},
    "page.ml_unsupervised": {"en": "M5 · Unsupervised", "tr": "M5 · Denetimsiz öğrenme"},
    "page.ml_model_selection": {"en": "M6 · Model selection", "tr": "M6 · Model seçimi"},
    "page.ml_practical": {"en": "M7 · Practical ML", "tr": "M7 · Pratik ML"},
    "page.ml_python": {"en": "M8 · ML in Python", "tr": "M8 · Python ile ML"},
    "page.math_overview": {"en": "Math overview", "tr": "Matematik genel bakış"},
    "page.math_vectors": {"en": "X1 · Vectors & dot product", "tr": "X1 · Vektörler ve dot product"},
    "page.math_calculus": {"en": "X2 · Calculus & gradients", "tr": "X2 · Kalkülüs ve gradyanlar"},
    "page.math_probability": {"en": "X3 · Probability & stats", "tr": "X3 · Olasılık ve istatistik"},
    "page.math_optimization": {"en": "X4 · Optimization", "tr": "X4 · Optimizasyon"},
    "page.math_information": {"en": "X5 · Information theory", "tr": "X5 · Bilgi kuramı"},
    "page.math_numerical": {"en": "X6 · Numerical computing", "tr": "X6 · Sayısal hesaplama"},
}


def lang() -> str:
    if not MULTILINGUAL:
        return DEFAULT_LANG
    return st.session_state.get("lang", DEFAULT_LANG)


def set_lang(value: str) -> None:
    st.session_state["lang"] = value if value in LANGUAGES else DEFAULT_LANG


def t(key: str, *, language: str | None = None) -> str:
    language = language or lang()
    item = TEXT.get(key)
    if item is None:
        return key
    return item.get(language) or item.get(DEFAULT_LANG) or key


def select_language() -> str:
    if not MULTILINGUAL:  # selector hidden — English only
        return DEFAULT_LANG
    current = lang()
    labels = list(LANGUAGES.values())
    codes = list(LANGUAGES.keys())
    default_index = codes.index(current) if current in codes else 0
    label = st.segmented_control(
        t("app.language", language=current),
        labels,
        default=labels[default_index],
        key="language_selector",
    )
    selected = codes[labels.index(label or labels[default_index])]
    set_lang(selected)
    return selected


# --------------------------------------------------------------------------- #
# Render-time content localization
#
# Page files stay 100% English (one source of truth). When the language is "tr",
# we transparently swap each *static* English UI string for its Turkish entry in
# catalog_tr.TR. Anything not in the catalog (dynamic f-strings, SVG diagrams,
# code) falls through to English, so the app always renders. This wraps the
# Streamlit text helpers once, at startup, instead of editing 35 view files.
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _catalog() -> dict[str, str]:
    try:
        from catalog_tr import TR
        return TR
    except Exception:
        return {}


def localize(text: str) -> str:
    """Return the Turkish version of a known UI string, else the input unchanged."""
    if not isinstance(text, str) or lang() == DEFAULT_LANG:
        return text
    return _catalog().get(text, text)


_LOCALIZED_FLAG = "_ailab_localized"
_TEXT_FNS = ("title", "header", "subheader", "markdown", "caption",
             "info", "success", "warning", "write")
# Widgets whose first positional arg is a user-facing label (NOT page_link — its
# first arg is a file path that must never be translated).
_WIDGET_FNS = ("metric", "button", "download_button", "link_button", "checkbox",
               "toggle", "radio", "selectbox", "multiselect", "select_slider",
               "slider", "number_input", "text_input", "text_area",
               "segmented_control", "pills", "color_picker", "expander", "status")
_LIST_FNS = ("tabs",)


def install_localization() -> None:
    """Wrap Streamlit's text helpers so they localize their text argument.

    Idempotent: safe to call on every rerun. Patches both the top-level
    ``st.*`` helpers and ``DeltaGenerator`` methods (so ``col.markdown(...)``,
    ``tab.caption(...)`` etc. are localized too)."""
    from streamlit.delta_generator import DeltaGenerator

    if getattr(st, _LOCALIZED_FLAG, False):
        return

    # Localize the FIRST string argument. This handles both the bound module
    # helpers (st.markdown("x") -> args[0] is the text) and the DeltaGenerator
    # methods used on columns/containers (col.markdown("x") -> args[0] is self,
    # args[1] is the text). Labels/content always come before any other string.
    def wrap_text(orig):
        @functools.wraps(orig)
        def inner(*args, **kwargs):
            args = list(args)
            for i, a in enumerate(args):
                if isinstance(a, str):
                    args[i] = localize(a)
                    break
            return orig(*args, **kwargs)
        return inner

    # Localize the first list/tuple argument element-wise (e.g. st.tabs([...])).
    def wrap_list(orig):
        @functools.wraps(orig)
        def inner(*args, **kwargs):
            args = list(args)
            for i, a in enumerate(args):
                if isinstance(a, (list, tuple)):
                    args[i] = [localize(x) if isinstance(x, str) else x for x in a]
                    break
            return orig(*args, **kwargs)
        return inner

    for name in _TEXT_FNS + _WIDGET_FNS:
        if hasattr(DeltaGenerator, name):
            setattr(DeltaGenerator, name, wrap_text(getattr(DeltaGenerator, name)))
        if hasattr(st, name):
            setattr(st, name, wrap_text(getattr(st, name)))
    for name in _LIST_FNS:
        if hasattr(DeltaGenerator, name):
            setattr(DeltaGenerator, name, wrap_list(getattr(DeltaGenerator, name)))
        if hasattr(st, name):
            setattr(st, name, wrap_list(getattr(st, name)))

    setattr(st, _LOCALIZED_FLAG, True)
