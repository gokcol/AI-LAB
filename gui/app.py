"""AI Lab GUI — entrypoint.

Three modules, selectable in the sidebar:
  - ANN  : neural networks, single neuron -> GPT (what you've been using).
  - ML   : classical machine learning, same approach (theory + playground + quiz).
  - Math : the mathematical basics needed to study ML & ANN.

Launch:
    cd ai-lab && source .venv/bin/activate
    streamlit run gui/app.py
"""

import os
import pathlib
import sys

# Matplotlib must use a non-GUI backend: Streamlit runs scripts on a worker thread, and the
# macOS GUI backend refuses to create figures off the main thread. Set before any view loads.
os.environ.setdefault("MPLBACKEND", "Agg")

import streamlit as st
import i18n
import ui

# Make the lab root importable so `import core` works without `pip install -e .`
# (e.g. on Streamlit Community Cloud, which only installs requirements.txt).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="AI Lab", page_icon="🧠", layout="wide")
ui.inject_theme()
i18n.install_localization()  # transparently localizes static UI text when lang == "tr"

# --- sidebar: brand + language/module selectors pinned to the TOP ----------
with st.sidebar:
    st.markdown(f"## 🧠 {i18n.t('app.brand')}")
    i18n.select_language()
    track_blurb = {
        "ANN": i18n.t("track.ann"),
        "ML": i18n.t("track.ml"),
        "Math": i18n.t("track.math"),
    }
    module = st.segmented_control(
        i18n.t("app.module"), ["ANN", "ML", "Math"], default="ANN", key="module",
        help=i18n.t("app.module_help"),
    ) or "ANN"
    st.caption(track_blurb[module])
    st.divider()


def page(path: str, title_key: str, icon: str, *, default: bool = False):
    return st.Page(path, title=i18n.t(title_key), icon=icon, default=default)


# Shared across modules — a Python scratchpad with numpy + the lab's core preloaded.
SANDBOX = page("views/sandbox.py", "page.sandbox", ":material/code:")
COACH = page("views/study_coach.py", "page.study_coach", ":material/self_improvement:")

# Pages are grouped into sections that follow the learning path. Passing a {section: [pages]}
# dict to st.navigation renders the keys as sidebar section headers (much less crowded).
ANN = {
    i18n.t("section.overview"): [
        page("views/dashboard.py", "page.dashboard", ":material/dashboard:", default=True),
        COACH,
        page("views/the_chain.py", "page.big_picture", ":material/route:"),
    ],
    i18n.t("section.level1"): [
        page("views/playground.py", "page.neuron_playground", ":material/tune:"),
        page("views/two_neurons.py", "page.two_neurons", ":material/hub:"),
        page("views/neurons_compute.py", "page.neurons_compute", ":material/calculate:"),
    ],
    i18n.t("section.level2"): [
        page("views/worked_numbers.py", "page.worked_numbers", ":material/functions:"),
        page("views/worked_numbers2.py", "page.worked_numbers2", ":material/grid_on:"),
        page("views/activations.py", "page.activations", ":material/ssid_chart:"),
        page("views/backprop.py", "page.backprop", ":material/sync_alt:"),
        page("views/mlp.py", "page.mlp", ":material/network_node:"),
        page("views/optimizers.py", "page.optimizers", ":material/trending_down:"),
        page("views/deep_playground.py", "page.deep_nets", ":material/blur_on:"),
        page("views/regularization.py", "page.regularization", ":material/tune:"),
    ],
    i18n.t("section.level3"): [
        page("views/cnn.py", "page.cnn", ":material/image:"),
        page("views/rnn.py", "page.rnn", ":material/repeat:"),
    ],
    i18n.t("section.level4"): [
        page("views/tokenization.py", "page.tokenization", ":material/content_cut:"),
        page("views/attention.py", "page.attention", ":material/auto_awesome:"),
        page("views/transformer.py", "page.tiny_gpt", ":material/smart_toy:"),
    ],
    i18n.t("section.level5"): [
        page("views/sampling.py", "page.sampling", ":material/casino:"),
        page("views/embeddings.py", "page.embeddings", ":material/database:"),
        page("views/posttraining.py", "page.posttraining", ":material/psychology:"),
    ],
    i18n.t("section.lab"): [
        page("views/experiments.py", "page.experiments", ":material/science:"),
        page("views/infobase.py", "page.infobase", ":material/menu_book:"),
        page("views/tests.py", "page.tests", ":material/check_circle:"),
    ],
}
ML = {
    i18n.t("section.overview"): [
        page("views/ml_overview.py", "page.ml_overview", ":material/dashboard:", default=True),
        page("views/ml_foundations.py", "page.ml_foundations", ":material/school:"),
    ],
    i18n.t("section.supervised"): [
        page("views/ml_regression.py", "page.ml_regression", ":material/timeline:"),
        page("views/ml_classification.py", "page.ml_classification", ":material/scatter_plot:"),
        page("views/ml_trees.py", "page.ml_trees", ":material/account_tree:"),
        page("views/ml_svm.py", "page.ml_svm", ":material/linear_scale:"),
    ],
    i18n.t("section.unsupervised"): [
        page("views/ml_unsupervised.py", "page.ml_unsupervised", ":material/bubble_chart:"),
    ],
    i18n.t("section.practice"): [
        page("views/ml_model_selection.py", "page.ml_model_selection", ":material/checklist:"),
        page("views/ml_practical.py", "page.ml_practical", ":material/build:"),
        page("views/ml_python.py", "page.ml_python", ":material/code:"),
    ],
}
MATH = {
    i18n.t("section.overview"): [
        page("views/math_overview.py", "page.math_overview", ":material/dashboard:", default=True),
    ],
    i18n.t("section.algebra"): [
        page("views/math_vectors.py", "page.math_vectors", ":material/call_made:"),
        page("views/math_calculus.py", "page.math_calculus", ":material/show_chart:"),
    ],
    i18n.t("section.probability"): [
        page("views/math_probability.py", "page.math_probability", ":material/bar_chart:"),
        page("views/math_optimization.py", "page.math_optimization", ":material/trending_down:"),
    ],
    i18n.t("section.information"): [
        page("views/math_information.py", "page.math_information", ":material/data_usage:"),
        page("views/math_numerical.py", "page.math_numerical", ":material/calculate:"),
    ],
}
MODULES = {"ANN": ANN, "ML": ML, "Math": MATH}

# The Sandbox runs arbitrary Python in-process, so expose it ONLY when explicitly
# enabled — start.sh sets AILAB_ENABLE_SANDBOX=1 locally; it's unset on a public
# deploy (Streamlit Cloud), so the Sandbox is hidden there.
SANDBOX_ENABLED = os.environ.get("AILAB_ENABLE_SANDBOX") == "1"
if SANDBOX_ENABLED:
    for _sections in MODULES.values():
        _sections[i18n.t("section.tools")] = [SANDBOX]

# position="hidden": we render our own grouped links below, so the Module selector sits on top.
nav = st.navigation(MODULES[module], position="hidden")

with st.sidebar:
    for section, pages in MODULES[module].items():
        st.markdown(
            f"<div style='font-size:.72rem;font-weight:700;letter-spacing:.04em;"
            f"text-transform:uppercase;color:#9C9B95;margin:.6rem 0 .1rem'>{section}</div>",
            unsafe_allow_html=True,
        )
        for p in pages:
            st.page_link(p)
    st.divider()
    st.caption(f"{i18n.t('app.footer')} 🧠")

nav.run()
