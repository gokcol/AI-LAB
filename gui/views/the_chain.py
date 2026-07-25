"""The big picture — from one neuron to computing (ANN module).

An attractive, illustrated map of the whole lab: what a neural network is, the chain from
one neuron to a GPT, the single operation it all rests on, and the two views (logic +
function) that meet in a trained network — with links into every stop.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

# --------------------------------------------------------------------------- #
# Illustrations
# --------------------------------------------------------------------------- #
_HERO_SVG = '''<div style="text-align:center;margin:0.2rem 0 0.6rem"><svg viewBox="0 0 760 250" style="width:100%;max-width:820px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The chain from a single neuron to a GPT: one neuron → logic gate → a layer → MLP → Transformer → LLM, each step a weighted sum plus a nonlinearity."><defs><marker id="chah" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#7A8A99"/></marker><linearGradient id="chbg" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#EAF1FB"/><stop offset="1" stop-color="#E6F4EC"/></linearGradient></defs><rect x="1" y="1" width="758" height="248" rx="16" fill="url(#chbg)" stroke="#E2E2DA"/><text x="380" y="34" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="700" fill="#2C3A33">From a single neuron to a GPT</text><g font-family="sans-serif"><rect x="18" y="92" width="104" height="56" rx="9" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.8"/><rect x="142" y="92" width="104" height="56" rx="9" fill="#E1EDF1" stroke="#4F96AC" stroke-width="1.8"/><rect x="266" y="92" width="104" height="56" rx="9" fill="#DEEFEA" stroke="#2E9E8C" stroke-width="1.8"/><rect x="390" y="92" width="104" height="56" rx="9" fill="#DCEFE2" stroke="#1D9E75" stroke-width="1.8"/><rect x="514" y="92" width="104" height="56" rx="9" fill="#DAEFDC" stroke="#2FA15C" stroke-width="1.8"/><rect x="638" y="92" width="104" height="56" rx="9" fill="#D8EFD7" stroke="#3DA147" stroke-width="1.8"/><g font-size="14.5" font-weight="600" fill="#2C3A33" text-anchor="middle"><text x="70" y="123">1 neuron</text><text x="194" y="123">logic gate</text><text x="318" y="123">a layer</text><text x="442" y="123">MLP</text><text x="576" y="123">Transformer</text><text x="690" y="123">GPT / LLM</text></g><g font-size="9.5" fill="#6B6A66" text-anchor="middle"><text x="70" y="140">a weighted vote</text><text x="194" y="140">AND / OR</text><text x="318" y="140">features</text><text x="442" y="140">any function</text><text x="576" y="140">attention</text><text x="690" y="140">next token</text></g><g stroke="#7A8A99" stroke-width="2.2"><line x1="122" y1="120" x2="139" y2="120" marker-end="url(#chah)"/><line x1="246" y1="120" x2="263" y2="120" marker-end="url(#chah)"/><line x1="370" y1="120" x2="387" y2="120" marker-end="url(#chah)"/><line x1="494" y1="120" x2="511" y2="120" marker-end="url(#chah)"/><line x1="618" y1="120" x2="635" y2="120" marker-end="url(#chah)"/></g></g><text x="380" y="222" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5A6B61">every box is the same move — a weighted sum (matrix multiply) + a nonlinearity — just bigger</text></svg></div>'''

_NEURON_SVG = '''<div style="text-align:center;margin:.4rem 0"><svg viewBox="0 0 440 175" style="width:100%;max-width:430px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A single neuron: three inputs are each multiplied by a weight, summed with a bias, then passed through an activation to produce one output number."><defs><marker id="bgn" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9C9B95"/></marker></defs><rect x="1" y="1" width="438" height="173" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g stroke="#9C9B95" stroke-width="1.5" fill="none"><line x1="62" y1="46" x2="191" y2="84" marker-end="url(#bgn)"/><line x1="62" y1="88" x2="191" y2="88" marker-end="url(#bgn)"/><line x1="62" y1="130" x2="191" y2="94" marker-end="url(#bgn)"/><line x1="241" y1="88" x2="289" y2="88" marker-end="url(#bgn)"/><line x1="337" y1="88" x2="384" y2="88" marker-end="url(#bgn)"/></g><g fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.6"><circle cx="46" cy="46" r="16"/><circle cx="46" cy="88" r="16"/><circle cx="46" cy="130" r="16"/></g><g font-family="sans-serif" font-size="11" fill="#0C447C" text-anchor="middle"><text x="46" y="50">x₁</text><text x="46" y="92">x₂</text><text x="46" y="134">x₃</text></g><g font-family="sans-serif" font-size="10" fill="#9A6A2A" text-anchor="middle"><text x="118" y="56">w₁</text><text x="126" y="82">w₂</text><text x="118" y="120">w₃</text></g><circle cx="215" cy="88" r="26" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.8"/><text x="215" y="86" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#5A3E14">Σ</text><text x="215" y="103" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#5A3E14">+ b</text><circle cx="315" cy="88" r="22" fill="#D7EFE5" stroke="#1D9E75" stroke-width="1.8"/><text x="315" y="94" text-anchor="middle" font-family="sans-serif" font-size="17" fill="#0E5E45">φ</text><text x="396" y="93" font-family="sans-serif" font-size="16" fill="#33312E">a</text><g font-family="sans-serif" font-size="9.5" fill="#9C9B95" text-anchor="middle"><text x="46" y="162">inputs</text><text x="215" y="130">weighted sum</text><text x="315" y="124">activation</text><text x="398" y="112">output</text></g></svg></div>'''

_SCALE_SVG = '''<div style="text-align:center;margin:.4rem 0"><svg viewBox="0 0 600 165" style="width:100%;max-width:600px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Scale: one neuron, then a layer of a few, then a GPT — the same unit repeated up to about a hundred billion times."><defs><marker id="scah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9C9B95"/></marker></defs><rect x="1" y="1" width="598" height="163" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><circle cx="60" cy="82" r="12" fill="#185FA5"/><g fill="#1D9E75"><circle cx="200" cy="46" r="7"/><circle cx="200" cy="64" r="7"/><circle cx="200" cy="82" r="7"/><circle cx="200" cy="100" r="7"/><circle cx="200" cy="118" r="7"/></g><g fill="#9A6A2A"><circle cx="338" cy="48" r="3"/><circle cx="362" cy="48" r="3"/><circle cx="386" cy="48" r="3"/><circle cx="410" cy="48" r="3"/><circle cx="434" cy="48" r="3"/><circle cx="458" cy="48" r="3"/><circle cx="482" cy="48" r="3"/><circle cx="506" cy="48" r="3"/><circle cx="530" cy="48" r="3"/><circle cx="338" cy="66" r="3"/><circle cx="362" cy="66" r="3"/><circle cx="386" cy="66" r="3"/><circle cx="410" cy="66" r="3"/><circle cx="434" cy="66" r="3"/><circle cx="458" cy="66" r="3"/><circle cx="482" cy="66" r="3"/><circle cx="506" cy="66" r="3"/><circle cx="530" cy="66" r="3"/><circle cx="338" cy="84" r="3"/><circle cx="362" cy="84" r="3"/><circle cx="386" cy="84" r="3"/><circle cx="410" cy="84" r="3"/><circle cx="434" cy="84" r="3"/><circle cx="458" cy="84" r="3"/><circle cx="482" cy="84" r="3"/><circle cx="506" cy="84" r="3"/><circle cx="530" cy="84" r="3"/><circle cx="338" cy="102" r="3"/><circle cx="362" cy="102" r="3"/><circle cx="386" cy="102" r="3"/><circle cx="410" cy="102" r="3"/><circle cx="434" cy="102" r="3"/><circle cx="458" cy="102" r="3"/><circle cx="482" cy="102" r="3"/><circle cx="506" cy="102" r="3"/><circle cx="530" cy="102" r="3"/><circle cx="338" cy="120" r="3"/><circle cx="362" cy="120" r="3"/><circle cx="386" cy="120" r="3"/><circle cx="410" cy="120" r="3"/><circle cx="434" cy="120" r="3"/><circle cx="458" cy="120" r="3"/><circle cx="482" cy="120" r="3"/><circle cx="506" cy="120" r="3"/><circle cx="530" cy="120" r="3"/></g><g stroke="#9C9B95" stroke-width="1.8" fill="none"><line x1="82" y1="82" x2="170" y2="82" marker-end="url(#scah)"/><line x1="228" y1="82" x2="300" y2="82" marker-end="url(#scah)"/></g><g font-family="sans-serif" font-size="12" fill="#33312E" text-anchor="middle"><text x="60" y="142">1 neuron</text><text x="200" y="150">a layer</text><text x="434" y="150">a GPT</text></g><g font-family="sans-serif" font-size="9.5" fill="#9C9B95" text-anchor="middle"><text x="200" y="163">(tens–hundreds)</text><text x="434" y="163">(~100 billion)</text></g></svg></div>'''

_VIEWS_SVG = '''<div style="text-align:center;margin:.4rem 0"><svg viewBox="0 0 620 210" style="width:100%;max-width:620px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The same neuron seen two ways: a logic view (neuron → gate → circuit → computer) and a function view (neuron → MLP → any function); a trained network is both at once."><defs><marker id="vah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9C9B95"/></marker></defs><rect x="1" y="1" width="618" height="208" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><rect x="24" y="82" width="92" height="44" rx="9" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.8"/><text x="70" y="108" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#5A3E14">1 neuron</text><g font-family="sans-serif" font-size="12" text-anchor="middle"><rect x="180" y="22" width="96" height="38" rx="8" fill="#FBEAF0" stroke="#C0507A"/><text x="228" y="46" fill="#8A2351">logic gate</text><rect x="304" y="22" width="96" height="38" rx="8" fill="#FBEAF0" stroke="#C0507A"/><text x="352" y="46" fill="#8A2351">circuit</text><rect x="428" y="22" width="120" height="38" rx="8" fill="#FBEAF0" stroke="#C0507A"/><text x="488" y="46" fill="#8A2351">a computer</text><rect x="180" y="148" width="120" height="38" rx="8" fill="#E6F1FB" stroke="#5B8FC2"/><text x="240" y="172" fill="#0C447C">MLP (layers)</text><rect x="332" y="148" width="130" height="38" rx="8" fill="#E6F1FB" stroke="#5B8FC2"/><text x="397" y="172" fill="#0C447C">any function</text></g><g stroke="#9C9B95" stroke-width="1.6" fill="none"><path d="M116,96 C150,96 150,42 178,42" marker-end="url(#vah)"/><line x1="276" y1="41" x2="302" y2="41" marker-end="url(#vah)"/><line x1="400" y1="41" x2="426" y2="41" marker-end="url(#vah)"/><path d="M116,112 C150,112 150,167 178,167" marker-end="url(#vah)"/><line x1="300" y1="167" x2="330" y2="167" marker-end="url(#vah)"/></g><text x="492" y="80" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#C0507A">logic view</text><text x="397" y="200" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#5B8FC2">function view</text><text x="560" y="120" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6B6A66">both at</text><text x="560" y="133" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6B6A66">once ✦</text></svg></div>'''


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("🧭 The big picture — from one neuron to computing")
st.caption("A 5-minute map of the whole lab: what a neural network is, and how a single "
           "neuron scales all the way to a GPT.")

st.markdown(
    "**One idea scales to all of it.** A neuron is a tiny decision-maker: multiply each input "
    "by a *weight*, add them up, and bend the result with a simple curve. Wire a few together "
    "and they do **logic and arithmetic**. Stack and *train* them and they recognise images, "
    "translate languages, and — at billions strong — **write text**. This page is the map; "
    "the sidebar is the journey."
)
st.markdown(_HERO_SVG, unsafe_allow_html=True)

# --- What is a neural network? --------------------------------------------- #
with st.container(border=True):
    st.markdown("#### 🧩 What is a neural network? (in 60 seconds)")
    st.markdown(
        "A **neural network** is just **many neurons wired together and tuned by data**. "
        "Each neuron does the same little thing:"
    )
    st.markdown(_NEURON_SVG, unsafe_allow_html=True)
    left, right = st.columns([0.52, 0.48])
    with left:
        st.markdown(
            "| term | in plain words |\n|---|---|\n"
            "| **Neuron** | weights × inputs, summed, then a curve → one number |\n"
            "| **Weight** | how much an input matters *(learned)* |\n"
            "| **Bias** | a baseline / threshold *(learned)* |\n"
            "| **Activation** | the curve (ReLU, sigmoid…) that lets it bend |\n"
            "| **Layer** | many neurons working in parallel |\n"
            "| **Network** | layers stacked — each builds on the last |\n"
            "| **Training** | nudging every weight to reduce error |"
        )
    with right:
        st.info(
            "**The only operation, repeated:** a **weighted sum → a nonlinearity**. "
            "Do it once = a neuron. Do it in parallel = a layer. Do it deep = a network. "
            "Do it billions of times = an LLM.\n\n"
            "**Learning** = *backprop* finds how to change each weight, an *optimizer* takes "
            "the step (Levels 2 & the Math track).",
            icon=":material/lightbulb:")

# --- The chain, stop by stop ----------------------------------------------- #
with st.container(border=True):
    st.markdown("#### 🪜 The chain, stop by stop")
    st.caption("Each stop is a page in this lab — click to open it.")
    STOPS = [
        ("1 · One neuron", "A weighted vote → a yes/no decision (one straight line). "
         "*Example: is this email spam?*", "views/playground.py", "Neuron playground", ":material/tune:"),
        ("2 · Logic gate / linear model", "With the right weights a neuron is **AND / OR / "
         "NAND**, or logistic regression. Gates wire up into arithmetic.",
         "views/neurons_compute.py", "Neurons → computer", ":material/calculate:"),
        ("3 · A layer (many neurons)", "Neurons in parallel build **features**; **two** "
         "already break the XOR wall and can add numbers.",
         "views/two_neurons.py", "Two neurons", ":material/hub:"),
        ("4 · MLP (a deep network)", "Stacked layers + a nonlinearity = a **universal "
         "function approximator**, trained by backprop. *Example: pixels → 'cat'.*",
         "views/mlp.py", "MLP (train it)", ":material/network_node:"),
        ("5 · Transformer", "**Attention** lets every token gather context — the architecture "
         "behind modern AI. *Example: translate a sentence.*",
         "views/attention.py", "Attention", ":material/auto_awesome:"),
        ("6 · GPT / LLM", "Billions of these neurons in Transformer blocks predicting the "
         "**next token**. *Example: a chatbot.*", "views/transformer.py", "Tiny GPT", ":material/smart_toy:"),
    ]
    for title, desc, path, label, icon in STOPS:
        c = st.columns([0.66, 0.34])
        c[0].markdown(f"**{title}** — {desc}")
        c[1].page_link(path, label=label, icon=icon)

# --- Scale ----------------------------------------------------------------- #
with st.container(border=True):
    st.markdown("#### 🔁 It's the same operation — just bigger")
    st.markdown(
        "Nothing fundamentally new happens as you go right along the chain. Every box is a "
        "**weighted sum (a matrix multiply) followed by a nonlinearity** — exactly what runs "
        "on a GPU. **Depth** means doing it many times, each layer composing richer features "
        "from the last; **scale** means doing it *a lot*."
    )
    st.markdown(_SCALE_SVG, unsafe_allow_html=True)
    st.caption("Same unit, repeated — from 1 neuron, to a layer, to ~100 billion in a frontier LLM. "
               "And like a brain it's **robust**: no single weight is critical, because every "
               "decision rests on a sum of many.")

# --- Two paradigms of intelligence (Hinton) -------------------------------- #
with st.container(border=True):
    st.markdown("#### 🧠 Two paradigms of intelligence — Geoffrey Hinton")
    st.caption("Hinton frames the history of AI as two bets on what intelligence *is*.")
    st.markdown(
        """<div style="text-align:center;margin:0.3rem 0"><svg viewBox="0 0 760 250" style="width:100%;max-width:760px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The two paradigms of intelligence (Geoffrey Hinton). The logic-inspired approach: intelligence is reasoning via symbolic rules, and learning can wait. The biologically-inspired approach: intelligence is learning the connection strengths in a neural network, and reasoning can wait. This lab follows the biologically-inspired approach."><rect x="1" y="1" width="758" height="248" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="380" y="30" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="700" fill="#33312E">Two paradigms of intelligence</text><rect x="24" y="48" width="344" height="188" rx="10" fill="#EAF0FB" stroke="#C7D6EE"/><rect x="392" y="48" width="344" height="188" rx="10" fill="#E7F3EC" stroke="#BFE0CC"/><g font-family="sans-serif"><text x="44" y="80" font-size="15" font-weight="700" fill="#2C5CBF">The logic-inspired approach</text><text x="44" y="106" font-size="12.5" fill="#33312E">The essence of intelligence is reasoning —</text><text x="44" y="124" font-size="12.5" fill="#33312E">symbolic rules manipulating symbolic expressions.</text><text x="44" y="156" font-size="12" font-style="italic" fill="#178A4C">“Learning can wait. First understand how</text><text x="44" y="174" font-size="12" font-style="italic" fill="#178A4C">knowledge is represented.”</text><text x="44" y="214" font-size="10.5" fill="#6B6A66">symbolic AI · logic · expert systems</text><text x="412" y="80" font-size="15" font-weight="700" fill="#2C5CBF">The biologically-inspired approach</text><text x="412" y="106" font-size="12.5" fill="#33312E">The essence of intelligence is learning the</text><text x="412" y="124" font-size="12.5" fill="#33312E">strengths of the connections in a neural network.</text><text x="412" y="156" font-size="12" font-style="italic" fill="#178A4C">“Reasoning can wait. First understand how</text><text x="412" y="174" font-size="12" font-style="italic" fill="#178A4C">learning works.”</text><text x="412" y="214" font-size="10.5" font-weight="700" fill="#178A4C">connectionism · deep learning · ← this lab</text></g></svg></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "For decades the **logic-inspired** paradigm led — hand-built symbolic rules "
        "(expert systems, logic, search). This lab takes the **biologically-inspired** bet: "
        "intelligence as **learning the connection strengths (the weights)** of a network. "
        "Every weight you train here — single neuron → GPT — *is* one of those connections. "
        "The modern twist: **reasoning is emerging *from* learning** (chain-of-thought in LLMs), "
        "so the two paradigms are converging — which is exactly the *two views* below."
    )

# --- Two views ------------------------------------------------------------- #
with st.container(border=True):
    st.markdown("#### 🔭 Two views that meet")
    st.markdown(_VIEWS_SVG, unsafe_allow_html=True)
    st.markdown(
        "- **Logic view:** neuron → gate → **circuit** → (with memory) a **universal "
        "computer**. NAND-completeness makes this exact — see *Neurons → computer*.\n"
        "- **Function view:** neuron → linear model → **MLP** → fits **any function** → "
        "specialised to images and text.\n\n"
        "A **trained network is both at once** — soft, *learned* logic gates **and** a "
        "function approximator. The 'learning' is just gradient descent (Math **X4**) finding "
        "the weights, and backprop (Math **X2**) supplying the gradients."
    )

# --- Where it shows up ------------------------------------------------------ #
with st.container(border=True):
    st.markdown("#### 🌍 Where it shows up — it's all the same recipe")
    st.markdown(
        "| Application | the network maps… |\n|---|---|\n"
        "| Spam filter | email words → *spam?* |\n"
        "| Image recognition | pixels → *'cat'* |\n"
        "| Recommendations | your history → *what's next* |\n"
        "| Translation | English → French |\n"
        "| Chatbot / LLM | the text so far → *the next word* |\n\n"
        "Different data, same machine: numbers in → weighted sums + nonlinearities → numbers out, "
        "with the weights **learned from examples**."
    )

# --- Start the journey ----------------------------------------------------- #
with st.container(border=True):
    st.markdown("#### 🚩 Start the journey")
    st.markdown("Follow the five levels in the sidebar, or jump in here:")
    c = st.columns(4)
    c[0].page_link("views/playground.py", label="Meet a neuron", icon=":material/tune:")
    c[1].page_link("views/mlp.py", label="Train a network", icon=":material/network_node:")
    c[2].page_link("views/attention.py", label="Attention", icon=":material/auto_awesome:")
    c[3].page_link("views/transformer.py", label="Build a GPT", icon=":material/smart_toy:")

st.info("Where you are: stops **1–6 are all built and runnable** — a single neuron, an MLP that "
        "learns XOR live, attention, a tiny GPT, and a real PyTorch nanoGPT (e21). "
        "Everything on this map is something you can open and play with.", icon=":material/flag:")
