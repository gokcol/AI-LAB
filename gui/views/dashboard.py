import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import i18n
import lab
import ui

LANG = i18n.lang()

COPY = {
    "en": {
        "hero_title": "From a single neuron to a small GPT",
        "hero_body": "Train the fundamentals by rebuilding them: run the experiment, explain the result, derive the formula, then break it on purpose.",
        "roadmap_title": "Learning roadmap — a single neuron → a GPT",
        "roadmap_intro": (
            "Five levels, **basics to frontier** — work top to bottom; each builds on the last and "
            "**clicking any step opens it**.\n\n"
            "The lab has **three tracks** (switch with the sidebar **Module** selector): **ANN** — "
            "this path, a single neuron → a GPT · **ML** — classical machine learning (M0–M8) · "
            "**Math** — the foundations (X1–X6) you *dip into as needed*. Each level flags the math "
            "it leans on."
        ),
        "big_picture": "Start with the big picture — how it all connects",
        "math_caption": "Math you'll lean on:",
        "math_suffix": "(Math module in the sidebar)",
        "status_title": "Experiment build status",
        "status_caption": "Live status of the code experiments under `experiments/`. Heavy training demos are skipped by default.",
        "include_heavy": "Include heavy training demos",
        "include_heavy_help": "Runs slower experiments such as e21 nanoGPT. Leave off for a quick dashboard scan.",
        "rescan": "Re-scan",
        "metric_experiments": "Experiments",
        "metric_done": "Done",
        "metric_deferred": "Deferred",
        "metric_remaining": "Remaining",
        "view_roadmap": "View ROADMAP.md",
        "journey_title": "Training journey — see the same loop grow",
        "journey_body": "Follow one idea across the lab: fit a line, classify a machine, train an XOR network, then predict the next character. Each stop exposes forward pass → loss → gradients → update.",
        "journey_regression": "1 · Train a line",
        "journey_classification": "2 · Train a classifier",
        "journey_network": "3 · Train a network",
        "journey_text": "4 · Train a text model",
        "levels": [
            (
                "Level 1 · The neuron — foundations",
                "A neuron is just a **weighted sum + a nonlinearity** — one decision line. Meet it, watch **two** neurons already beat XOR, and see how neurons become **logic gates** and even do arithmetic. This is the atom everything else is built from.",
                [("views/playground.py", "Neuron playground", ":material/tune:"),
                 ("views/two_neurons.py", "Two neurons", ":material/hub:"),
                 ("views/neurons_compute.py", "Neurons → computer", ":material/calculate:")],
                "**X1** vectors & the dot product — a neuron *is* a dot product.",
            ),
            (
                "Level 2 · How a network learns — training",
                "The engine of deep learning: **backprop** finds a gradient for every weight, an **optimizer** steps downhill, and **regularization** stops it memorizing noise. Train your first network — it learns XOR live — and watch overfitting appear and get tamed.",
                [("views/backprop.py", "Backprop", ":material/sync_alt:"),
                 ("views/mlp.py", "MLP (train it)", ":material/network_node:"),
                 ("views/optimizers.py", "Optimizers", ":material/trending_down:"),
                 ("views/deep_playground.py", "Deep nets (2D)", ":material/blur_on:"),
                 ("views/regularization.py", "Regularization", ":material/tune:")],
                "**X2** calculus & the chain rule (backprop) · **X4** optimization (the optimizers).",
            ),
            (
                "Level 3 · Architectures — images & sequences",
                "Wiring matched to the data: **convolutions** share a small filter across an image (**CNN**); **recurrence** carries a hidden state across a sequence (**RNN**). The RNN's memory limits are exactly what motivates attention next.",
                [("views/cnn.py", "CNN (images)", ":material/image:"),
                 ("views/rnn.py", "RNN (sequences)", ":material/repeat:")],
                "**X1** dot products (convolution) · **X2** gradients through time (BPTT).",
            ),
            (
                "Level 4 · Attention & Transformers",
                "The leap to modern AI: text becomes **tokens**, **self-attention** lets every token gather context by dot-product + softmax, and a stack of attention blocks predicts the **next token** — a tiny **GPT** you can run right here.",
                [("views/tokenization.py", "Tokenization", ":material/content_cut:"),
                 ("views/attention.py", "Attention (LLMs)", ":material/auto_awesome:"),
                 ("views/transformer.py", "Tiny GPT", ":material/smart_toy:")],
                "**X1** dot product = similarity · **X5** softmax.",
            ),
            (
                "Level 5 · LLMs in practice",
                "From a base model to a useful assistant: **decoding** turns probabilities into text, **embeddings + RAG** ground answers in facts, and **fine-tuning + RLHF** align behaviour. Then train the *real* thing — the **e21 nanoGPT** — from the Experiments page.",
                [("views/sampling.py", "Decoding (sampling)", ":material/casino:"),
                 ("views/embeddings.py", "Embeddings & RAG", ":material/database:"),
                 ("views/posttraining.py", "Post-training (RLHF)", ":material/psychology:"),
                 ("views/experiments.py", "Experiments (e21 nanoGPT)", ":material/science:")],
                "**X5** cross-entropy & KL (training, decoding, alignment) · **X1** cosine similarity (RAG).",
            ),
        ],
    },
    "tr": {
        "hero_title": "Tek nörondan küçük bir GPT'ye",
        "hero_body": "Temelleri yeniden kurarak öğren: deneyi çalıştır, sonucu açıkla, formülü türet, sonra bilerek bir şeyi boz.",
        "roadmap_title": "Öğrenme haritası — tek nörondan GPT'ye",
        "roadmap_intro": (
            "Beş seviye, **temelden sınıra** — yukarıdan aşağı ilerle; her adım bir öncekini kullanır ve "
            "**her adıma tıklayarak açabilirsin**.\n\n"
            "Laboratuvarda **üç kanal** var (sidebar'daki **Modül** seçiciyle değiştir): **ANN** — "
            "tek nörondan GPT'ye bu yol · **ML** — klasik makine öğrenmesi (M0-M8) · "
            "**Math** — ihtiyaç oldukça döneceğin matematik temelleri (X1-X6). Her seviye dayandığı matematiği gösterir."
        ),
        "big_picture": "Büyük resimle başla — her şey nasıl bağlanıyor",
        "math_caption": "Dayanacağı matematik:",
        "math_suffix": "(sidebar'daki Math modülü)",
        "status_title": "Deney çalışma durumu",
        "status_caption": "`experiments/` altındaki kod deneylerinin canlı durumu. Ağır eğitim demoları varsayılan olarak ertelenir.",
        "include_heavy": "Ağır eğitim demolarını dahil et",
        "include_heavy_help": "e21 nanoGPT gibi yavaş deneyleri çalıştırır. Hızlı tarama için kapalı bırak.",
        "rescan": "Yeniden tara",
        "metric_experiments": "Deney",
        "metric_done": "Tamam",
        "metric_deferred": "Ertelenen",
        "metric_remaining": "Kalan",
        "view_roadmap": "ROADMAP.md dosyasını göster",
        "journey_title": "Eğitim yolculuğu — aynı döngünün büyümesini gör",
        "journey_body": "Laboratuvar boyunca tek bir fikri izle: bir doğru uydur, bir makineyi sınıflandır, XOR ağı eğit, sonra sonraki karakteri tahmin et. Her durak ileri geçiş → loss → gradyan → güncellemeyi açar.",
        "journey_regression": "1 · Bir doğruyu eğit",
        "journey_classification": "2 · Sınıflandırıcıyı eğit",
        "journey_network": "3 · Bir ağı eğit",
        "journey_text": "4 · Bir metin modelini eğit",
        "levels": [
            (
                "Seviye 1 · Nöron — temeller",
                "Bir nöron yalnızca **ağırlıklı toplam + doğrusal olmayan aktivasyon** hesaplar: tek bir karar çizgisi. Onu tanı, **iki** nöronun XOR'u nasıl aşabildiğini gör, nöronların **mantık kapılarına** ve hatta aritmetiğe nasıl dönüştüğünü izle. Her şeyin atomu burası.",
                [("views/playground.py", "Nöron oyun alanı", ":material/tune:"),
                 ("views/two_neurons.py", "İki nöron", ":material/hub:"),
                 ("views/neurons_compute.py", "Nöronlar → bilgisayar", ":material/calculate:")],
                "**X1** vektörler ve dot product — bir nöron zaten dot product'tır.",
            ),
            (
                "Seviye 2 · Ağ nasıl öğrenir — eğitim",
                "Derin öğrenmenin motoru: **backprop** her ağırlık için gradyan bulur, **optimizer** aşağı doğru adım atar, **regularization** ise modelin gürültüyü ezberlemesini engeller. İlk ağını eğit — XOR'u canlı öğrenir — ve overfitting'in nasıl oluşup dizginlendiğini gör.",
                [("views/backprop.py", "Backprop", ":material/sync_alt:"),
                 ("views/mlp.py", "MLP (eğit)", ":material/network_node:"),
                 ("views/optimizers.py", "Optimizer'lar", ":material/trending_down:"),
                 ("views/deep_playground.py", "Derin ağlar (2D)", ":material/blur_on:"),
                 ("views/regularization.py", "Regularizasyon", ":material/tune:")],
                "**X2** kalkülüs ve zincir kuralı (backprop) · **X4** optimizasyon.",
            ),
            (
                "Seviye 3 · Mimariler — görüntüler ve diziler",
                "Veriye göre kablolama: **convolution** küçük bir filtreyi görüntü boyunca paylaşır (**CNN**); **recurrence** dizi boyunca gizli durum taşır (**RNN**). RNN'in bellek sınırları attention'a geçişi motive eder.",
                [("views/cnn.py", "CNN (görüntüler)", ":material/image:"),
                 ("views/rnn.py", "RNN (diziler)", ":material/repeat:")],
                "**X1** dot product (convolution) · **X2** zamanda geriye gradyanlar (BPTT).",
            ),
            (
                "Seviye 4 · Attention ve Transformer",
                "Modern AI sıçraması: metin **token** olur, **self-attention** her token'ın dot-product + softmax ile bağlam toplamasını sağlar, attention blokları da **bir sonraki token'ı** tahmin eder — burada çalıştırabileceğin küçük bir **GPT**.",
                [("views/tokenization.py", "Tokenizasyon", ":material/content_cut:"),
                 ("views/attention.py", "Attention (LLM'ler)", ":material/auto_awesome:"),
                 ("views/transformer.py", "Küçük GPT", ":material/smart_toy:")],
                "**X1** dot product = benzerlik · **X5** softmax.",
            ),
            (
                "Seviye 5 · Pratikte LLM'ler",
                "Bir base modelden kullanışlı asistana: **decoding** olasılıkları metne çevirir, **embedding + RAG** cevapları bilgiye bağlar, **fine-tuning + RLHF** davranışı hizalar. Sonra gerçek versiyonu, **e21 nanoGPT** deneyini çalıştır.",
                [("views/sampling.py", "Decoding (sampling)", ":material/casino:"),
                 ("views/embeddings.py", "Embedding ve RAG", ":material/database:"),
                 ("views/posttraining.py", "Post-training (RLHF)", ":material/psychology:"),
                 ("views/experiments.py", "Deneyler (e21 nanoGPT)", ":material/science:")],
                "**X5** cross-entropy ve KL · **X1** cosine similarity (RAG).",
            ),
        ],
    },
}

C = COPY.get(LANG, COPY["en"])

st.title("🧠 AI Lab")
ui.hero(C["hero_title"], C["hero_body"])

# --------------------------------------------------------------------------- #
# Learning roadmap — five levels, basics → frontier, with links to each page
# --------------------------------------------------------------------------- #
st.subheader(f"📚 {C['roadmap_title']}")
st.markdown(C["roadmap_intro"])
st.page_link("views/the_chain.py", label=C["big_picture"], icon=":material/route:")

for title, intro, items, math in C["levels"]:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.markdown(intro)
        # lay the page links out in rows of up to 3
        for i in range(0, len(items), 3):
            row = items[i:i + 3]
            cols = st.columns(3)
            for col, (path, label, icon) in zip(cols, row):
                col.page_link(path, label=label, icon=icon)
        st.caption(f"➕ **{C['math_caption']}** {math}  *{C['math_suffix']}*")

st.subheader(f"🏋 {C['journey_title']}")
st.markdown(C["journey_body"])


def _open_training_track(module: str, path: str):
    st.session_state["module"] = module
    st.session_state["_goto_page"] = path
    st.rerun()


journey = st.columns(4)
if journey[0].button(C["journey_regression"], key="journey_regression", icon=":material/timeline:"):
    _open_training_track("ML", "views/ml_regression.py")
if journey[1].button(C["journey_classification"], key="journey_classification", icon=":material/scatter_plot:"):
    _open_training_track("ML", "views/ml_classification.py")
journey[2].page_link("views/mlp.py", label=C["journey_network"], icon=":material/network_node:")
journey[3].page_link("views/transformer.py", label=C["journey_text"], icon=":material/smart_toy:")

st.divider()

# --------------------------------------------------------------------------- #
# Build status — runs each experiment's run.py (done / todo / error)
# --------------------------------------------------------------------------- #
st.subheader(f"🔬 {C['status_title']}")
st.caption(C["status_caption"])


@st.cache_data(show_spinner="Scanning experiments…")
def scan(include_heavy: bool):
    """Run each experiment once; status = done / todo / error / heavy."""
    out = {}
    for tier, exps in lab.list_tiers().items():
        out[tier] = [(e, lab.status_of(e, include_heavy=include_heavy)) for e in exps]
    return out


ICON = {"done": "🟢", "todo": "⚪", "error": "🔴", "heavy": "⏱️"}

top = st.container()
controls = st.columns([0.7, 0.3])
include_heavy = controls[0].checkbox(
    C["include_heavy"],
    value=False,
    help=C["include_heavy_help"],
    key="dashboard_include_heavy",
)
if controls[1].button(C["rescan"], icon=":material/refresh:", key="dashboard_rescan"):
    scan.clear()

data = scan(include_heavy)
flat = [s for rows in data.values() for _, s in rows]
done = flat.count("done")
heavy = flat.count("heavy")

with top:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(C["metric_experiments"], len(flat))
    c2.metric(C["metric_done"], done)
    c3.metric(C["metric_deferred"], heavy)
    c4.metric(C["metric_remaining"], len(flat) - done - heavy)
    st.progress(done / len(flat) if flat else 0.0)

for tier, rows in data.items():
    t_done = sum(1 for _, s in rows if s == "done")
    st.subheader(f"{lab.tier_label(tier)}  ·  {t_done}/{len(rows)}")
    for exp, status in rows:
        cols = st.columns([0.08, 0.5, 0.42])
        cols[0].write(ICON[status])
        cols[1].write(f"**{exp.id}** {exp.name}")
        cols[2].caption(status)

with st.expander(C["view_roadmap"]):
    st.markdown(lab.read_localized(lab.LAB_ROOT / "ROADMAP.md", LANG))
