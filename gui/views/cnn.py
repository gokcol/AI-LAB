"""CNN — convolutional networks for images (ANN module, roadmap e14 / Tier 3).

The architecture that made deep learning work on images. The Live tab slides real
convolution kernels over a small image so you can *see* a filter become an edge/blur/
sharpen detector (a "feature map"), then max-pool it — the two operations a CNN stacks.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.signal import correlate2d

import lessons


def _image():
    img = np.full((32, 32), 0.12)
    yy, xx = np.mgrid[0:32, 0:32]
    img[(xx - 9) ** 2 + (yy - 9) ** 2 < 34] = 0.95          # filled circle
    img[18:29, 4:15] = 0.65                                  # filled square
    for i in range(32):                                      # diagonal stripe
        if 0 <= 30 - i < 32:
            img[i, 30 - i] = 1.0
    img[3:5, 18:30] = 0.8                                    # a couple of bars
    return img


KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "blur (box)": np.ones((3, 3)) / 9,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "edge — vertical (Sobel x)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "edge — horizontal (Sobel y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "edges (Laplacian)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
}


def _maxpool(a, k=2):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).max(axis=(1, 3))


_ARCH_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 720 180" style="width:100%;max-width:720px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A CNN: an input image goes through repeated convolution, ReLU and pooling blocks (spatial size shrinks, the number of channels grows), then is flattened into fully-connected layers ending in a softmax over classes."><defs><marker id="cnv" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/></marker></defs><rect x="1" y="1" width="718" height="178" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g font-family="sans-serif" font-size="11" text-anchor="middle"><rect x="22" y="52" width="74" height="74" fill="#E6F1FB" stroke="#5B8FC2"/><text x="59" y="142" fill="#6B6A66">image</text><rect x="130" y="58" width="58" height="62" fill="#FBEAD6" stroke="#9A6A2A"/><rect x="138" y="64" width="58" height="62" fill="#FBEAD6" stroke="#9A6A2A"/><text x="167" y="142" fill="#5A3E14">conv+ReLU</text><rect x="222" y="66" width="42" height="46" fill="#FBEAD6" stroke="#9A6A2A"/><text x="243" y="142" fill="#6B6A66">pool</text><rect x="300" y="62" width="34" height="40" fill="#FBEAD6" stroke="#9A6A2A"/><rect x="306" y="66" width="34" height="40" fill="#FBEAD6" stroke="#9A6A2A"/><rect x="312" y="70" width="34" height="40" fill="#FBEAD6" stroke="#9A6A2A"/><text x="330" y="142" fill="#5A3E14">conv → pool</text><rect x="404" y="50" width="14" height="80" fill="#E6F1FB" stroke="#5B8FC2"/><text x="411" y="142" fill="#6B6A66">flatten</text><rect x="470" y="58" width="70" height="64" rx="6" fill="#E6F1FB" stroke="#5B8FC2"/><text x="505" y="94" fill="#0C447C">fully-conn.</text><rect x="566" y="62" width="66" height="56" rx="6" fill="#D7EFE5" stroke="#1D9E75"/><text x="599" y="86" fill="#0E5E45">softmax</text><text x="599" y="100" fill="#0E5E45" font-size="9">classes</text></g><g stroke="#5B8FC2" stroke-width="1.7" fill="none"><line x1="96" y1="89" x2="128" y2="89" marker-end="url(#cnv)"/><line x1="196" y1="89" x2="220" y2="89" marker-end="url(#cnv)"/><line x1="264" y1="89" x2="298" y2="89" marker-end="url(#cnv)"/><line x1="348" y1="89" x2="402" y2="89" marker-end="url(#cnv)"/><line x1="418" y1="89" x2="468" y2="89" marker-end="url(#cnv)"/><line x1="540" y1="89" x2="564" y2="89" marker-end="url(#cnv)"/></g><text x="360" y="166" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9C9B95">spatial size shrinks ↓, channels grow → ; early layers = edges, deep layers = objects</text></svg></div>'''


_THEORY = r"""
## 1. Why not just an MLP?

Flatten a $200\times200$ RGB image and you have **120,000** inputs; one fully-connected
hidden layer of 1,000 units is then **120 million** weights — for *one* layer. Worse, an MLP
treats every pixel independently, so it has **no idea that nearby pixels are related** and
isn't **translation-invariant**: a cat in the top-left looks, to an MLP, nothing like the
same cat in the center. Images need an architecture that respects their 2-D structure.

## 2. Convolution — a sliding filter

A **convolutional layer** slides a tiny **kernel** (e.g. $3\times3$) across the image,
computing a **dot product** at each position (Math X1). The result is a **feature map** that
lights up wherever the kernel's pattern occurs — an edge, a corner, a colour blob. Two ideas
do the heavy lifting:

- **Local receptive field** — each output looks at only a small patch, matching how image
  structure is local.
- **Weight sharing** — the *same* kernel is used at every position. That's a massive
  parameter saving **and** gives **translation equivariance**: detect the pattern anywhere.

The Live tab is exactly this — pick a kernel and watch the feature map.

## 3. Feature maps & channels

One kernel → one feature map. A conv layer has **many** kernels → many output **channels**
(a vertical-edge map, a horizontal-edge map, a blob map, …). Stack conv layers and they
compose: edges → textures → parts → objects — features the network **learns**, not ones you
hand-design (the demo's kernels are fixed only to make the idea visible).

## 4. Pooling

A **pooling** layer downsamples each feature map — **max-pooling** keeps the strongest
response in each $2\times2$ block. It shrinks the spatial size (less compute), adds a little
**translation invariance** (small shifts don't change the max much), and lets deeper layers
see a larger effective region. Try the pool toggle in the demo.

## 5. The architecture (LeNet → modern)

A classic CNN stacks **[conv → ReLU → pool]** blocks, then **flattens** and finishes with
fully-connected layers + a **softmax** over classes:

<ARCH/>

As you go deeper the maps get **spatially smaller but deeper** (more channels) — trading
"where" for "what." LeNet (1998) did digits; AlexNet (2012) ignited deep learning on
ImageNet; ResNet added the **residual connections** you met in the Transformer.

## 6. Why it works

Convolution bakes in the right **priors** for images — locality and translation
equivariance — so a CNN needs **far fewer parameters** than an MLP and generalizes from
fewer examples. It's still trained the same way: forward → loss → **backprop** → optimizer
step; the kernels are just weights, learned by gradient descent.

## 7. Beyond images

The same idea — a small shared filter sweeping over structured input — also works on
**audio** and **time series** (1-D convolutions). And **Vision Transformers** now rival CNNs
by cutting an image into patches and running **attention** over them (Attention page) — the
two big architectures meeting. *(Roadmap e14; the real version is a LeNet/CNN trained in
PyTorch.)*
"""

_QUIZ = [
    lessons.Question(
        "Why are CNNs preferred over MLPs for images?",
        ["they have more parameters", "weight-sharing + locality slash parameters and give translation invariance, respecting 2-D structure",
         "they don't need training", "MLPs can't use ReLU"], 1,
        "A shared small kernel detects a pattern anywhere with few weights — an MLP would need a separate weight per pixel and isn't shift-invariant."),
    lessons.Question(
        "A convolution kernel produces a 'feature map' that…",
        ["lists the image's pixels", "lights up where the kernel's pattern (e.g. an edge) appears in the image",
         "is always smaller by half", "removes all edges"], 1,
        "Sliding the kernel and taking dot products highlights every location matching its pattern."),
    lessons.Question(
        "Max-pooling does what?",
        ["adds parameters", "downsamples each feature map (keeps the max per block), adding invariance and cutting compute",
         "applies softmax", "increases the image size"], 1,
        "Pooling shrinks spatial size, tolerates small shifts, and lets deeper layers see more context."),
    lessons.Question(
        "Going deeper in a CNN, feature maps typically become…",
        ["spatially larger, fewer channels", "spatially smaller, more channels (where → what)",
         "unchanged", "always 1×1 immediately"], 1,
        "Pooling/striding shrink the spatial grid while conv layers add channels — trading location for richer features."),
]

_TASKS = r"""
### In the Convolve tab
1. Apply **edge — vertical** vs **edge — horizontal** — which strokes of the shapes light up
   in each, and why? (The kernel responds to *that* orientation of edge.)
2. Try **blur** then **sharpen** — opposite effects; look at the kernel values to see why.
3. Turn on **max-pool** after an edge filter — the map halves in size but the strong edges
   survive. That's the invariance/compression a CNN relies on.

### Concept
4. Estimate the parameters in a conv layer with 16 kernels of size 3×3 over a 1-channel
   input (incl. biases). Compare to a fully-connected layer over a 32×32 image.
5. Why does weight-sharing give *translation equivariance*? What does pooling add on top?
"""

_REFS = r"""
- LeCun et al. (1998) — *LeNet* (CNNs for digit recognition).
- Krizhevsky et al. (2012) — *AlexNet*; He et al. (2015) — *ResNet*.
- Stanford **CS231n** — Convolutional Neural Networks for Visual Recognition.
- Dosovitskiy et al. (2020) — *Vision Transformer* (attention for images).
- In this lab: **Backprop** / **Optimizers** (training), Math **X1** (the dot product),
  **Attention** (ViT connection).
"""


st.title("CNN — convolutional networks for images")
st.caption("Slide real kernels over an image and watch each become an edge / blur / sharpen "
           "detector (a feature map), then pool it — the two ops a CNN stacks.")

lessons.predict(
    'Pick the **vertical-edge** kernel and slide it over the image — *where* will the feature map light up (large values)?',
    'Right on the **vertical edges** — where dark meets light along a column. A convolution scores how much each patch resembles the kernel; a CNN learns *many* kernels, so early layers fire on edges, deeper ones on textures and object parts.',
)

tab_live, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🖼 Convolve", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_live:
    img = _image()
    cc = st.columns([0.6, 0.4])
    kname = cc[0].selectbox("kernel (filter)", list(KERNELS), index=3, key="cnn_k")
    pool = cc[1].checkbox("max-pool 2×2 the result", key="cnn_pool")
    K = KERNELS[kname]
    fmap = correlate2d(img, K, mode="same", boundary="symm")
    shown = _maxpool(fmap) if pool else fmap

    g = st.columns(3)
    g[0].metric("image", f"{img.shape[0]}×{img.shape[1]}")
    g[1].metric("kernel", "3×3")
    g[2].metric("feature map", f"{shown.shape[0]}×{shown.shape[1]}")

    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.3))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("input", fontsize=10)
    ax[1].imshow(shown, cmap="gray"); ax[1].set_title(f"after {kname}" + (" + pool" if pool else ""), fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    st.pyplot(fig, width="stretch")

    st.caption("The 3×3 kernel applied (each output = this filter · the patch under it):")
    st.dataframe(K.astype(float), hide_index=True, width="content")
    st.info("This is one convolution with one fixed kernel. A CNN stacks **many learned** "
            "kernels per layer (channels), interleaves **pooling**, and trains the kernels by "
            "backprop — building edges → textures → objects.", icon=":material/filter_center_focus:")

    with st.expander("🐦 Hinton's example: how edge detection builds up to \"bird\""):
        st.markdown(
            """<div style="text-align:center;margin:0.3rem 0"><svg viewBox="0 0 720 300" style="width:100%;max-width:720px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A feature hierarchy for recognizing a bird: an edge-detection layer feeds part detectors beak and eye, which feed an object-part detector bird's head, which votes among the output classes cat, dog, bird and politician — bird wins."><defs><marker id="cnh" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/></marker></defs><rect x="1" y="1" width="718" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g stroke="#5B8FC2" stroke-width="2" fill="none"><line x1="130" y1="126" x2="270" y2="90" marker-end="url(#cnh)"/><line x1="130" y1="174" x2="270" y2="210" marker-end="url(#cnh)"/><line x1="332" y1="90" x2="460" y2="136" marker-end="url(#cnh)"/><line x1="332" y1="210" x2="460" y2="164" marker-end="url(#cnh)"/><line x1="540" y1="136" x2="640" y2="56" marker-end="url(#cnh)"/><line x1="543" y1="147" x2="639" y2="110" marker-end="url(#cnh)"/><line x1="543" y1="153" x2="639" y2="174" marker-end="url(#cnh)"/><line x1="540" y1="164" x2="640" y2="236" marker-end="url(#cnh)"/></g><g font-family="sans-serif" text-anchor="middle" font-weight="600" fill="#FFFFFF"><circle cx="86" cy="150" r="46" fill="#8FBF4A" stroke="#5E8A2E" stroke-width="2"/><text x="86" y="146" font-size="11">Edge</text><text x="86" y="162" font-size="11">Detection</text><circle cx="300" cy="82" r="34" fill="#8FBF4A" stroke="#5E8A2E" stroke-width="2"/><text x="300" y="87" font-size="12">Beak</text><circle cx="300" cy="218" r="34" fill="#8FBF4A" stroke="#5E8A2E" stroke-width="2"/><text x="300" y="223" font-size="12">Eye</text><circle cx="500" cy="150" r="44" fill="#8FBF4A" stroke="#5E8A2E" stroke-width="2"/><text x="500" y="147" font-size="11.5">Bird's</text><text x="500" y="163" font-size="11.5">Head</text><circle cx="666" cy="48" r="27" fill="#8FBF4A" stroke="#5E8A2E" stroke-width="2"/><text x="666" y="52" font-size="11">Cat</text><circle cx="666" cy="112" r="27" fill="#8FBF4A" stroke="#5E8A2E" stroke-width="2"/><text x="666" y="116" font-size="11">Dog</text><circle cx="666" cy="176" r="29" fill="#8FBF4A" stroke="#E0A32A" stroke-width="3.5"/><text x="666" y="180" font-size="12">Bird</text><circle cx="666" cy="244" r="27" fill="#8FBF4A" stroke="#5E8A2E" stroke-width="2"/><text x="666" y="248" font-size="8.5">Politician</text></g><text x="360" y="292" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#6B6A66">edges → parts → object part → class · each layer's features are built from the one below</text></svg></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            "The feature map you just made **is layer 1 — edge detection.** In a deep vision net "
            "(Hinton's *StarTalk* example) those edges combine into **part** detectors — a "
            "**beak**, an **eye** — which combine into an **object-part** detector — a **bird's "
            "head** — which finally votes for a **class** (*bird*, not cat/dog). Each layer "
            "detects features built from the layer below, and the whole hierarchy is **learned "
            "bottom-up by backprop**, never hand-designed — which is exactly why *depth* matters "
            "for vision."
        )

with tab_theory:
    st.markdown(_THEORY.replace("<ARCH/>", _ARCH_SVG), unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="cnn")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** The **vertical**-edge kernel lights up on vertical strokes (where intensity changes left↔right); the **horizontal** one on horizontal strokes. Each kernel correlates high only where the intensity change matches *its* orientation.

**2.** **Blur** is an all-positive averaging kernel (a low-pass filter that smooths); **sharpen** is a positive centre with a negative surround (a high-pass filter that amplifies differences). Opposite signs → opposite effects.

**3.** Max-pool keeps the strongest response in each 2×2 window: the map halves in size but the strong edges survive. That buys small-shift **invariance** and compression — the two things a CNN stacks between conv layers.""",
        label="Convolve tab 1–3",
    )
    lessons.solution(
        r"""**4.** Conv layer: $16\times(3\times3\times1) + 16 = 160$ params — and it's **independent of image size**. A fully-connected layer from a $32\times32=1024$ input to just 16 units already needs $1024\times16+16 = 16{,}400$, and it grows with the image. Weight-sharing is the huge saving.

**5.** The *same* kernel slides over every position, so shifting the input shifts the output — **translation equivariance**. **Pooling** then discards small position differences, adding a degree of translation *invariance* on top.""",
        label="Concept 4–5",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
