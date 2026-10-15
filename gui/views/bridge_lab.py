"""Inspect the bridge to a Transformer without training a large model on the server."""
import numpy as np
import pandas as pd
import streamlit as st
from gui.practice import prediction_gate, save_result, hints
from core.attention import attention, projections, multihead, transformer_block, block_weights

st.title("Build the bridges")
st.caption("Goal: trace shapes and information flow. These are seeded forward-pass demonstrations, not trained language models. Guide updated 2026-09-20.")
st.page_link("views/attention.py", label="Prerequisite: attention and Q/K/V")
st.markdown("### 1 · A single attention head")
seed = st.number_input("Seed", 0, 1000, 7)
length = st.slider("Tokens", 2, 8, 4)
heads = st.select_slider("Heads", [1,2,4,8], value=2)
record = prediction_gate("bridge", title="Causal Transformer block", lesson="attention", seed=seed,
    settings={"tokens": length, "width": 8, "heads": heads},
    prompt="If only the last token changes, which earlier outputs can change? Predict the attention-map shape too.")
width = 8
if record:
    X = np.random.default_rng(seed).normal(size=(length,width))
    w = projections(width, seed)
    out, amap = attention(X @ w['q'], X @ w['k'], X @ w['v'])
    st.write(f"X: {X.shape} → Q, K, V: {X.shape} → attention weights: {amap.shape} → output: {out.shape}")
    st.dataframe(pd.DataFrame(amap).round(3), width="stretch")
    st.caption("Rows sum to one; weights above the diagonal are zero because future tokens are masked.")
    st.markdown("### 2 · Multiple heads")
    y, maps = multihead(X, w, heads)
    head = st.selectbox("Inspect head", range(heads))
    st.dataframe(pd.DataFrame(maps[head]).round(3), width="stretch")
    st.write(f"Each head: width {width//heads}. Concatenation restores width {width}; output projection keeps {y.shape}.")
    st.markdown("### 3 · Add normalization, residuals and a feed-forward layer")
    result, details = transformer_block(X, block_weights(width, seed), heads)
    st.dataframe(pd.DataFrame({"input norm":np.linalg.norm(X,axis=1),
        "after attention residual":np.linalg.norm(details['residual'],axis=1),
        "after feed-forward residual":np.linalg.norm(result,axis=1)}), width="stretch")
    perturbed = X.copy(); perturbed[-1] += 10
    changed, _ = transformer_block(perturbed, block_weights(width,seed),heads)
    st.metric("Earlier-token change when only the last token changes", f"{np.max(np.abs(changed[:-1]-result[:-1])):.2e}")
    st.info("This must be zero for causal attention: changing the future cannot alter earlier outputs. Test that invariant before training.")
    hints("bridge", ["Inspect the triangle removed by the causal mask.", "Each head has a tokens × tokens map; its value width is model width / heads."])
    save_result(record, f"head map={amap.shape}; heads={heads}; future perturbation difference={np.max(np.abs(changed[:-1]-result[:-1])):.2e}")
st.markdown("### Rebuild it locally")
st.code("python experiments/tier4_attention/e18_attention/run.py\npython experiments/tier4_attention/e19_multihead/run.py\npython experiments/tier4_attention/e20_transformer_block/run.py")
st.markdown("### From NumPy to PyTorch on MNIST")
st.write("e12 and e13 use the same data, initial weights, minibatch order, precision and SGD updates. Compare the loss curves and parameter differences before changing the framework or optimizer.")
st.code("pip install -r requirements-dl.txt\npython experiments/tier2_training/e12_mnist/run.py --download\npython experiments/tier2_training/e13_pytorch/run.py --download")
st.caption("MNIST downloads only with --download; the public app does not fetch datasets or launch these training jobs. The full Transformer training experiment is e21.")
st.markdown("Sources: [Attention Is All You Need](https://arxiv.org/abs/1706.03762); [MNIST](https://yann.lecun.com/exdb/mnist/).")

for experiment in ("e12", "e13", "e18", "e19", "e20", "e21"):
    st.link_button(f"Open {experiment}: code and runnable exercise", f"/experiments?experiment={experiment}")
