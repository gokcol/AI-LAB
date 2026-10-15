"""A complete bilingual first learning session: predict, calculate, change, explain."""
import numpy as np
import streamlit as st
from gui import i18n, progress

tr = i18n.lang() == "tr"
def text(en, turkish):
    return turkish if tr else en

st.title(text("Start here · one neuron in 20 minutes", "Buradan başla · 20 dakikada bir nöron"))
st.caption(text("Prerequisites: addition and multiplication. Goal: predict a neuron's output and explain its bias.",
                "Ön koşul: toplama ve çarpma. Hedef: nöron çıktısını tahmin etmek ve bias'ın etkisini açıklamak."))
if i18n.MULTILINGUAL:
    st.info(text("This guided introduction is available in English and Turkish. Turkish is an AI-generated preview, not a human-reviewed translation.",
                 "Bu rehber İngilizce ve Türkçe sunulur. Türkçe metin, insan tarafından gözden geçirilmemiş yapay zekâ çevirisidir."))
else:
    st.info("Human review: pending.")
st.markdown(text("### 1 · Predict (5 minutes)", "### 1 · Tahmin et (5 dakika)"))
st.write(text("Imagine a gate with two switches. A neuron multiplies each input by a weight, adds a bias, then applies an activation. Here the activation outputs 1 when the sum is at least zero; otherwise it outputs 0.",
              "İki anahtarlı bir kapı düşün. Nöron her girdiyi ağırlığıyla çarpar, bias ekler ve aktivasyonu uygular. Burada toplam sıfır veya pozitifse çıktı 1, negatifse 0 olur."))
st.latex(r"z=1\cdot x_1+1\cdot x_2-1.5,\qquad a=\mathbf{1}[z\geq 0]")
choice = st.radio(text("For inputs (1, 0), predict the output:", "Girdiler (1, 0) iken çıktıyı tahmin et:"), [0, 1], index=None)
if choice is not None:
    if choice == 0:
        st.success(text("Yes: z = 1 + 0 − 1.5 = −0.5, so a = 0.", "Doğru: z = 1 + 0 − 1.5 = −0.5, dolayısıyla a = 0."))
    else:
        st.info(text("Hint: calculate the sum before applying the threshold.", "İpucu: eşiği uygulamadan önce toplamı hesapla."))
st.markdown(text("### 2 · Change one thing (8 minutes)", "### 2 · Tek bir şeyi değiştir (8 dakika)"))
bias = st.slider(text("Bias", "Bias (sabit terim)"), -2.5, .5, -1.5, .5)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
z = X.sum(axis=1) + bias
st.dataframe({"x₁": X[:, 0], "x₂": X[:, 1], "z": z, "a": (z >= 0).astype(int)}, hide_index=True)
st.write(text("At bias −1.5, both switches must be on: AND. At −0.5, either is enough: OR. The weights are unchanged. Bias shifts the threshold; weights control how much each input contributes.",
              "Bias −1.5 iken iki anahtar da açık olmalı: AND. −0.5 iken birinin açık olması yeterli: OR. Ağırlıklar değişmedi. Bias karar eşiğini kaydırır; ağırlıklar girdilerin katkısını belirler."))
st.markdown(text("### 3 · Explain and rebuild (7 minutes)", "### 3 · Açıkla ve yeniden kur (7 dakika)"))
st.write(text("Without looking above, calculate z for (1, 1) at bias −1.5. Then explain why changing only the bias turns AND into OR. In Python, rebuild the two lines below from memory.",
              "Yukarı bakmadan, bias −1.5 iken (1, 1) için z'yi hesapla. Ardından yalnızca bias değiştirmenin AND'i OR'a nasıl çevirdiğini açıkla. Aşağıdaki iki Python satırını hafızandan yeniden yaz."))
guess = st.number_input(text("Your value of z", "Hesapladığın z"), value=None, step=.1)
if guess is not None:
    (st.success if abs(guess - .5) < 1e-9 else st.info)(text("Correct: 0.5." if abs(guess-.5)<1e-9 else "Add the two weighted inputs, then add the bias.", "Doğru: 0.5." if abs(guess-.5)<1e-9 else "İki ağırlıklı girdiyi topla, ardından bias ekle."))
with st.expander(text("Reveal the implementation", "Uygulamayı göster")):
    st.code("z = w1 * x1 + w2 * x2 + bias\na = int(z >= 0)", language="python")
with st.form("starter_checkpoint"):
    note = st.text_area(text("My explanation", "Kendi açıklamam"), max_chars=4000)
    stage = st.selectbox(text("What can I do independently?", "Bağımsız olarak ne yapabiliyorum?"), ["Explored", "Explained", "Rebuilt"],
                        format_func=lambda v: {"Explored":"İnceledim", "Explained":"Açıklayabiliyorum", "Rebuilt":"Yeniden kurabiliyorum"}[v] if tr else v)
    if st.form_submit_button(text("Save my first checkpoint", "İlk kaydımı sakla")):
        progress.update_topic("playground", stage=stage, bookmark=True, note=note)
        st.success(text("Saved. Continue with the neuron playground, then XOR.", "Kaydedildi. Nöron oyun alanıyla, ardından XOR ile devam et."))
st.page_link("views/playground.py", label=text("Next: explore weights and activations", "Sıradaki: ağırlıklar ve aktivasyonlar"))
st.page_link("views/two_neurons.py", label=text("Then: why XOR needs hidden features", "Sonra: XOR neden gizli özelliklere ihtiyaç duyar?"))
st.caption(text("Scope: hand-set weights and a step activation; no training algorithm yet. Source: the lab's single-neuron and perceptron notes. Guide updated 2026-09-20.",
                "Kapsam: elle seçilen ağırlıklar ve basamak aktivasyonu; henüz eğitim algoritması yok. Kaynak: laboratuvarın tek nöron ve perceptron notları. Rehber güncellemesi: 2026-09-20."))

st.page_link("views/beginner_path.py", label=text("Continue the path: learn → XOR → evaluate", "İki dilde devam et: öğren → XOR → değerlendir"))
