# Yol Haritası

Her seviye `core/` içine yeni parçalar ekler ve önceki parçaları yeniden kullanır.
Durum: `[ ]` yapılacak · `[~]` devam ediyor · `[x]` tamam.

## Seviye 0 — Nöron
- [x] **e01 — Tek nöron**: forward pass, ağırlıklar, bias, aktivasyon.
- [x] **e02 — Perceptron kapıları**: AND / OR öğrenir.
- [x] **e03 — XOR başarısızlığı**: tek lineer nöron XOR öğrenemez.

## Seviye 1 — Backprop ve autograd motoru
- [x] **e04 — Autograd motoru**: scalar reverse-mode autodiff.
- [x] **e05 — MLP XOR çözer**: `Value` tabanlı MLP 4/4 doğru öğrenir.
- [x] **e06 — Manual backprop**: numerical gradient check.
- [x] **e07 — Aktivasyonlar**: sigmoid vs ReLU ve vanishing gradients.
- [x] **e08 — Loss fonksiyonları**: MSE vs cross-entropy.

## Seviye 2 — Eğitim dinamikleri
- [x] **e09 — Optimizer'lar**: SGD → Momentum → Adam.
- [x] **e10 — Initialization**: Xavier/He ve sinyal ölçeği.
- [x] **e11 — Regularization**: L2, overfitting/underfitting.
- [ ] **e12 — MNIST from scratch**: saf NumPy MLP.
- [ ] **e13 — PyTorch karşılığı**: e12'yi PyTorch ile tekrarla.

## Seviye 3 — Mimariler
- [ ] e14 — CNN (LeNet) on MNIST.
- [ ] e15 — RNN/LSTM char-level model.
- [ ] e16 — Embeddings.
- [ ] e17 — Seq2seq + attention bottleneck.

## Seviye 4 — Attention ve Transformers
- [x] e18 — Attention from scratch.
- [x] e19 — Multi-head self-attention.
- [x] e20 — Full Transformer block.
- [x] e21 — nanoGPT-style char-level GPT.
- [ ] e22 — BPE tokenization.

## Seviye 5 — LLM
- [ ] e23 — Sampling: greedy / temperature / top-k / top-p.
- [ ] e24 — Küçük pretrained model fine-tune.
- [ ] e25 — LoRA.
- [ ] e26 — Instruction tuning ve RLHF/DPO.
- [ ] e27 — Embeddings + RAG; inference ve KV-cache.

## İki büyük "aha" hedefi
1. **Seviye 1 sonu:** Autograd motorunu ve XOR öğrenen MLP'yi kendin yazdın.
2. **Seviye 4 sonu:** Metin üreten küçük bir GPT eğittin.

