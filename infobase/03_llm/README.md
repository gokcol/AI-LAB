# Language models

A language model estimates the next-token distribution given a context. Generation
repeatedly samples or chooses a token and appends it to that context.

Tokenization defines the discrete units. Embeddings map units to vectors. Causal
attention mixes available context; feed-forward layers transform features at each
position. Residuals preserve an additive path; normalization controls scale.

The GUI Tiny GPT page uses a bigram model and an MLP to isolate the learning objective.
They are not Transformers. e18–e20 build a forward-pass Transformer block; e21 trains
causal multi-head attention with next-token cross-entropy.

RAG supplies retrieved evidence at inference time. Fine-tuning changes parameters.
Evaluate retrieval and answer support separately; similar text may not answer the question.
The Evaluation challenges include an unanswerable question to make this distinction visible.

Checkpoint: explain which data reaches a token, where the loss is computed, and why
next-token training alone does not guarantee factual answers.
Sources: [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
[Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401).
