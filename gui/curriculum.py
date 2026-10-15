"""One public curriculum catalog for routing, prerequisites and learner checkpoints."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    key: str
    title: str
    track: str
    minutes: int
    objective: str
    prerequisites: tuple[str, ...] = ()
    assumption: str = "Small educational examples isolate a concept; results are not production benchmarks."

    updated: str = "2026-09-20"
    reviewed: str | None = None
    turkish_status: str = "Partial interface translation; English lesson content remains. Human review pending."

    @property
    def path(self):
        return f"views/{self.key}.py"


TOPICS = [
    Topic("playground", "One neuron", "ANN", 20, "Predict how a weight or bias changes the output.", (), "Weights are hand-set here; a single neuron has a linear decision boundary."),
    Topic("two_neurons", "Two neurons and XOR", "ANN", 15, "Explain why hidden features make XOR separable.", ("playground",)),
    Topic("neurons_compute", "Neurons compute", "ANN", 15, "Construct a logic gate from thresholds.", ("two_neurons",)),
    Topic("backprop", "Backpropagation", "ANN", 25, "Reconstruct a gradient using the chain rule.", ("playground", "math_calculus")),
    Topic("worked_numbers", "Worked numbers", "ANN", 25, "Calculate a forward pass and an update by hand.", ("backprop",)),
    Topic("worked_numbers2", "Worked numbers II", "ANN", 30, "Track matrix shapes and gradients through a larger example.", ("worked_numbers", "math_vectors")),
    Topic("activations", "Activation functions", "ANN", 25, "Relate activation derivatives to gradient flow.", ("backprop",)),
    Topic("optimizers", "Optimizers", "ANN", 20, "Compare SGD, momentum and Adam on the same objective.", ("backprop",)),
    Topic("mlp", "Train an MLP", "ANN", 30, "Explain forward, loss, backward and update on XOR.", ("backprop", "optimizers")),
    Topic("deep_playground", "Deep networks", "ANN", 25, "Distinguish model capacity from generalization.", ("mlp",)),
    Topic("regularization", "Regularization", "ANN", 20, "Choose a penalty using validation data.", ("mlp",)),
    Topic("cnn", "Convolutional networks", "ANN", 25, "Explain convolution and weight sharing.", ("mlp", "math_vectors")),
    Topic("rnn", "Recurrent networks", "ANN", 25, "Trace hidden state through a sequence.", ("mlp",)),
    Topic("tokenization", "Tokenization", "ANN", 20, "Explain vocabulary size and a BPE merge.", ("playground",)),
    Topic("attention", "Attention", "ANN", 30, "Compute QK scores, masking, softmax and weighted values.", ("math_vectors", "math_information")),
    Topic("transformer", "Transformer and next-token learning", "ANN", 30, "Distinguish a bigram model, an MLP and a Transformer.", ("attention", "tokenization"), "The browser trains a bigram and an MLP. Experiment e21 is the actual Transformer."),
    Topic("generative", "Generative models", "ANN", 25, "Compare how different models generate samples.", ("mlp", "math_probability")),
    Topic("interpretability", "Interpretability", "ANN", 25, "Distinguish probing from causal intervention.", ("mlp",)),
    Topic("sampling", "Sampling", "ANN", 20, "Explain how temperature and truncation change a distribution.", ("transformer",)),
    Topic("embeddings", "Embeddings and RAG", "ANN", 25, "Separate retrieval quality from answer quality.", ("math_vectors",), "The small retrieval demo illustrates ranking; it does not establish real-world RAG accuracy."),
    Topic("posttraining", "Post-training", "ANN", 25, "Choose between SFT, preference training and retrieval.", ("transformer",)),
    Topic("ml_foundations", "ML foundations", "ML", 15, "Separate features, targets, training and evaluation."),
    Topic("ml_regression", "Regression", "ML", 25, "Fit a line and compare held-out error to a baseline.", ("ml_foundations",)),
    Topic("ml_classification", "Classification", "ML", 25, "Choose a threshold and inspect false positives and negatives.", ("ml_regression",)),
    Topic("ml_trees", "Trees and ensembles", "ML", 20, "Explain tree splits, depth and ensembling.", ("ml_classification",)),
    Topic("ml_svm", "SVM and kernels", "ML", 20, "Explain margins and nonlinear feature spaces.", ("ml_classification", "math_vectors")),
    Topic("ml_unsupervised", "Unsupervised learning", "ML", 25, "Interpret clusters and dimensionality reduction.", ("ml_foundations", "math_vectors")),
    Topic("ml_model_selection", "Model selection", "ML", 25, "Tune on validation data and reserve the test set.", ("ml_classification",)),
    Topic("ml_practical", "Practical ML", "ML", 20, "Identify leakage in preprocessing and splitting.", ("ml_model_selection",)),
    Topic("ml_python", "ML in Python", "ML", 30, "Build and evaluate a scikit-learn pipeline.", ("ml_practical",)),
    Topic("math_vectors", "Vectors and matrices", "Math", 25, "Compute dot products and check matrix shapes."),
    Topic("math_calculus", "Calculus", "Math", 25, "Connect derivatives, gradients and the chain rule.", ("math_vectors",)),
    Topic("math_probability", "Probability", "Math", 25, "Reason about conditional probability and uncertainty."),
    Topic("math_optimization", "Optimization", "Math", 25, "Explain how step size and curvature affect convergence.", ("math_calculus",)),
    Topic("math_information", "Information theory", "Math", 20, "Calculate entropy and cross-entropy.", ("math_probability",)),
    Topic("math_numerical", "Numerical computing", "Math", 20, "Recognize overflow, cancellation and gradient-check error.", ("math_calculus",)),
]
CATALOG = {t.key: t for t in TOPICS}


def next_topic(key):
    topic = CATALOG.get(key)
    if not topic:
        return None
    track = [t for t in TOPICS if t.track == topic.track]
    i = track.index(topic)
    return track[i + 1] if i + 1 < len(track) else None

# Scope statements describe the actual teaching simplifications, not model capabilities.
ASSUMPTIONS = {
    "two_neurons": "Hand-set threshold features solve four XOR inputs; this does not demonstrate learning or generalization.",
    "neurons_compute": "Logic gates use ideal binary inputs and chosen weights; real measurements may be noisy.",
    "backprop": "Small differentiable examples expose the chain rule; full training adds batching and optimizer state.",
    "worked_numbers": "Tiny fixed examples make arithmetic traceable; one update is not evidence of convergence.",
    "worked_numbers2": "Small dense matrices omit large-model memory and numerical-precision constraints.",
    "activations": "Random-network gradient probes isolate initialization and activation effects; they do not benchmark trained models.",
    "optimizers": "Low-dimensional objectives expose update behavior; an optimizer winning here need not win on real data.",
    "mlp": "Four XOR examples illustrate the training loop; fitting them does not measure held-out performance.",
    "deep_playground": "Synthetic low-dimensional data illustrate capacity; conclusions depend on the seed and split.",
    "regularization": "Toy validation curves illustrate bias and variance; choose penalties on validation data, not the test set.",
    "cnn": "Small images and kernels illustrate convolution; visual similarity does not establish learned feature quality.",
    "rnn": "Short sequences isolate recurrence; they do not establish long-context memory.",
    "tokenization": "Tiny vocabularies illustrate tokenization; token IDs and compression depend on the fitted corpus.",
    "attention": "Small attention matrices expose weighting and masking; an attention weight is not a causal explanation.",
    "generative": "Small synthetic models illustrate generation mechanisms; sample appearance alone is not an evaluation metric.",
    "interpretability": "Probes show associations; causal claims require controlled interventions and suitable baselines.",
    "sampling": "Sampling changes an existing distribution; it does not improve the model's knowledge or factual accuracy.",
    "posttraining": "Simplified objectives illustrate SFT and preference training; proxy rewards are not verified truth.",
    "ml_foundations": "Teaching datasets isolate features and targets; a split is valid only if it matches the intended future use.",
    "ml_regression": "Synthetic regression assumes stable train/test relationships; real data may violate that assumption.",
    "ml_classification": "Synthetic labels and costs illustrate thresholds; accuracy alone cannot justify deployment.",
    "ml_trees": "Small datasets expose splitting and ensembling; feature importance is not a causal effect.",
    "ml_svm": "Low-dimensional visualizations show margins; scaling and kernel choices change the geometry.",
    "ml_unsupervised": "Clusters and projections describe structure under a chosen metric; they need not be meaningful categories.",
    "ml_model_selection": "Validation compares candidates; repeated tuning against a test set makes it part of training.",
    "ml_practical": "Illustrative pipelines assume the split respects entities and time; preprocessing must fit training data only.",
    "ml_python": "Compact scikit-learn examples teach pipelines; production also requires data and distribution checks.",
    "math_vectors": "Small dense real-valued matrices illustrate shapes; array layout and floating-point arithmetic add constraints.",
    "math_calculus": "Derivatives assume local smoothness; numerical differences are approximate and step-size dependent.",
    "math_probability": "Small probability models use stated independence assumptions; correlation does not imply independence.",
    "math_optimization": "Low-dimensional surfaces illustrate local updates; nonconvex training need not find a global optimum.",
    "math_information": "Finite distributions illustrate entropy; the logarithm base determines the units.",
    "math_numerical": "Floating-point demonstrations depend on precision and scale; finite differences are checks, not exact derivatives.",
}
EXPERIMENT_BRIDGES = {
    "playground": ("e01", "e02"), "two_neurons": ("e03",),
    "backprop": ("e05", "e06"), "mlp": ("e12", "e13"),
    "deep_playground": ("e12", "e13"), "attention": ("e18", "e19", "e20"),
    "transformer": ("e18", "e19", "e20", "e21"),
}
