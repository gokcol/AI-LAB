"""Small, deterministic teaching examples and checked exercises. No UI or execution of learner code."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Exercise:
    title: str
    prompt: str
    answer: float | str
    hints: tuple[str, str]
    solution: str
    options: tuple[str, ...] = ()
    mistakes: tuple[tuple[float | str, str], ...] = ()
    tolerance: float = .005

    def check(self, value):
        if self.options:
            return value == self.answer
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(number) and abs(number - self.answer) <= self.tolerance

    def feedback(self, value):
        for wrong, message in self.mistakes:
            if self.options and value == wrong:
                return message
            if not self.options:
                try:
                    if abs(float(value) - float(wrong)) <= self.tolerance:
                        return message
                except (TypeError, ValueError):
                    pass
        return "Try again. Write down the intermediate steps, or request a hint."


LESSONS = {
    "math_vectors": {
        "title": "Dot products: turn features into a score", "track": "Math",
        "intro": "A vector is an ordered list of numbers. Think of two features of a parcel: its mass and its distance to travel. A weight tells us how much each feature contributes to a score. Multiply matching entries, then add the contributions.",
        "formula": r"s = w_1 x_1 + w_2 x_2 = \mathbf w \cdot \mathbf x",
        "symbols": "x₁, x₂ are feature values; w₁, w₂ are their weights; s is the score. The subscripts name positions, not powers. A negative contribution subtracts from the score. A bias b is an extra constant added after the dot product.",
        "scope": "We use two invented, already scaled features. A score is not automatically a probability. Comparing directions with cosine similarity also requires nonzero vectors.",
        "worked": "For w = (2, −1) and x = (3, 4):\n\n1. First contribution: 2 × 3 = **6**.\n2. Second contribution: −1 × 4 = **−4**.\n3. Add: w · x = 6 − 4 = **2**.\n4. With bias b = 1, the final score is **3**.\n\nIncreasing x₂ by one decreases this score by one because its weight is −1.",
        "recall": (("What does a negative weight do?", "Increasing that feature lowers the score, with the other features fixed."), ("Is a dot product the same as cosine similarity?", "No. A dot product depends on both length and direction. Cosine divides by the two nonzero lengths."), ("Why must the two vectors have the same length?", "Every feature needs a matching weight before the contributions can be added.")),
        "derive": r"For a change only in the second feature, $s' - s = w_2(x_2+\Delta)-w_2x_2=w_2\Delta$. With $w_2=-1$ and $\Delta=2$, the score falls by 2. This is also the partial derivative $\partial s/\partial x_2=w_2$.",
        "bridge": "A regression model uses this score plus a bias to predict a number. Logistic regression passes the score through a sigmoid to get a value between zero and one.",
        "links": (("ml_regression", "Use a weighted score in regression"), ("math_calculus", "Find how a score changes with its inputs")),
    },
    "math_calculus": {
        "title": "Gradients: choose a direction for improvement", "track": "Math",
        "intro": "A derivative tells you how quickly an output changes near one input value. A gradient collects those slopes when there is more than one adjustable number. To reduce a loss, take a small step against its gradient.",
        "formula": r"L(w)=(w-t)^2,\quad g=2(w-t),\quad w_{new}=w-\eta g",
        "symbols": "w is the adjustable value, t is the target, L is the squared-error loss, g is its derivative, and η (eta) is the learning rate. New means after one update. The derivative points uphill; the minus sign makes the update go downhill for a sufficiently small step.",
        "scope": "This is a smooth, one-parameter quadratic. A large step can increase loss. Real models have many parameters and may have nonconvex losses.",
        "worked": "Let target t = 1, current w = 3, and learning rate η = 0.25.\n\n1. Error: 3 − 1 = **2**; loss: 2² = **4**.\n2. Gradient: 2 × (3 − 1) = **4**.\n3. Update: 3 − 0.25 × 4 = **2**.\n4. New loss: (2 − 1)² = **1**.\n\nWith η = 1.5 instead, w becomes −3 and loss rises to 16. The direction was useful; the step was too large.",
        "recall": (("Why subtract the gradient?", "The gradient points toward the steepest local increase. Its negative is a local descent direction."), ("Does a negative gradient mean training is broken?", "No. It means increasing this parameter locally lowers the loss."), ("Does a zero gradient guarantee the best solution?", "No in general: it can also occur at a maximum or saddle. For our positive quadratic, it is the minimum.")),
        "derive": r"Write $u=w-t$ and $L=u^2$. The chain rule gives $dL/dw=(dL/du)(du/dw)=2u\times1=2(w-t)$. If the prediction instead is $wx$, then $d(wx-y)^2/dw=2(wx-y)x$: the extra $x$ comes from the inner function.",
        "bridge": "Regression learns weights by differentiating prediction error. Backpropagation applies the same chain rule through multiple operations.",
        "links": (("ml_regression", "Use a gradient to train a regression model"), ("backprop", "Follow the chain rule through a network")),
    },
    "ml_regression": {
        "title": "Regression: predict a number and measure the miss", "track": "ML",
        "intro": "Regression predicts a number, such as a delivery duration. Begin with a simple rule: multiply a feature by a slope, then add an intercept. Compare each prediction with its known answer before deciding whether the rule is useful.",
        "formula": r"\hat y=wx+b,\quad e=\hat y-y,\quad MSE=\frac{1}{n}\sum_{i=1}^{n}e_i^2",
        "symbols": "x is the input; y is the observed answer; ŷ (y-hat) is the prediction. w is the slope and b is the intercept. e is prediction minus observation here; n is the number of rows. Σ means add the squared errors from all rows, then divide by n.",
        "scope": "The guided example uses four invented rows and a straight line. Its displayed error is descriptive, not evidence of future accuracy. Use separate validation data to choose settings and a reserved test set for final evaluation.",
        "worked": "Use the two rows (x, y) = (1, 3), (2, 5) and the model ŷ = 2x.\n\n1. Predictions: **2, 4**.\n2. Errors (prediction − observation): **−1, −1**.\n3. Squared errors: **1, 1**; MSE = (1 + 1) / 2 = **1**.\n4. Add intercept b = 1: predictions become **3, 5**, so MSE is **0 on these two rows**.\n\nA perfect fit to these rows alone does not establish generalization.",
        "recall": (("Why square errors before averaging?", "Opposite signs cannot cancel, and larger misses receive a larger penalty. Squaring also makes outliers influential."), ("Why compare against a baseline?", "A complicated model should improve on a simple rule such as the training-target mean on the same held-out rows."), ("Where do you choose model settings?", "On validation data or cross-validation, with preprocessing fitted to training data. Keep the test set for the final estimate.")),
        "derive": r"For one row, $L=(wx+b-y)^2$. Its derivatives are $\partial L/\partial w=2(wx+b-y)x$ and $\partial L/\partial b=2(wx+b-y)$. For MSE, average these derivatives across rows. This uses the full squared loss, not half the squared loss.",
        "bridge": "A multi-feature regression prediction is a dot product plus a bias. Its updates use the same derivatives as the gradient guide.",
        "links": (("math_vectors", "Revisit the dot product"), ("math_calculus", "Revisit the gradient"), ("ml_classification", "Predict a class instead of a number")),
    },
    "ml_classification": {
        "title": "Classification: separate a score from a decision", "track": "ML",
        "intro": "Classification predicts a category. Imagine a spam filter that scores messages from zero to one. The score and the decision are different: you choose a threshold, then label a message positive when its score meets or exceeds that threshold.",
        "formula": r"p=\sigma(z)=\frac{1}{1+e^{-z}},\quad \hat y=1\ \text{if}\ p\geq\tau",
        "symbols": "z is a weighted score; σ (sigmoid) maps it between zero and one; e is the mathematical constant. p is the model's estimated positive-class probability, not a guarantee. τ (tau) is the decision threshold; ŷ is the predicted class, either 0 or 1.",
        "scope": "We use fixed invented scores and known labels to isolate threshold effects. They are not calibrated real-world probabilities. Choose thresholds on validation data using the costs of errors; this small example does not select a deployment threshold.",
        "worked": "Scores are [0.9, 0.6, 0.4, 0.2]; true labels are [1, 0, 1, 0]. With threshold 0.5:\n\n1. Predicted labels: **[1, 1, 0, 0]**.\n2. One true positive (TP), one false positive (FP), one false negative (FN), one true negative (TN).\n3. Precision = TP / (TP + FP) = **1/2**.\n4. Recall = TP / (TP + FN) = **1/2**.\n\nRaise the threshold to 0.8: FP disappears, precision becomes 1, and recall stays 1/2. Precision does not always rise when the threshold rises; which examples drop out matters.",
        "recall": (("What is the difference between precision and recall?", "Precision asks how many positive predictions were right. Recall asks how many actual positives were found."), ("What happens to recall when you raise a fixed model's threshold?", "It cannot increase, provided the dataset and positive class stay fixed. There can be no new positive predictions."), ("Is 99% accuracy necessarily useful?", "No. With 1% positives, predicting negative for every row gets 99% accuracy and misses every positive.")),
        "derive": r"Precision divides TP by all predicted positives: $TP/(TP+FP)$. Recall divides TP by all actual positives: $TP/(TP+FN)$. If a denominator is zero, that ratio is undefined; the guided explorer labels it unavailable instead of silently treating it as zero.",
        "bridge": "Logistic regression begins with a dot product plus bias. Training changes those weights; changing the decision threshold alone does not retrain the model.",
        "links": (("math_vectors", "Build the score from features"), ("ml_model_selection", "Choose settings using validation data")),
    },
    "math_probability": {
        "title": "Probability: update a belief when evidence arrives", "track": "Math",
        "intro": "Probability describes uncertainty. A conditional probability narrows the group you are talking about. In a spam filter, 'how often spam is flagged' and 'how often a flagged message is spam' describe different groups. Counting both groups makes the difference visible.",
        "formula": r"P(S\mid F)=\frac{P(F\mid S)P(S)}{P(F\mid S)P(S)+P(F\mid \neg S)P(\neg S)}",
        "symbols": "S means spam; F means flagged. P(S) is the base rate before checking the flag. The vertical bar means 'given'; ¬S means not spam. P(F | S) is the fraction of spam caught, while P(F | ¬S) is the false-positive rate. The denominator counts all flags.",
        "scope": "Our invented filter has fixed catch and false-positive rates. Expected counts may be fractional; they are not sampled messages. Bayes' rule does not require independent S and F. The rates may change in a real population.",
        "worked": "Imagine 1,000 messages, with 10% spam. The filter catches 80% of spam and flags 10% of legitimate messages.\n\n1. There are **100 spam** and **900 legitimate** messages.\n2. Expected flags: **80 spam** and **90 legitimate**.\n3. Among 170 flags, the spam share is **80 / 170 ≈ 0.471**.\n\nCatching 80% of spam does not mean 80% of flags are spam. The many legitimate messages matter.",
        "recall": (("What changes when you condition on a flag?", "The denominator becomes the flagged group, not the whole inbox."), ("Can a useful filter have mostly false alarms?", "Yes. When spam is rare, false flags from the large legitimate group can outnumber true flags."), ("When may you multiply marginal probabilities?", "For independent events. In general P(A and B) = P(A) P(B | A).")),
        "derive": r"The flagged spam fraction of all messages is $P(S)P(F\mid S)$. All flags come from two disjoint groups: spam and non-spam. Add their fractions, then divide the flagged spam fraction by that total. For an indicator $X$ that is 1 for spam and 0 otherwise, $E[X]=P(S)$ and $\mathrm{Var}(X)=P(S)(1-P(S))$.",
        "bridge": "The probability that a flag is correct is the filter's precision. A change in base rate can change precision even when recall and false-positive rate stay fixed.",
        "links": (("ml_classification", "Connect conditional probability to precision and recall"), ("ml_model_selection", "Evaluate a model on the population you care about"), ("math_information", "Connect probabilities to information and loss")),
    },
    "math_optimization": {
        "title": "Optimization: repeat useful steps without overshooting", "track": "Math",
        "intro": "A gradient gives a local direction. An optimizer decides how to use it repeatedly. A learning rate that works on a gentle curve can be too large on a steep one. Watch the parameter and the loss separately: the parameter can cross the minimum while the loss still falls.",
        "formula": r"L(w)=c(w-t)^2,\quad w_{k+1}=w_k-\eta\,2c(w_k-t),\quad e_{k+1}=(1-2\eta c)e_k",
        "symbols": "c is a positive scale controlling curvature; t is the target; k counts updates; η is the learning rate. eₖ = wₖ − t is the signed distance from the optimum. The factor 1 − 2ηc tells us whether that distance shrinks, flips sign or grows.",
        "scope": "This is deterministic, full-gradient descent on one positive quadratic, without momentum. The convergence condition 0 < η < 1/c applies to this example from a non-optimal start, not to every neural network. At the optimum the gradient is already zero.",
        "worked": "Let L(w) = 2w², w₀ = 4 and η = 0.1.\n\n1. Gradient at 4: **2 × 2 × 4 = 16**.\n2. First update: **4 − 0.1 × 16 = 2.4**.\n3. Next gradient: **9.6**; next w: **1.44**.\n4. Loss falls **32 → 11.52 → 4.1472**.\n\nThe distance is multiplied by 0.6 each step. With η = 0.6, it is multiplied by −1.4: alternating sides and growing, so loss increases.",
        "recall": (("Does crossing the minimum always mean failure?", "No. Alternating sides can still converge when the distance shrinks on each step."), ("Why can the same learning rate fail on a steeper loss?", "Larger curvature gives larger gradients and lowers the stable learning-rate range."), ("Does lower training loss prove better predictions?", "No. Optimization improves the training objective; held-out evaluation checks generalization.")),
        "derive": r"Subtract $t$ from the update: $w_{k+1}-t=(1-2\eta c)(w_k-t)$. Thus $e_k=(1-2\eta c)^k e_0$. For a nonzero initial error, convergence needs $|1-2\eta c|<1$, giving $0<\eta<1/c$. At $\eta=1/c$, the sign flips with unchanged loss; at $\eta=1/(2c)$ it reaches the target in one step.",
        "bridge": "Regression training repeats gradient updates. Evaluation asks whether those improvements carry to new data. The older playground adds ravines and momentum after you understand the simple update.",
        "links": (("math_calculus", "Revisit where the gradient comes from"), ("ml_regression", "Train a regression model"), ("ml_model_selection", "Separate optimization from generalization")),
    },
    "ml_model_selection": {
        "title": "Model evaluation: choose what a good score means", "track": "ML",
        "intro": "Evaluation asks how well a model works on examples it did not learn from. First define the task and the cost of mistakes. Then compare with a simple baseline on the same data. Training data fits the model, validation data selects it, and a reserved test set estimates the selected model's performance.",
        "formula": r"\mathrm{accuracy}=\frac{TP+TN}{n},\quad \mathrm{recall}=\frac{TP}{TP+FN},\quad F_1=\frac{2TP}{2TP+FP+FN}",
        "symbols": "TP and TN are correct positive and negative predictions; FP is a false alarm; FN is a missed positive. n is their sum. Recall asks how many real positives were found. F₁ balances precision and recall but ignores true negatives and does not encode your actual error costs.",
        "scope": "We use small, invented confusion counts and candidate scores. One split is a noisy estimate, not proof of future performance. Choose metrics, thresholds and preprocessing using training/validation only. Use grouped or time-ordered splits when ordinary random splits would leak information.",
        "worked": "In 100 messages, 10 are spam.\n\n1. An always-legitimate baseline has **90% accuracy** and **0 recall**.\n2. Another model catches 8 spam, misses 2, and falsely flags 6 legitimate messages: TP=8, FN=2, FP=6, TN=84.\n3. Accuracy = **92/100 = 0.92**; recall = **8/10 = 0.8**.\n4. F₁ = **16/(16+6+2) ≈ 0.667**.\n\nThe confusion counts explain what accuracy hides. If false alarms and misses have different costs, compare those costs before choosing.",
        "recall": (("Why keep a test set separate from validation?", "Repeatedly choosing settings from a score adapts the model-selection process to that data. The test set must stay outside that loop."), ("Where should a scaler be fitted during cross-validation?", "Inside each fold, on its training rows only; then apply it to that fold's validation rows."), ("Does a higher F₁ always mean a better model?", "No. The useful metric depends on the task, positive class, error costs and deployment population.")),
        "derive": r"For $k$ equally sized validation folds with errors $E_1,\ldots,E_k$, mean CV error is $\frac{1}{k}\sum_j E_j$. Choose a candidate using that validation measure, refit on the permitted training data, then evaluate on the reserved test set. The score used to choose the winner can be optimistic; a final test or nested CV evaluates the whole selection process.",
        "bridge": "Probability explains base-rate effects; classification supplies the decisions; optimization fits the model. Evaluation joins them by asking which claims the available evidence supports.",
        "links": (("math_probability", "Understand why base rates change precision"), ("ml_classification", "Inspect thresholds and confusion counts"), ("evaluation_lab", "Try misleading scores and insufficient-evidence challenges"), ("ml_practical", "Prevent leakage with splits and pipelines")),
    },
}


def exercises(topic, variant=0):
    """Five reproducible number sets; stable question identities support saved practice."""
    if variant not in range(5):
        raise ValueError("Choose a practice set from 1 to 5.")
    a = 2 + variant
    E = Exercise
    if topic == "math_vectors":
        return [
            E("Multiply, then add", f"w = ({a}, −1), x = (3, 4). What is w · x?", 3*a-4, ("Pair entries in the same position.", "Add the first product to −4."), f"{a} × 3 + (−1) × 4 = {3*a-4}.", mistakes=((3*a+4, "Keep the negative sign on the second weight."),)),
            E("Add the bias", f"The dot product is {a}. With bias b = −3, what is z?", a-3, ("Bias is added after the dot product.", "Adding a negative number is subtraction."), f"z = {a} + (−3) = {a-3}."),
            E("Predict a change", f"The second weight is −{a}. Only x₂ increases by 2. What is the signed change in score?", -2*a, ("The unchanged contribution cancels.", "Change = second weight × change in second feature."), f"Δs = −{a} × 2 = {-2*a}.", mistakes=((2*a, "A negative weight makes this increase lower the score."),)),
            E("Perpendicular vectors", f"What is ({a}, 0) · (0, 3)?", 0, ("Calculate both products separately.", "Each product contains a zero."), "Both contributions are zero, so the dot product is zero."),
            E("Length or direction?", "If a nonzero vector doubles in length without changing direction, what happens to its cosine similarity with another nonzero vector?", "Unchanged", ("Cosine divides the dot product by both lengths.", "The numerator and denominator both double."), "The scale cancels; cosine measures direction.", ("Doubles", "Unchanged", "Becomes zero")),
            E("Check the shape", f"A data row has {a+1} features. How many weights does a single dot-product score need, excluding bias?", a+1, ("Each feature has one matching weight.", "Bias is a separate scalar."), f"{a+1} features need {a+1} weights."),
            E("Complete the implementation", "For equal-length numeric lists w and x, select the expression that computes their dot product.", "sum(wi * xi for wi, xi in zip(w, x))", ("You need products of matching entries.", "Reduce the products to one number by adding."), "zip pairs entries; multiplication makes contributions; sum adds them.", ("sum(w) * sum(x)", "sum(wi * xi for wi, xi in zip(w, x))", "sum(wi + xi for wi, xi in zip(w, x))")),
            E("Transfer to a prediction", f"A toy delivery model is y = {a} × mass + 1 × distance + 2. For mass = 3 and distance = 4, what does it predict?", 3*a+6, ("Calculate the two weighted features first.", "Remember the final bias of 2."), f"{a} × 3 + 4 + 2 = {3*a+6}."),
        ]
    if topic == "math_calculus":
        return [
            E("Read a slope", f"For f(x) = x², what is f′({a})?", 2*a, ("The derivative of x² is 2x.", "Substitute the given x after differentiating."), f"f′({a}) = 2 × {a} = {2*a}.", mistakes=((a*a, "That is the function value, not its slope."),)),
            E("Shift the target", f"L(w) = (w − 1)². What is the gradient at w = {a+1}?", 2*a, ("First find the distance from the target.", "Multiply that signed distance by 2."), f"g = 2 × ({a+1} − 1) = {2*a}."),
            E("Take one step", f"w = {a+1}, gradient = {2*a}, η = 0.25. What is the new w?", a+1-.5*a, ("Update = old value minus learning rate times gradient.", f"The amount to subtract is {2*a} × 0.25."), f"w_new = {a+1} − 0.25 × {2*a} = {a+1-.5*a:g}.", mistakes=((a+1+.5*a, "You added the gradient. Gradient descent subtracts it."),)),
            E("A negative gradient", "At w = 0, L(w) = (w − 2)² has gradient −4. Which way should a small descent step move w?", "Increase w", ("The update subtracts the gradient.", "Subtracting a negative number adds a positive number."), "w_new = 0 − η(−4) = 4η, so w increases.", ("Increase w", "Decrease w", "Keep w fixed")),
            E("Chain rule", f"L(w) = ({a}w − 1)². What is dL/dw at w = 1?", 2*(a-1)*a, ("Differentiate the square, then the expression inside it.", f"The inner derivative is {a}, not 1."), f"g = 2({a} × 1 − 1) × {a} = {2*(a-1)*a}.", mistakes=((2*(a-1), "The inner derivative is missing. Multiply by the input coefficient."),)),
            E("Too large a step", "L(w) = w², w = 1, η = 1.5. What is the loss AFTER one gradient-descent step?", 4, ("The gradient is 2w = 2.", "First update w, then square the new w."), "w_new = 1 − 1.5 × 2 = −2; new loss = (−2)² = 4.", mistakes=((-2, "That is the new parameter. The question asks for its loss."),)),
            E("Repair the update", "Which line performs gradient descent when grad is dL/dw?", "w = w - learning_rate * grad", ("A positive derivative points toward increasing loss.", "Use the negative-gradient direction."), "Subtract a suitably scaled gradient.", ("w = w + learning_rate * grad", "w = w - learning_rate * grad", "w = learning_rate - grad")),
            E("A partial derivative", f"L(u, v) = u² + 3v². At (u, v) = (1, {a}), what is ∂L/∂v?", 6*a, ("Treat u as constant when differentiating with respect to v.", "d(3v²)/dv = 6v."), f"∂L/∂v = 6 × {a} = {6*a}."),
        ]
    if topic == "ml_regression":
        return [
            E("Make a prediction", f"ŷ = {a}x + 1. At x = 3, what is ŷ?", 3*a+1, ("Multiply by the slope first.", "Then add the intercept."), f"ŷ = {a} × 3 + 1 = {3*a+1}."),
            E("Keep the error sign", f"Prediction = {a}, observation = {a+3}. Using e = prediction − observation, what is e?", -3, ("Order matters in subtraction.", "The prediction is below the observation."), f"e = {a} − {a+3} = −3.", mistakes=((3, "That is observation minus prediction. Use the convention given here."),)),
            E("Average squared errors", f"Two prediction errors are −{a} and {a+1}. What is their MSE?", (a*a+(a+1)**2)/2, ("Square each error before adding.", "There are two rows; divide the sum by 2."), f"MSE = ({a*a} + {(a+1)**2}) / 2 = {(a*a+(a+1)**2)/2:g}.", mistakes=((a*a+(a+1)**2, "You found the sum. MSE is the mean: divide by the number of rows."),)),
            E("Build a baseline", f"Training targets are [{a}, {a+2}, {a+4}]. What constant training-mean baseline would you predict for a new row?", a+2, ("Add the training targets and divide by their count.", "Do not use the new row's unknown target."), f"({a} + {a+2} + {a+4}) / 3 = {a+2}."),
            E("Differentiate one row", f"L = (wx − y)², x = {a}, y = 1, w = 1. What is dL/dw?", 2*(a-1)*a, ("Compute the prediction error first.", "Gradient = 2 × error × input."), f"g = 2 × ({a} − 1) × {a} = {2*(a-1)*a}."),
            E("Reserve the test set", "You are choosing between model settings. Which data should guide that choice?", "Validation data / cross-validation", ("Selection needs data not used to fit that candidate.", "Save the test set for the final estimate."), "Use validation or CV for selection, then evaluate once on the reserved test set.", ("Training error alone", "Validation data / cross-validation", "Test data after every change")),
            E("Complete the loss", "Given NumPy arrays prediction and target with the same shape, select MSE.", "np.mean((prediction - target) ** 2)", ("Square errors before averaging.", "Squaring the mean error allows cancellation first."), "np.mean((prediction - target) ** 2) averages per-row squared errors.", ("np.mean(prediction - target) ** 2", "np.mean((prediction - target) ** 2)", "np.sum(prediction - target)")),
            E("Recognize an outlier", f"One error grows from {a} to {2*a}. By what factor does its SQUARED contribution grow?", 4, ("Compare (2e)² with e².", "Doubling before squaring multiplies by 2²."), "(2e)² / e² = 4 for nonzero e. This is why MSE is sensitive to large misses.", mistakes=((2, "The error doubled, but its squared contribution quadrupled."),)),
        ]
    if topic == "ml_classification":
        return [
            E("Score versus label", "p = 0.6 and threshold τ = 0.6. Using p ≥ τ, what label is predicted?", 1, ("The boundary is included in the positive class.", "Equality satisfies ≥."), "0.6 ≥ 0.6, so the predicted label is 1."),
            E("Name the mistake", "A normal message (true label 0) is flagged as spam (predicted label 1). What is this?", "False positive", ("Positive means the model flagged it.", "False means that decision was incorrect."), "This is a false positive: an incorrect positive prediction.", ("True positive", "False positive", "False negative")),
            E("Calculate precision", f"TP = {a}, FP = 1, FN = 2. What is precision? Enter a decimal, not a percentage.", a/(a+1), ("Precision asks about predicted positives.", "Divide TP by TP + FP."), f"Precision = {a}/({a}+1) = {a/(a+1):.4f}.", mistakes=((a/(a+2), "That denominator uses missed positives. It calculates recall, not precision."),)),
            E("Calculate recall", f"TP = {a}, FP = 1, FN = 2. What is recall? Enter a decimal, not a percentage.", a/(a+2), ("Recall asks about actual positives.", "Divide TP by TP + FN."), f"Recall = {a}/({a}+2) = {a/(a+2):.4f}.", mistakes=((a/(a+1), "That is precision. Recall includes missed positives in the denominator."),)),
            E("Raise the threshold", "On fixed scores and labels, raising the threshold makes recall…", "Stay the same or decrease", ("A higher threshold can only remove positive predictions.", "No new true positive can be added."), "The number of true positives cannot increase; the number of actual positives stays fixed.", ("Always increase", "Stay the same or decrease", "Always become zero")),
            E("Misleading accuracy", f"Among 100 rows, {a} are positive. A model predicts negative for every row. What is its recall?", 0, ("Count the positives the model found.", "TP = 0; actual positives are nonzero."), f"Recall = 0/{a} = 0, despite accuracy of {100-a}%."),
            E("Complete the decision", "Given a NumPy array p of scores, which expression follows the rule positive when p ≥ threshold?", "(p >= threshold).astype(int)", ("Compare each score with the threshold.", "Convert each True/False decision to 1/0."), "(p >= threshold).astype(int) implements the stated boundary exactly.", ("(p < threshold).astype(int)", "(p >= threshold).astype(int)", "p.astype(int)")),
            E("No positive predictions", "If a model predicts no positives, what is TP / (TP + FP)?", "Undefined: denominator is zero", ("Both TP and FP count predicted positives.", "Here both counts are zero."), "Precision is 0/0 and undefined. A library may choose a reporting convention; state it explicitly.", ("Always 1", "Undefined: denominator is zero", "Equal to accuracy")),
        ]
    if topic == "math_probability":
        p = a/10
        return [
            E("Count a probability", f"A bag contains {a} blue counters and {10-a} red counters. For a uniform random draw, what is P(blue), as a decimal?", p, ("Count favourable outcomes and all outcomes.", "Divide the blue count by 10."), f"P(blue) = {a}/10 = {p:g}."),
            E("Use the complement", f"P(spam) = {p:g}. What is P(not spam)?", 1-p, ("The two categories cover every message.", "Their probabilities sum to 1."), f"1 − {p:g} = {1-p:g}."),
            E("Condition on the right group", f"Of {a*10} flagged messages, {a*6} are spam. What is P(spam | flagged)?", .6, ("The given group is flagged messages.", "Divide flagged spam by all flags."), f"{a*6}/{a*10} = 0.6."),
            E("Independent events", f"Two independent draws each have P(blue) = {p:g}. What is the probability both are blue?", p*p, ("Independence permits multiplying the probabilities.", "Multiply the given probability by itself."), f"{p:g} × {p:g} = {p*p:g}."),
            E("Reverse the condition", f"In 1,000 messages, {a*100} are spam. A filter flags 80% of spam and 10% of non-spam. What fraction of flags are spam?", .8*p/(.8*p+.1*(1-p)), ("Count flagged spam and falsely flagged non-spam separately.", f"Divide {a*80} by {a*80} + {(10-a)*10}."), f"{a*80}/({a*80}+{(10-a)*10}) = {.8*p/(.8*p+.1*(1-p)):.4f}.", mistakes=((.8, "That is P(flag | spam); the question reverses the condition."),)),
            E("An expected count", f"Each of 100 messages has spam probability {p:g}. What is the expected spam count?", 100*p, ("Add the 100 individual probabilities.", "Expected count = number of messages × probability."), f"100 × {p:g} = {100*p:g}. Linearity of expectation does not require independence."),
            E("Complete Bayes' rule", "Given prior, catch_rate and false_alarm_rate, which denominator gives the probability of a flag?", "catch_rate * prior + false_alarm_rate * (1 - prior)", ("Flags come from spam and non-spam.", "Weight each conditional rate by its group's base rate."), "Add the two disjoint contributions to all flags.", ("catch_rate + false_alarm_rate", "catch_rate * prior + false_alarm_rate * (1 - prior)", "catch_rate * false_alarm_rate")),
            E("Change the base rate", "Spam becomes rarer, while nonzero catch and false-positive rates stay fixed. What happens to the fraction of flags that are spam?", "It decreases", ("There are fewer true flags and more non-spam messages.", "Compare the two groups in the Bayes denominator."), "True flags shrink relative to false flags, so precision decreases.", ("It increases", "It decreases", "It must stay fixed")),
        ]
    if topic == "math_optimization":
        return [
            E("Include curvature", f"L(w) = {a}w². What is its gradient at w = 2?", 4*a, ("Differentiate the square and keep its coefficient.", f"The derivative is 2 × {a} × w."), f"2 × {a} × 2 = {4*a}."),
            E("Take a scaled step", f"w = 2, gradient = {4*a}, learning rate = {1/(4*a):.6f}. What is new w? Round to 2 decimal places.", 1, ("Subtract learning rate times gradient.", "The update is approximately 1."), "Using η = 1/(4c), new w = 2 − 1 = 1."),
            E("Repeat the update", f"For L = w², η = 0.25 and w₀ = {a}, what is w after TWO updates?", a/4, ("Each update halves w.", "Apply the factor 0.5 twice."), f"{a} × 0.5 × 0.5 = {a/4:g}."),
            E("Read the stability boundary", f"For L = {a}w², gradient descent from w ≠ 0 converges when 0 < η < 1/{a}. What is the upper boundary as a decimal (not included)?", 1/a, ("The boundary is the reciprocal of the coefficient.", f"Calculate 1 divided by {a}."), f"Boundary = {1/a:.4f}. Equality oscillates; it is not in the convergent interval."),
            E("Crossing the minimum", "The signed distance to the optimum is multiplied by −0.5 each update. What happens?", "Alternates sides and converges", ("The negative sign flips sides.", "The absolute factor is below 1, so distance shrinks."), "Distance halves while changing sign; loss falls by a factor of 0.25 per update.", ("Alternates sides and converges", "Always diverges", "Keeps constant loss")),
            E("Spot a stalled loss", "For L = w², w₀ = 2 and η = 1, what is the loss after two updates?", 4, ("Each step multiplies w by −1.", "The parameter changes sign but its square does not change."), "w: 2 → −2 → 2. Loss stays 4."),
            E("Repair the training loop", "Which update recalculates the gradient for L = c(w − target)² at the current w each iteration?", "w = w - rate * (2 * c * (w - target))", ("Use the derivative at the current parameter.", "The coefficient c and shift by target must both appear."), "Recompute 2c(w − target) on every iteration, then subtract a scaled gradient.", ("w = w + rate * (2 * c * (w - target))", "w = w - rate * (2 * c * (w - target))", "w = w - rate * initial_gradient")),
            E("Training versus evaluation", "Training loss keeps falling but validation loss starts rising. What is the most useful next step?", "Check overfitting and choose stopping using validation", ("Optimization and generalization answer different questions.", "Keep the test set outside the tuning loop."), "Use validation to choose stopping or regularization; falling training loss alone is not success.", ("Keep training until training loss is zero", "Check overfitting and choose stopping using validation", "Tune every epoch against the test set")),
        ]
    if topic == "ml_model_selection":
        return [
            E("Measure the baseline", f"Of 100 messages, {a*5} are spam. An always-negative model gets what accuracy, as a decimal?", 1-a*.05, ("Only the non-spam rows are correct.", "Divide the correct count by 100."), f"Accuracy = {100-a*5}/100 = {1-a*.05:g}; recall is zero."),
            E("Count missed positives", f"A classifier has TP = {a*4}, FN = {a}, FP = 3, TN = 20. What is recall?", .8, ("Recall uses actual positives as its denominator.", f"Actual positives = {a*4} + {a}."), f"Recall = {a*4}/{a*5} = 0.8."),
            E("Calculate F₁", f"TP = {a}, FP = 1, FN = 2. What is F₁?", 2*a/(2*a+3), ("Use 2TP / (2TP + FP + FN).", "True negatives do not enter this measure."), f"F₁ = {2*a}/({2*a}+1+2) = {2*a/(2*a+3):.4f}."),
            E("Average validation error", f"Three equal-sized folds have MSEs {a}, {a+1}, {a+5}. What is mean CV MSE?", a+2, ("Add the three fold errors.", "Divide by 3, not by the number of training rows."), f"({a} + {a+1} + {a+5}) / 3 = {a+2}."),
            E("Choose the candidate", "Candidate A: train MSE 0.1, validation MSE 0.9. Candidate B: train MSE 0.3, validation MSE 0.5. If lower validation MSE is the preselected criterion, which candidate wins?", "B", ("The training score is not the selection criterion.", "Compare 0.9 with 0.5."), "B has lower validation error; reserve test data for the final estimate.", ("A", "B", "Choose using whichever has lower test error")),
            E("Repair leakage", "You scale features before cross-validation using all rows. How should you fix this?", "Fit the scaler on each fold's training rows inside a pipeline", ("Validation rows must not influence fitted preprocessing.", "Each fold needs its own fitted preprocessing and model."), "Fit on fold-training data, then transform fold-validation data with that fitted scaler.", ("Fit the scaler on each fold's training rows inside a pipeline", "Scale the test set independently", "Shuffle the leaked features again")),
            E("Match the split to the task", "You will predict next month's demand. Which evaluation better matches that use?", "Train on earlier dates and validate on later dates", ("Future data would be unavailable when predicting the past.", "Preserve the direction of time."), "Use time-ordered evaluation and prevent features from using future information.", ("Randomly mix all dates", "Train on earlier dates and validate on later dates", "Select the easiest dates for validation")),
            E("Make an honest claim", "You tuned 40 settings repeatedly against the same test set. What can you claim about the winning score?", "It was used for selection; get a new independent evaluation", ("Selection used information from this test set.", "Renaming the split does not restore independence."), "Treat that set as validation. Use genuinely untouched data or an appropriate nested evaluation.", ("It is an unbiased final test score", "It was used for selection; get a new independent evaluation", "It proves the winner will generalize")),
        ]
    raise ValueError("Unknown guided lesson")


def practice_key(topic, index, variant):
    return f"{topic}:{index}:{variant}"


PRACTICE_KEYS = {practice_key(t, i, v) for t in LESSONS for i in range(8) for v in range(5)}


def valid_practice_key(key):
    return isinstance(key, str) and key in PRACTICE_KEYS


def practice_summary(topic, records):
    """Count exercise families, not repeated successes on the same question."""
    solved = independent = 0
    for i in range(8):
        variants = [records.get(practice_key(topic, i, v), {}) for v in range(5)]
        solved += any(r.get("solved") for r in variants)
        independent += any(r.get("independent") for r in variants)
    return {"solved": solved, "independent": independent, "total": 8}
