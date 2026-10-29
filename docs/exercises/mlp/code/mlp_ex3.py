"""Exercise 3 -- three classes with the same network code.  Figures 4, 5.

The model comes from mlp.py exactly as Exercise 2 used it: same ``fit``, same
``forward`` / ``backward`` / ``step``.  The only things that change are the
arguments -- 4 inputs instead of 2 and 3 output units instead of 1, which
switches the output to softmax and the loss to categorical cross-entropy.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from style import apply_style, title, SERIES, INK
from mlp import fit, predict, accuracy, n_params

apply_style()
FIG = "../figures"
RES = "../results"
for _d in (FIG, RES):
    os.makedirs(_d, exist_ok=True)
OUT = {}

SIZES = [4, 16, 3]
ETA, EPOCHS = 0.3, 2000

rng = np.random.default_rng(42)

# ------------------------------------------------------------------ A: data
X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=2, class_sep=1.0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)
OUT["n_train"], OUT["n_test"] = int(len(y_tr)), int(len(y_te))
OUT["class_counts"] = np.bincount(y).tolist()

# Linear reference: multinomial logistic regression on the same split.
lin = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
OUT["baseline_test_acc"] = float(lin.score(X_te, y_te))

# -------------------------------------------------------------- C: training
params, history = fit(SIZES, X_tr, y_tr, ETA, EPOCHS, rng)
OUT["sizes"] = SIZES
OUT["n_params"] = n_params(SIZES)
OUT["eta"], OUT["epochs"] = ETA, EPOCHS
OUT["loss_first"], OUT["loss_last"] = float(history[0]), float(history[-1])
OUT["train_acc"] = accuracy(params, X_tr, y_tr)
OUT["test_acc"] = accuracy(params, X_te, y_te)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(np.arange(1, EPOCHS + 1), history, color=SERIES[0], lw=1.6,
        label=f"MLP {SIZES}, eta={ETA}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (categorical cross-entropy)")
ax.legend(loc="upper right")
title(ax, "Figure 4 - Training loss per epoch",
      f"Full-batch gradient descent, {EPOCHS} epochs; final loss {history[-1]:.4f}")
fig.savefig(f"{FIG}/fig4_loss.png")
plt.close(fig)

# ----------------------------------------------------------- confusion matrix
cm = confusion_matrix(y_te, predict(params, X_te), labels=[0, 1, 2])
OUT["confusion"] = cm.tolist()
# Most-confused pair: errors in both directions added together.
pairs = {(i, j): int(cm[i, j] + cm[j, i]) for i in range(3) for j in range(i + 1, 3)}
worst = max(pairs, key=pairs.get)
OUT["pair_errors"] = {f"{i}-{j}": v for (i, j), v in pairs.items()}
OUT["most_confused"] = list(worst)

fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.grid(False)
im = ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > cm.max() / 2 else INK)
ax.set_xticks(range(3), [f"Class {k}" for k in range(3)])
ax.set_yticks(range(3), [f"Class {k}" for k in range(3)])
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Test samples")
title(ax, "Figure 5 - Test confusion matrix",
      f"{OUT['n_test']} test samples; accuracy {OUT['test_acc']:.1%}")
fig.savefig(f"{FIG}/fig5_confusion.png")
plt.close(fig)

# ------------------------------------------------------- cluster structure (D)
# make_classification puts each class's 2 clusters on vertices of a 4-D
# hypercube with side 2*class_sep.  Recover the 6 cluster centres with a tiny
# NumPy 2-means per class (descriptive only -- nothing here feeds the model).
def two_means(Z, rng, iters=50):
    C = Z[rng.choice(len(Z), 2, replace=False)]
    for _ in range(iters):
        a = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        C = np.array([Z[a == k].mean(0) for k in (0, 1)])
    return C

crng = np.random.default_rng(42)
centres = {k: two_means(X[y == k], crng) for k in range(3)}
OUT["cluster_centres"] = {str(k): np.round(c, 2).tolist() for k, c in centres.items()}
dist = {}
for i in range(3):
    for j in range(i + 1, 3):
        d = np.linalg.norm(centres[i][:, None, :] - centres[j][None], axis=-1)
        dist[f"{i}-{j}"] = float(d.min())
OUT["nearest_centre_distance"] = dist

with open(os.path.join(RES, "mlp3.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
