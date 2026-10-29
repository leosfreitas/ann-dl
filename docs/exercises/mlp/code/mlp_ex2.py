"""Exercise 2 -- binary classification a straight line cannot do.  Figures 1, 2, 3."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from style import apply_style, title, SERIES, MARKERS, INK_SOFT
from mlp import fit, forward, accuracy, n_params

apply_style()
FIG = "../figures"
RES = "../results"
for _d in (FIG, RES):
    os.makedirs(_d, exist_ok=True)
OUT = {}

SIZES = [2, 8, 1]
ETA, EPOCHS = 0.1, 2000

rng = np.random.default_rng(42)

# ------------------------------------------------------------------ A: data
# Three blobs; cluster 0 is class 0, clusters 1 and 2 together are class 1, so
# class 1 sits on both sides of class 0 along the diagonal.
X, c = make_blobs(n_samples=[500, 250, 250], centers=[[0, 0], [3, 3], [-3, -3]],
                  cluster_std=1.2, random_state=42)
y = (c > 0).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)
OUT["n_train"], OUT["n_test"] = int(len(y_tr)), int(len(y_te))
OUT["class_counts"] = np.bincount(y).tolist()

LABELS = ["Class 0 (centre blob)", "Class 1 (two outer blobs)"]

fig, ax = plt.subplots(figsize=(6.2, 5.2))
for k in (0, 1):
    m = y == k
    ax.scatter(X[m, 0], X[m, 1], s=11, alpha=0.7, color=SERIES[k],
               marker=MARKERS[k], linewidths=0, label=f"{LABELS[k]}, n={m.sum()}")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.legend(loc="upper left")
title(ax, "Figure 1 - Three blobs, two classes",
      "Class 1 surrounds class 0 on opposite sides of the diagonal")
fig.savefig(f"{FIG}/fig1_scatter.png")
plt.close(fig)

# -------------------------------------------------------------- B: baseline
lin = LogisticRegression().fit(X_tr, y_tr)
OUT["baseline_test_acc"] = float(lin.score(X_te, y_te))
OUT["baseline_train_acc"] = float(lin.score(X_tr, y_tr))
OUT["baseline_coef"] = lin.coef_[0].tolist()
OUT["baseline_intercept"] = float(lin.intercept_[0])

# ------------------------------------------------------------------- C: MLP
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
ax.set_ylabel("Training loss (binary cross-entropy)")
ax.legend(loc="upper right")
title(ax, "Figure 2 - Training loss per epoch",
      f"Full-batch gradient descent, {EPOCHS} epochs; final loss {history[-1]:.4f}")
fig.savefig(f"{FIG}/fig2_loss.png")
plt.close(fig)

# Decision regions over a grid, test points on top.
x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 400)
x2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 400)
G1, G2 = np.meshgrid(x1, x2)
grid = np.c_[G1.ravel(), G2.ravel()]
p = forward(params, grid)[-1][:, 0].reshape(G1.shape)

fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.contourf(G1, G2, (p >= 0.5).astype(int), levels=[-0.5, 0.5, 1.5],
            cmap=ListedColormap(["#d6e5f7", "#fbdccd"]))
ax.contour(G1, G2, p, levels=[0.5], colors=INK_SOFT, linewidths=1.0)
for k in (0, 1):
    m = y_te == k
    ax.scatter(X_te[m, 0], X_te[m, 1], s=16, color=SERIES[k], marker=MARKERS[k],
               edgecolors="white", linewidths=0.4, label=f"{LABELS[k]} (test)")
ax.fill_between([], [], color="#d6e5f7", ec=INK_SOFT, lw=0.5, label="Region predicted class 0")
ax.fill_between([], [], color="#fbdccd", ec=INK_SOFT, lw=0.5, label="Region predicted class 1")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5)
title(ax, "Figure 3 - MLP decision regions",
      f"Test accuracy {OUT['test_acc']:.1%}; line = p(class 1) = 0.5")
fig.savefig(f"{FIG}/fig3_boundary.png")
plt.close(fig)

with open(os.path.join(RES, "mlp2.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
