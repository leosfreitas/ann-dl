"""Exercise 4 -- does a second hidden layer help?  Figure 6.

Same data, same split, same learning rate, same epoch budget as Exercise 3;
only the depth changes.  Both networks are trained from several seeds so the
comparison is between means, not between two lucky or unlucky draws.  Seed 42
is the run reported as "the" Exercise 3 / Exercise 4 network.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from style import apply_style, title, SERIES
from mlp import fit, accuracy, n_params

apply_style()
FIG = "../figures"
RES = "../results"
for _d in (FIG, RES):
    os.makedirs(_d, exist_ok=True)
OUT = {}

SHALLOW = [4, 16, 3]         # Exercise 3
DEEP = [4, 16, 16, 3]        # Exercise 4: one extra hidden layer, same width
ETA, EPOCHS = 0.3, 2000      # identical to Exercise 3
SEEDS = [42, 43, 44, 45, 46]

X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=2, class_sep=1.0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)

runs = {}
for name, sizes in (("shallow", SHALLOW), ("deep", DEEP)):
    runs[name] = {"sizes": sizes, "n_params": n_params(sizes), "seeds": SEEDS,
                  "final_loss": [], "train_acc": [], "test_acc": [], "history": []}
    for seed in SEEDS:
        params, history = fit(sizes, X_tr, y_tr, ETA, EPOCHS,
                              np.random.default_rng(seed))
        r = runs[name]
        r["final_loss"].append(float(history[-1]))
        r["train_acc"].append(accuracy(params, X_tr, y_tr))
        r["test_acc"].append(accuracy(params, X_te, y_te))
        r["history"].append(history)
    r = runs[name]
    for k in ("final_loss", "train_acc", "test_acc"):
        r[k + "_mean"] = float(np.mean(r[k]))
        r[k + "_std"] = float(np.std(r[k], ddof=1))

OUT["eta"], OUT["epochs"], OUT["seeds"] = ETA, EPOCHS, SEEDS
OUT["runs"] = {n: {k: v for k, v in r.items() if k != "history"} for n, r in runs.items()}
OUT["diff_test_acc_mean"] = runs["deep"]["test_acc_mean"] - runs["shallow"]["test_acc_mean"]
diffs = np.array(runs["deep"]["test_acc"]) - np.array(runs["shallow"]["test_acc"])
OUT["diff_test_acc_per_seed"] = diffs.tolist()
OUT["diff_test_acc_std"] = float(np.std(diffs, ddof=1))
OUT["diff_final_loss_mean"] = runs["deep"]["final_loss_mean"] - runs["shallow"]["final_loss_mean"]

# Figure 6: seed-42 curves bold, the other seeds faint behind them.
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ep = np.arange(1, EPOCHS + 1)
for k, (name, label) in enumerate((("shallow", "Exercise 3"), ("deep", "Exercise 4"))):
    r = runs[name]
    for h in r["history"][1:]:
        ax.plot(ep, h, color=SERIES[k], lw=0.7, alpha=0.3)
    ax.plot(ep, r["history"][0], color=SERIES[k], lw=1.8,
            label=f"{label}: {r['sizes']}, {r['n_params']} params")
ax.plot([], [], color="grey", lw=0.7, alpha=0.5, label="Seeds 43-46 (faint)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (categorical cross-entropy)")
ax.set_ylim(0, None)
ax.legend(loc="upper right")
title(ax, "Figure 6 - One vs two hidden layers",
      f"Same data, eta={ETA}, {EPOCHS} epochs; bold = seed 42")
fig.savefig(f"{FIG}/fig6_depth.png")
plt.close(fig)

with open(os.path.join(RES, "mlp4.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
