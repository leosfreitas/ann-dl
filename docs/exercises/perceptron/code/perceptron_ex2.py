"""Exercise 2 -- The same perceptron on overlapping data.  Figures 4, 5, 6."""

import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from style import apply_style, title, SERIES, INK, INK_SOFT, INK_MUTED
from perceptron import fit, predict, accuracy

apply_style()
FIG = "../figures"
RES = "../results"
for _d in (FIG, RES):
    os.makedirs(_d, exist_ok=True)
OUT = {}

MU0, MU1 = np.array([3.0, 3.0]), np.array([4.0, 4.0])
COV = np.array([[1.5, 0.0], [0.0, 1.5]])
N_PER_CLASS = 1000
ETA, MAX_EPOCHS = 0.01, 100

# Same generator discipline as Exercise 1: one seed, drawn in a fixed order.
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ A: data
X0 = rng.multivariate_normal(MU0, COV, N_PER_CLASS)
X1 = rng.multivariate_normal(MU1, COV, N_PER_CLASS)
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(N_PER_CLASS, int), np.ones(N_PER_CLASS, int)])

OUT["n_per_class"] = N_PER_CLASS
OUT["mu0"], OUT["mu1"] = MU0.tolist(), MU1.tolist()
OUT["cov"] = COV.tolist()
OUT["centre_distance"] = float(np.linalg.norm(MU1 - MU0))
OUT["sigma"] = float(np.sqrt(COV[0, 0]))
OUT["eta"], OUT["max_epochs"] = ETA, MAX_EPOCHS
OUT["sigma_multiple"] = float(
    np.linalg.norm(MU1 - MU0) / np.sqrt(COV[0, 0]))

w0 = rng.normal(0.0, 0.01, size=2)
OUT["w_init"] = w0.tolist()
OUT["b_init"] = 0.0


# ------------------------------------------------------------------ Figure 4
def scatter_classes(ax, alpha=0.5, s=13):
    ax.scatter(X0[:, 0], X0[:, 1], s=s, c=SERIES[0], marker="o", alpha=alpha,
               linewidths=0.0, label="Class 0", zorder=3)
    ax.scatter(X1[:, 0], X1[:, 1], s=s, c=SERIES[1], marker="s", alpha=alpha,
               linewidths=0.0, label="Class 1", zorder=3)
    for mu, c in ((MU0, SERIES[0]), (MU1, SERIES[1])):
        ax.scatter(mu[0], mu[1], s=170, marker="X", c=c, edgecolors="white",
                   linewidths=1.6, zorder=6)


fig, ax = plt.subplots(figsize=(7.4, 5.2))
scatter_classes(ax)
for mu, lab in ((MU0, r"$\mu_0$ = (3, 3)"), (MU1, r"$\mu_1$ = (4, 4)")):
    ax.annotate(lab, xy=mu, xytext=(0, 15), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK, weight="600",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=INK_SOFT,
                          lw=0.8, alpha=0.92), zorder=7)
title(ax, "Figure 4 - Two heavily overlapping Gaussian classes",
      "2000 points (1000 per class), covariance 1.5I; the centres are 1.41 apart "
      "against sigma = 1.22")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(loc="upper left")
fig.savefig(FIG + "/fig4_scatter.png")
plt.close(fig)

# ------------------------------------- B: train, keeping the best weights
# The implementation is imported unchanged from Exercise 1; the only difference
# is pocket=True, which records the best epoch without feeding back into it.
res = fit(X, y, w0, eta=ETA, max_epochs=MAX_EPOCHS, pocket=True)
w, b = res["w"], res["b"]
pw, pb = res["pocket_w"], res["pocket_b"]

pred_final = predict(X, w, b)
pred_pocket = predict(X, pw, pb)
mis_final = pred_final != y
mis_pocket = pred_pocket != y

OUT["final"] = {
    "w": w.tolist(), "b": b, "acc": res["acc"], "epochs": res["epochs"],
    "converged": res["converged"],
    "n_misclassified": int(mis_final.sum()),
    "predicted_class1_fraction": float(pred_final.mean()),
    "w_norm": float(np.linalg.norm(w)),
}
OUT["pocket"] = {
    "w": pw.tolist(), "b": pb, "acc": res["pocket_acc"],
    "epoch": res["pocket_epoch"], "update": res["pocket_update"],
    "n_misclassified": int(mis_pocket.sum()),
    "w_norm": float(np.linalg.norm(pw)),
}

# Where a boundary actually sits.  The signed distance from the origin to
# w.x + b = 0 is -b / ||w||; for the line to cut through the clouds that has to
# come out near the projection of the cloud centre on the same unit normal.
# This is the quantity the bias is too slow to supply (see "scale" below).
mid = (MU0 + MU1) / 2.0


def offsets(wv, bv):
    n = float(np.linalg.norm(wv))
    return {"offset": float(-bv / n),
            "needed": float((wv / n) @ mid),
            "w_norm": n}


OUT["final"]["geometry"] = offsets(w, b)
OUT["pocket"]["geometry"] = offsets(pw, pb)
OUT["curve"] = res["curve"]
OUT["best_curve"] = res["best_curve"]
OUT["updates_per_epoch"] = res["updates_per_epoch"]
OUT["total_updates"] = int(sum(res["updates_per_epoch"]))
OUT["min_updates_epoch"] = int(min(res["updates_per_epoch"]))
OUT["curve_min"] = float(min(res["curve"]))
OUT["curve_max"] = float(max(res["curve"]))

# Scale facts behind the analysis in 2D: one mistake moves w by eta*||x|| but
# moves b by only eta, so the normal vector travels about ||x|| times faster
# than the offset.  With the clouds sitting ~5 units from the origin that is a
# factor of roughly five per mistake.
norms = np.linalg.norm(X, axis=1)
OUT["scale"] = {
    "mean_norm": float(norms.mean()),
    "median_norm": float(np.median(norms)),
    "dw_per_mistake": float(ETA * norms.mean()),
    "db_per_mistake": ETA,
    "ratio": float(norms.mean()),
}

# The reference linear rule.  With equal spherical covariances the Bayes-optimal
# linear boundary is the perpendicular bisector of the centres, with error
# Phi(-d / 2 sigma).  That is optimal *for the distributions*; on a finite sample
# some other line can score slightly higher, and the pocket in fact does.
d = float(np.linalg.norm(MU1 - MU0))
sigma = float(np.sqrt(COV[0, 0]))
z = d / (2.0 * sigma)
bayes_acc = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

u = (MU1 - MU0) / d
mid = (MU0 + MU1) / 2.0
bis_w, bis_b = u, float(-u @ mid)
OUT["bayes"] = {
    "d": d, "sigma": sigma, "z": z,
    "acc_theoretical": bayes_acc,
    "acc_empirical": accuracy(X, y, bis_w, bis_b),
    "w": bis_w.tolist(), "b": bis_b,
}

# ------------------------------------------------------------------ Figure 5
pad = 0.6
xlim = (X[:, 0].min() - pad, X[:, 0].max() + pad)
ylim = (X[:, 1].min() - pad, X[:, 1].max() + pad)
xs = np.linspace(xlim[0], xlim[1], 200)


def line(wv, bv):
    return -(wv[0] * xs + bv) / wv[1]


fig, ax = plt.subplots(figsize=(7.6, 5.6))
scatter_classes(ax, alpha=0.4, s=11)
ax.scatter(X[mis_pocket, 0], X[mis_pocket, 1], s=26, facecolors="none",
           edgecolors=SERIES[3], linewidths=0.6, marker="o", alpha=0.5, zorder=4,
           label="Misclassified by pocket (" + str(int(mis_pocket.sum())) + ")")
ax.plot(xs, line(pw, pb), color=SERIES[2], lw=2.6, zorder=8,
        label="Pocket boundary - {0:.2f}%".format(res["pocket_acc"] * 100))
ax.plot(xs, line(w, b), color=INK, lw=2.0, ls="--", zorder=8,
        label="Final boundary - {0:.2f}% ({1} wrong)".format(
            res["acc"] * 100, int(mis_final.sum())))
title(ax, "Figure 5 - Final and pocket boundaries after 100 epochs",
      "The final line has drifted off the data entirely; the pocket line cuts "
      "between the centres")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
# Legend below the axes: at this density every corner of the panel has points in it.
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)
fig.savefig(FIG + "/fig5_boundaries.png")
plt.close(fig)

# ------------------------------------------------------------------ Figure 6
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ep = np.arange(1, len(res["curve"]) + 1)
ax.plot(ep, np.array(res["curve"]) * 100, color=SERIES[0], lw=1.3, alpha=0.85,
        zorder=3, label="Current weights (after each epoch)")
ax.plot(ep, np.array(res["best_curve"]) * 100, color=SERIES[2], lw=2.4, zorder=4,
        label="Best so far (pocket)")
ax.axhline(OUT["bayes"]["acc_empirical"] * 100, color=INK_MUTED, lw=1.0, ls=":",
           zorder=2,
           label="Distribution-optimal line - {0:.2f}%".format(
               OUT["bayes"]["acc_empirical"] * 100))
ax.scatter([res["pocket_epoch"]], [res["pocket_acc"] * 100], s=70, marker="o",
           facecolors="white", edgecolors=SERIES[2], linewidths=2.0, zorder=6)
ax.annotate("pocket best {0:.2f}% at epoch {1}".format(
                res["pocket_acc"] * 100, res["pocket_epoch"]),
            xy=(res["pocket_epoch"], res["pocket_acc"] * 100), xytext=(14, -30),
            textcoords="offset points", fontsize=8.5, color=INK_SOFT,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9))
title(ax, "Figure 6 - Accuracy per epoch: running weights vs. the pocket",
      "The running curve never settles - on non-separable data the updates never stop")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training accuracy (%)")
ax.set_ylim(0, 100)
ax.legend(loc="lower right", fontsize=8.0)
fig.savefig(FIG + "/fig6_pocket.png")
plt.close(fig)

# ----------------------------------------------------------------- Figure 6b
# Figure 6 is flat because it samples once per epoch, and every epoch ends just
# after the class-1 block.  Resolving the same run per *update* shows what is
# really happening: the boundary sweeps across the clouds and back, and the
# pocket exists to catch it mid-sweep.
tr = res["trace"]
acc_tr = np.array([t[2] for t in tr]) * 100
ks = np.array([t[1] for t in tr])
OUT["trace_len"] = len(tr)
OUT["trace"] = [[int(e), int(k), float(a)] for e, k, a in tr]
OUT["trace_stats"] = {
    "above_60": int((acc_tr > 60).sum()),
    "above_70": int((acc_tr > 70).sum()),
    "median": float(np.median(acc_tr)),
}

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.plot(ks, acc_tr, color=SERIES[0], lw=1.2, zorder=4,
        label="Accuracy after each weight update")
ax.axhline(OUT["bayes"]["acc_empirical"] * 100, color=INK_MUTED, lw=1.0, ls=":",
           zorder=2, label="Distribution-optimal line - {0:.2f}%".format(
               OUT["bayes"]["acc_empirical"] * 100))
ax.axhline(50, color=INK_MUTED, lw=0.9, ls="--", zorder=2, label="Chance - 50%")
# Mark where each epoch ends -- exactly the points Figure 6 samples.
ends = np.cumsum(res["updates_per_epoch"])
ax.scatter(ends, np.array(res["curve"]) * 100, s=16, marker="v", c=SERIES[1],
           linewidths=0.0, alpha=0.8, zorder=5,
           label="End of epoch (what Figure 6 records)")
ax.scatter([res["pocket_update"]], [res["pocket_acc"] * 100], s=80, marker="o",
           facecolors="white", edgecolors=SERIES[2], linewidths=2.0, zorder=7)
ax.annotate("pocket best {0:.2f}%".format(res["pocket_acc"] * 100),
            xy=(res["pocket_update"], res["pocket_acc"] * 100), xytext=(16, 12),
            textcoords="offset points", fontsize=8.5, color=INK_SOFT,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9))
title(ax, "Figure 6b - The same run resolved per update, all 100 epochs",
      "Every epoch sweeps the boundary through the clouds and back; the end-of-epoch "
      "samples always land in the trough")
ax.set_xlabel("Cumulative weight update")
ax.set_ylabel("Training accuracy (%)")
ax.set_ylim(40, 80)
ax.legend(loc="lower right", fontsize=8.0, ncol=2)
fig.savefig(FIG + "/fig6b_updates.png")
plt.close(fig)

json.dump(OUT, open(RES + "/perceptron2.json", "w"), indent=2)

print("final  -> w={0}, b={1:.4f}, acc={2:.2f}%, epochs={3}, converged={4}".format(
    w, b, res["acc"] * 100, res["epochs"], res["converged"]))
print("pocket -> w={0}, b={1:.4f}, acc={2:.2f}% at epoch {3}".format(
    pw, pb, res["pocket_acc"] * 100, res["pocket_epoch"]))
print("final predicts class 1 for {0:.1%} of the set".format(
    OUT["final"]["predicted_class1_fraction"]))
print("curve range {0:.2f}% .. {1:.2f}%, min updates in an epoch: {2}".format(
    OUT["curve_min"] * 100, OUT["curve_max"] * 100, OUT["min_updates_epoch"]))
print("linear ceiling: {0:.2f}% theoretical, {1:.2f}% on this sample".format(
    bayes_acc * 100, OUT["bayes"]["acc_empirical"] * 100))
print("mean ||x|| = {0:.3f}  ->  |dw| = {1:.4f} vs |db| = {2:.4f} per mistake".format(
    OUT["scale"]["mean_norm"], OUT["scale"]["dw_per_mistake"], ETA))
print("offset -b/||w||: final {0:.2f} (needs {1:.2f}), pocket {2:.2f} (needs {3:.2f})".format(
    OUT["final"]["geometry"]["offset"], OUT["final"]["geometry"]["needed"],
    OUT["pocket"]["geometry"]["offset"], OUT["pocket"]["geometry"]["needed"]))
print("total updates over 100 epochs: {0}".format(OUT["total_updates"]))
