"""Exercise 1 -- A perceptron on linearly separable data.  Figures 1, 2, 3."""

import json
import numpy as np
import matplotlib.pyplot as plt
from style import apply_style, title, SERIES, INK, INK_SOFT, INK_MUTED
from perceptron import fit, predict, accuracy

apply_style()
FIG = "../exercises/perceptron/figures"
OUT = {}

MU0, MU1 = np.array([1.5, 1.5]), np.array([5.0, 5.0])
COV = np.array([[0.5, 0.0], [0.0, 0.5]])
N_PER_CLASS = 1000
ETA, MAX_EPOCHS = 0.01, 100

# One generator, drawn from in a fixed documented order: class 0, class 1, then
# the weight initialisation.  Seeding here rather than sharing a generator
# across files is what lets this script be re-run on its own and still produce
# byte-identical numbers.
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ A: data
X0 = rng.multivariate_normal(MU0, COV, N_PER_CLASS)
X1 = rng.multivariate_normal(MU1, COV, N_PER_CLASS)
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(N_PER_CLASS, int), np.ones(N_PER_CLASS, int)])

# Deliberately NOT shuffled: the statement fixes the construction as one class
# stacked on the other, and that sample order is part of what the two exercises
# end up contrasting (in Exercise 2 the tail of every epoch is all one class).
# On separable data the order cannot change the final outcome.
OUT["n_per_class"] = N_PER_CLASS
OUT["mu0"], OUT["mu1"] = MU0.tolist(), MU1.tolist()
OUT["cov"] = COV.tolist()
OUT["centre_distance"] = float(np.linalg.norm(MU1 - MU0))
OUT["sigma"] = float(np.sqrt(COV[0, 0]))

# Bias starts at 0; the weights start small but non-zero, which is exactly the
# condition under which eta is observable at all (Section 1D).
w0 = rng.normal(0.0, 0.01, size=2)
OUT["w_init"] = w0.tolist()
OUT["b_init"] = 0.0
OUT["eta"], OUT["max_epochs"] = ETA, MAX_EPOCHS


# ------------------------------------------------------------------ Figure 1
def scatter_classes(ax, alpha=0.55, s=13):
    ax.scatter(X0[:, 0], X0[:, 1], s=s, c=SERIES[0], marker="o", alpha=alpha,
               linewidths=0.0, label="Class 0", zorder=3)
    ax.scatter(X1[:, 0], X1[:, 1], s=s, c=SERIES[1], marker="s", alpha=alpha,
               linewidths=0.0, label="Class 1", zorder=3)
    for mu, c in ((MU0, SERIES[0]), (MU1, SERIES[1])):
        ax.scatter(mu[0], mu[1], s=170, marker="X", c=c, edgecolors="white",
                   linewidths=1.6, zorder=6)


fig, ax = plt.subplots(figsize=(7.4, 5.2))
scatter_classes(ax)
for mu, lab in ((MU0, r"$\mu_0$ = (1.5, 1.5)"), (MU1, r"$\mu_1$ = (5, 5)")):
    ax.annotate(lab, xy=mu, xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK, weight="600",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=INK_SOFT,
                          lw=0.8, alpha=0.92), zorder=7)
title(ax, "Figure 1 - Two linearly separable Gaussian classes",
      "2000 points (1000 per class), covariance 0.5I; X marks each class centre")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(loc="upper left")
fig.savefig(FIG + "/fig1_scatter.png")
plt.close(fig)

# ---------------------------------------------------------- B, C: training
res = fit(X, y, w0, eta=ETA, max_epochs=MAX_EPOCHS)
w, b = res["w"], res["b"]
OUT["final"] = {
    "w": w.tolist(), "b": b, "acc": res["acc"], "epochs": res["epochs"],
    "converged": res["converged"], "curve": res["curve"],
    "updates_per_epoch": res["updates_per_epoch"],
    "total_updates": int(sum(res["updates_per_epoch"])),
    "w_dir": (w / np.linalg.norm(w)).tolist(),
    "w_norm": float(np.linalg.norm(w)),
}

pred = predict(X, w, b)
mis = pred != y
OUT["final"]["n_misclassified"] = int(mis.sum())

# Geometry behind "why so fast".  In augmented coordinates xa = [x, 1] the
# perceptron is homogeneous, and Novikoff bounds the total number of mistakes by
# (R / gamma)^2 for *any* separating hyperplane -- so we are free to pick a
# convenient witness rather than the one the algorithm happened to find.  The
# perpendicular bisector of the two class means is such a witness, and because
# any valid separator yields a valid bound, its margin gives a concrete finite
# one.  (Using the perceptron's own solution instead would be useless: it stops
# at the first consistent hyperplane, so its attained margin can be arbitrarily
# close to zero -- reported below as the contrast.)
Xa = np.hstack([X, np.ones((X.shape[0], 1))])
signed = 2 * y - 1
R = float(np.max(np.linalg.norm(Xa, axis=1)))

u = (MU1 - MU0) / np.linalg.norm(MU1 - MU0)   # unit normal, along the centre line
proj = X @ u

# The physical gap along u: how far the nearest class-1 point sits above the
# farthest class-0 point.  Offsetting the witness to the middle of *that* gap
# (rather than to the midpoint of the means, which nearly grazes class 0) is
# what makes it the widest-margin separator with this normal.
lo, hi = float(proj[y == 0].max()), float(proj[y == 1].min())
gap = hi - lo
wa_star = np.append(u, -(lo + hi) / 2.0)
gamma_star = float((signed * (Xa @ wa_star) / np.linalg.norm(wa_star)).min())

wa = np.append(w, b)
gamma_hat = float((signed * (Xa @ wa) / np.linalg.norm(wa)).min())

# Scale facts used in the eta discussion: how big one update is next to the
# initial weight vector.
norms = np.linalg.norm(X, axis=1)
OUT["scale"] = {
    "mean_norm": float(norms.mean()),
    "w_init_norm": float(np.linalg.norm(w0)),
    "sigma_multiple": float(np.linalg.norm(MU1 - MU0) / np.sqrt(COV[0, 0])),
}
OUT["scale"]["dw_eta001"] = float(ETA * norms.mean())
OUT["scale"]["dw_eta1"] = float(1.0 * norms.mean())
OUT["scale"]["ratio_eta001"] = OUT["scale"]["dw_eta001"] / OUT["scale"]["w_init_norm"]
OUT["scale"]["ratio_eta1"] = OUT["scale"]["dw_eta1"] / OUT["scale"]["w_init_norm"]

OUT["margin"] = {
    "R": R,
    "gamma_star": gamma_star,                 # margin of the gap-centred witness
    "gamma_attained": gamma_hat,              # margin the perceptron settled for
    "novikoff_bound": float((R / gamma_star) ** 2),
    "projection_gap": gap,
    "half_gap": gap / 2.0,
    "sample_presentations": int(X.shape[0] * res["epochs"]),
    "update_rate": float(OUT["final"]["total_updates"] / (X.shape[0] * res["epochs"])),
}


# ------------------------------------------------------------------ Figure 2
def boundary_xy(wv, bv, xs):
    """The line w.x + b = 0 rewritten as x2 = -(w1 x1 + b) / w2."""
    return -(wv[0] * xs + bv) / wv[1]


pad = 0.6
xlim = (X[:, 0].min() - pad, X[:, 0].max() + pad)
ylim = (X[:, 1].min() - pad, X[:, 1].max() + pad)
xs = np.linspace(xlim[0], xlim[1], 200)

fig, ax = plt.subplots(figsize=(7.4, 5.2))
scatter_classes(ax, alpha=0.45, s=12)
ax.plot(xs, boundary_xy(w, b, xs), color=INK, lw=2.0, zorder=5,
        label=r"Decision boundary  $\mathbf{w}\cdot\mathbf{x}+b=0$")
if mis.any():
    ax.scatter(X[mis, 0], X[mis, 1], s=90, facecolors="none", edgecolors=SERIES[3],
               linewidths=1.8, marker="o", zorder=7,
               label="Misclassified (" + str(int(mis.sum())) + ")")
else:
    ax.plot([], [], " ", label="Misclassified: none (0 of 2000)")
title(ax, "Figure 2 - Learned decision boundary on the separable data",
      "eta = {0}, converged after {1} epochs; accuracy {2:.2f}%".format(
          ETA, res["epochs"], res["acc"] * 100))
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.legend(loc="upper left")
fig.savefig(FIG + "/fig2_boundary.png")
plt.close(fig)

# ------------------------------------------------- D: the eta = 1.0 re-run
# Same data, same starting weights -- only eta changes, so any difference is
# attributable to the learning rate alone.
res_hi = fit(X, y, w0, eta=1.0, max_epochs=MAX_EPOCHS)
w_hi, b_hi = res_hi["w"], res_hi["b"]
d_lo = w / np.linalg.norm(w)
d_hi = w_hi / np.linalg.norm(w_hi)
cos = float(np.clip(d_lo @ d_hi, -1.0, 1.0))
OUT["eta1"] = {
    "eta": 1.0, "w": w_hi.tolist(), "b": b_hi, "acc": res_hi["acc"],
    "epochs": res_hi["epochs"], "converged": res_hi["converged"],
    "curve": res_hi["curve"], "w_dir": d_hi.tolist(),
    "w_norm": float(np.linalg.norm(w_hi)),
    "total_updates": int(sum(res_hi["updates_per_epoch"])),
    "cos_with_eta001": cos,
    "angle_deg": float(np.degrees(np.arccos(cos))),
    "norm_ratio": float(np.linalg.norm(w_hi) / np.linalg.norm(w)),
}

# Numerical check of the algebra in 1D-(iii): started from w = 0, b = 0, two
# learning rates must produce weights differing by exactly the factor
# eta_hi/eta_lo, with identical epoch counts and identical accuracy curves.
zero = np.zeros(2)
z_lo = fit(X, y, zero, eta=ETA, max_epochs=MAX_EPOCHS)
z_hi = fit(X, y, zero, eta=1.0, max_epochs=MAX_EPOCHS)
scale = 1.0 / ETA
resid = np.append(z_hi["w"] - scale * z_lo["w"], z_hi["b"] - scale * z_lo["b"])
OUT["zero_init"] = {
    "eta_lo": ETA, "eta_hi": 1.0, "scale": scale,
    "epochs_lo": z_lo["epochs"], "epochs_hi": z_hi["epochs"],
    "acc_lo": z_lo["acc"], "acc_hi": z_hi["acc"],
    "w_lo": z_lo["w"].tolist(), "b_lo": z_lo["b"],
    "w_hi": z_hi["w"].tolist(), "b_hi": z_hi["b"],
    "max_abs_residual": float(np.max(np.abs(resid))),
    "same_curve": z_lo["curve"] == z_hi["curve"],
}

# ------------------------------------------------------------------ Figure 3
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ep_lo = np.arange(1, len(res["curve"]) + 1)
ep_hi = np.arange(1, len(res_hi["curve"]) + 1)
a0 = accuracy(X, y, w0, 0.0)
ax.plot(np.r_[0, ep_lo], np.r_[a0, res["curve"]] * 100, marker="o", ms=5.5,
        color=SERIES[0], lw=2.0, zorder=4, label="eta = 0.01")
ax.plot(np.r_[0, ep_hi], np.r_[a0, res_hi["curve"]] * 100, marker="s", ms=5.0,
        color=SERIES[1], lw=1.6, ls="--", zorder=3, label="eta = 1.0")
ax.axhline(100, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
ax.annotate("converged, epoch {0}".format(res["epochs"]),
            xy=(res["epochs"], res["acc"] * 100), xytext=(-96, -34),
            textcoords="offset points", fontsize=8.5, color=INK_SOFT,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9))
title(ax, "Figure 3 - Training accuracy after each epoch",
      "Epoch 0 is the accuracy of the initial weights, before any update")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training accuracy (%)")
ax.set_xticks(np.arange(0, max(len(ep_lo), len(ep_hi)) + 2, 2))
ax.legend(loc="lower right")
fig.savefig(FIG + "/fig3_accuracy.png")
plt.close(fig)

json.dump(OUT, open("../results/perceptron1.json", "w"), indent=2)

print("eta=0.01 -> w={0}, b={1:.4f}, epochs={2}, acc={3:.2f}%".format(
    w, b, res["epochs"], res["acc"] * 100))
print("eta=1.00 -> w={0}, b={1:.4f}, epochs={2}, acc={3:.2f}%".format(
    w_hi, b_hi, res_hi["epochs"], res_hi["acc"] * 100))
print("angle between directions: {0:.4f} deg".format(OUT["eta1"]["angle_deg"]))
print("zero-init residual: {0:.3e}, epochs {1} vs {2}, same curve: {3}".format(
    OUT["zero_init"]["max_abs_residual"], z_lo["epochs"], z_hi["epochs"],
    OUT["zero_init"]["same_curve"]))
print("gamma*={0:.4f} (bisector), gamma_attained={1:.2e}, R={2:.4f}".format(
    gamma_star, gamma_hat, R))
print("Novikoff bound={0:.1f} mistakes, actual updates={1}, gap={2:.4f}".format(
    OUT["margin"]["novikoff_bound"], OUT["final"]["total_updates"], gap))
