"""Exercise 1 -- Point clouds: geometry and spread in 2D."""

import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from style import apply_style, title, SERIES, MARKERS, SURFACE, INK, INK_SOFT, INK_MUTED

apply_style()
FIG = "../exercises/data/figures"
OUT = {}

MU = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
SIGMA = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
LABELS = [f"Class {k}" for k in range(4)]

rng = np.random.default_rng(42)

# One pool of standard-normal deviates, reused for every scale factor.  Each
# dataset is then x = mu + (s * sigma) * z, so the panels differ only by spread
# and s = 1 reproduces Figure 1 exactly.
Z = rng.standard_normal((4, N_PER_CLASS, 2))
y = np.repeat(np.arange(4), N_PER_CLASS)


def sample(s):
    return (MU[:, None, :] + (s * SIGMA)[:, None, :] * Z).reshape(-1, 2)


X1 = sample(1.0)


def scatter_classes(ax, X, labels=True):
    for k in range(4):
        pts = X[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=26, c=SERIES[k], marker=MARKERS[k],
                   alpha=0.75, linewidths=0.4, edgecolors=SURFACE,
                   label=LABELS[k] if labels else None, zorder=3)
    for k in range(4):
        ax.scatter(*MU[k], s=190, marker="X", c=SERIES[k], edgecolors="white",
                   linewidths=1.6, zorder=5)


# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(figsize=(7.6, 5.0))
scatter_classes(ax, X1)
for k in range(4):
    ax.annotate(rf"$\mu_{k}$ = ({MU[k,0]:.0f}, {MU[k,1]:.0f})",
                xy=MU[k], xytext=(0, 13), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK, weight="600",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=SERIES[k],
                          lw=0.9, alpha=0.92), zorder=6)
title(ax, "Figure 1 - Four Gaussian point clouds at s = 1",
      "400 points (100 per class); X marks each class centre")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_ylim(top=X1[:, 1].max() + 2.6)
ax.legend(loc="upper left", ncol=2)
fig.savefig(f"{FIG}/fig1_scatter.png")
plt.close(fig)

# ------------------------------------------------- Figure 1b: boundaries
fig, ax = plt.subplots(figsize=(7.6, 5.0))
xs = np.linspace(-2, 19, 700)
ys = np.linspace(-6, 14, 700)
gx, gy = np.meshgrid(xs, ys)
grid = np.column_stack([gx.ravel(), gy.ravel()])
owner = np.argmin(((grid[:, None, :] - MU[None]) ** 2).sum(-1), axis=1).reshape(gx.shape)
ax.contourf(gx, gy, owner, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=SERIES, alpha=0.10, zorder=0)
ax.contour(gx, gy, owner, levels=[0.5, 1.5, 2.5], colors=[INK_SOFT], linewidths=1.4,
           linestyles="--", zorder=2)
scatter_classes(ax, X1)
title(ax, "Figure 1b - Piecewise-linear boundaries sketched on the s = 1 clouds",
      "Nearest-centroid (Voronoi) partition: three straight cuts, no single line")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim(-2, 19)
ax.set_ylim(-6, 14)
ax.legend(loc="upper left", ncol=2)
fig.savefig(f"{FIG}/fig1b_boundaries.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
scales = [0.5, 1.0, 2.0, 4.0]
datasets = {s: sample(s) for s in scales}
allpts = np.vstack(list(datasets.values()))
pad = 1.0
xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)

fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), scales):
    scatter_classes(ax, datasets[s], labels=(s == 0.5))
    ax.set_title(f"s = {s}", loc="left", color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
for ax in axes[1]:
    ax.set_xlabel("$x_1$")
for ax in axes[:, 0]:
    ax.set_ylabel("$x_2$")
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Figure 2 - Same centres, standard deviations scaled by s (shared axes)",
             x=0.065, ha="left", fontsize=12, color=INK, weight="600")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(f"{FIG}/fig2_scales.png")
plt.close(fig)

# ------------------------------------------------- separation ratios r_ij
sigma_bar = SIGMA.mean(axis=1)
rows = []
for i, j in itertools.combinations(range(4), 2):
    d = float(np.linalg.norm(MU[i] - MU[j]))
    r = d / (sigma_bar[i] + sigma_bar[j])
    rows.append({"pair": f"{i}-{j}", "i": i, "j": j, "dist": d,
                 "sum_sigma_bar": float(sigma_bar[i] + sigma_bar[j]),
                 "r_s1": r, "r_s2": r / 2.0})
worst = min(rows, key=lambda r_: r_["r_s1"])

# ------------------------------------------------------------ mixing rate
mixing = {}
for s in scales:
    X = datasets[s]
    nearest = np.argmin(((X[:, None, :] - MU[None]) ** 2).sum(-1), axis=1)
    mixing[s] = float((nearest != y).mean())

# ---------------------------------------------------------------- Figure 3
fig, ax = plt.subplots(figsize=(7.0, 4.4))
xs_ = np.array(scales)
ys_ = np.array([mixing[s] * 100 for s in scales])
ax.plot(xs_, ys_, color=SERIES[0], lw=2.0, zorder=3)
ax.scatter(xs_, ys_, s=64, color=SERIES[0], edgecolors=SURFACE, linewidths=1.4, zorder=4)
for xv, yv in zip(xs_, ys_):
    ax.annotate(f"{yv:.2f}%", xy=(xv, yv), xytext=(0, 11), textcoords="offset points",
                ha="center", fontsize=9, color=INK, weight="600")
title(ax, "Figure 3 - Mixing rate against spread factor s",
      "Share of points whose nearest class centre is not their own")
ax.set_xlabel("spread factor $s$")
ax.set_ylabel("mixing rate (%)")
ax.set_xticks(xs_)
ax.set_xticklabels([str(s) for s in scales])
ax.set_ylim(-2, max(ys_) * 1.28 + 2)
fig.savefig(f"{FIG}/fig3_mixing.png")
plt.close(fig)

OUT = {"sigma_bar": sigma_bar.tolist(), "pairs": rows, "worst": worst,
       "mixing": {str(k): v for k, v in mixing.items()}}
json.dump(OUT, open("../results/ex1.json", "w"), indent=2)

print("sigma_bar:", np.round(sigma_bar, 4).tolist())
for r_ in rows:
    print(f"  r_{r_['pair']}: d={r_['dist']:.4f} sum={r_['sum_sigma_bar']:.2f} "
          f"r(s=1)={r_['r_s1']:.4f}  r(s=2)={r_['r_s2']:.4f}")
print("worst pair:", worst["pair"], round(worst["r_s1"], 4), "-> s=2:", round(worst["r_s2"], 4))
for s in scales:
    print(f"  mixing s={s}: {mixing[s]*100:.2f}%  ({int(mixing[s]*400)} of 400)")
