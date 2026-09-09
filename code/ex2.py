"""Exercise 2 -- Non-linearity in higher dimensions (5D)."""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from style import apply_style, title, SERIES, MARKERS, SURFACE, INK, INK_SOFT, INK_MUTED

apply_style()
FIG = "../exercises/data/figures"
rng = np.random.default_rng(42)
N = 500

# ------------------------------------------------ Dataset I: shifted Gaussians
mu_A = np.zeros(5)
mu_B = np.full(5, 1.5)
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0]])
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5]])

eig = {}
for name, S in [("Sigma_A", Sigma_A), ("Sigma_B", Sigma_B)]:
    ev = np.linalg.eigvalsh(S)
    assert np.allclose(S, S.T), f"{name} not symmetric"
    eig[name] = {"eigenvalues": ev.tolist(), "min": float(ev.min()),
                 "positive_definite": bool((ev > 0).all())}
    print(f"{name}: symmetric, eigenvalues {np.round(ev, 4).tolist()}, "
          f"positive definite = {bool((ev > 0).all())}")

XA = rng.multivariate_normal(mu_A, Sigma_A, size=N)
XB = rng.multivariate_normal(mu_B, Sigma_B, size=N)
X_I = np.vstack([XA, XB])
y_I = np.repeat([0, 1], N)

# ------------------------------------------------ Dataset II: concentric shells
# Directions uniform on the unit 5-sphere; radius sets which shell a point lands on.
# N(2.0, 0.4) and N(5.0, 0.4) are read as mean and STANDARD DEVIATION.
def shell(radius_mean, radius_sd):
    v = rng.standard_normal((N, 5))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    rho = rng.normal(radius_mean, radius_sd, size=N)
    return u * rho[:, None]

XC = shell(2.0, 0.4)
XD = shell(5.0, 0.4)
X_II = np.vstack([XC, XD])
y_II = np.repeat([0, 1], N)

DATASETS = {
    "I": dict(X=X_I, y=y_I, names=("Class A", "Class B"),
              label="Dataset I - shifted Gaussians"),
    "II": dict(X=X_II, y=y_II, names=("Class C (core)", "Class D (shell)"),
               label="Dataset II - concentric shells"),
}

# ------------------------------------------------------------------ metrics
res = {}
for key, d in DATASETS.items():
    X, y = d["X"], d["y"]
    c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
    pca = PCA(n_components=5).fit(X)
    evr = pca.explained_variance_ratio_
    d["proj"] = pca.transform(X)
    d["evr"] = evr
    r0 = np.linalg.norm(X[y == 0], axis=1)
    r1 = np.linalg.norm(X[y == 1], axis=1)
    d["radii"] = (r0, r1)
    res[key] = {
        "center_0": c0.tolist(), "center_1": c1.tolist(),
        "center_distance": float(np.linalg.norm(c0 - c1)),
        "center_norm_0": float(np.linalg.norm(c0)),
        "center_norm_1": float(np.linalg.norm(c1)),
        "evr": evr.tolist(),
        "evr_pc1": float(evr[0]), "evr_pc2": float(evr[1]),
        "evr_pc1_pc2": float(evr[:2].sum()),
        "radius_mean_0": float(r0.mean()), "radius_sd_0": float(r0.std(ddof=1)),
        "radius_mean_1": float(r1.mean()), "radius_sd_1": float(r1.std(ddof=1)),
        "radius_max_0": float(r0.max()), "radius_min_1": float(r1.min()),
    }
res["I"]["center_distance_theoretical"] = float(np.linalg.norm(mu_A - mu_B))
res["eigen"] = eig

# a radius threshold midway between the two shells, for Dataset II
thr = (res["II"]["radius_mean_0"] + res["II"]["radius_mean_1"]) / 2
r0, r1 = DATASETS["II"]["radii"]
res["II"]["radius_threshold"] = float(thr)
res["II"]["radius_rule_accuracy"] = float(((r0 < thr).sum() + (r1 >= thr).sum()) / (2 * N))

# ---------------------------------------------------------------- Figure 4
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
for ax, (key, d) in zip(axes, DATASETS.items()):
    P = d["proj"]
    for k in (0, 1):
        pts = P[d["y"] == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=20, c=SERIES[k], marker=MARKERS[k],
                   alpha=0.62, linewidths=0.3, edgecolors=SURFACE, label=d["names"][k])
    e = d["evr"]
    title(ax, f"{d['label']}",
          f"PC1 {e[0]*100:.1f}% + PC2 {e[1]*100:.1f}% = {e[:2].sum()*100:.1f}% of variance")
    ax.set_xlabel(f"PC1 ({e[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({e[1]*100:.1f}%)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")
fig.suptitle("Figure 4 - Both 5D datasets projected onto their first two principal components",
             x=0.045, ha="left", fontsize=12, color=INK, weight="600")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{FIG}/fig4_pca.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
for ax, (key, d) in zip(axes, DATASETS.items()):
    ra, rb = d["radii"]
    bins = np.linspace(0, max(ra.max(), rb.max()) * 1.05, 46)
    for k, r in enumerate((ra, rb)):
        ax.hist(r, bins=bins, color=SERIES[k], alpha=0.62, label=d["names"][k],
                edgecolor=SURFACE, linewidth=0.6)
    if key == "II":
        ax.axvline(thr, color=INK_SOFT, lw=1.4, ls="--", zorder=5)
        ax.annotate(f"threshold {thr:.2f}", xy=(thr, ax.get_ylim()[1] * 0.55),
                    xytext=(7, 0), textcoords="offset points", fontsize=8.5,
                    color=INK_SOFT, ha="left")
    title(ax, d["label"], r"distribution of the radius $\|x\|$")
    ax.set_xlabel(r"$\|x\|$")
    ax.set_ylabel("count")
    ax.legend(loc="upper right")
fig.suptitle("Figure 5 - Radius histograms: overlapping for Dataset I, disjoint for Dataset II",
             x=0.045, ha="left", fontsize=12, color=INK, weight="600")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{FIG}/fig5_radius.png")
plt.close(fig)

json.dump(res, open("../results/ex2.json", "w"), indent=2)

for key in ("I", "II"):
    r = res[key]
    print(f"\nDataset {key}")
    print(f"  centre distance      : {r['center_distance']:.4f}")
    print(f"  ||c0||, ||c1||       : {r['center_norm_0']:.4f}, {r['center_norm_1']:.4f}")
    print(f"  EVR PC1, PC2, sum    : {r['evr_pc1']*100:.2f}%, {r['evr_pc2']*100:.2f}%, "
          f"{r['evr_pc1_pc2']*100:.2f}%")
    print(f"  radius c0 mean+-sd   : {r['radius_mean_0']:.4f} +- {r['radius_sd_0']:.4f}")
    print(f"  radius c1 mean+-sd   : {r['radius_mean_1']:.4f} +- {r['radius_sd_1']:.4f}")
    print(f"  max r(c0), min r(c1) : {r['radius_max_0']:.4f}, {r['radius_min_1']:.4f}")
print(f"\nDataset I theoretical centre distance: {res['I']['center_distance_theoretical']:.4f}")
print(f"Dataset II radius rule accuracy: {res['II']['radius_rule_accuracy']*100:.2f}%")
