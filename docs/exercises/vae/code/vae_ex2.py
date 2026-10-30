"""Exercise 2 -- a latent space you can see (d = 2), and interpolation.

Produces Figures 4, 5 and 6 and writes ``results/vae2.json``.

The bottom row of Figure 6 needs the d = 16 model of Exercise 1, so run that
script first; this one loads ``vae_d16.pt``.

    python vae_ex1.py && python vae_ex2.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier

import style
from mnist_data import load_mnist, test_matrix
from vae import VAE, evaluate, train_vae

D = 2
D16 = 16
EPOCHS = 20
LR = 1e-3
BATCH = 128
BETA = 1.0
SEED = 42
GRID = 15                      # 15 x 15 decoded plane
Q_LO, Q_HI = 0.05, 0.95        # the quantile range the statement fixes

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
CKPT = os.path.join(HERE, "vae_d16.pt")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def figure4(mu, labels, centroids):
    """Scatter of mu for all 10 000 test images, coloured by digit.

    Ten classes is more than colour can carry on its own, so identity is
    encoded three times over: hue, marker shape, and a direct label at each
    class centroid (see the note in ``style.py``).  Which two digits share a
    hue is chosen by ``pair_digits_by_distance`` so that overlapping classes
    never do.  The right panel is the same data as ten small multiples -- the
    readable view of where each class actually sits, which is what the analysis
    in 2E has to read off.
    """
    assign, pairs, min_sep = style.pair_digits_by_distance(centroids)

    fig = plt.figure(figsize=(12.6, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.30)

    # -- left: all ten classes on one pair of axes (what the statement asks) --
    ax = fig.add_subplot(gs[0, 0])
    handles = []
    for digit in range(10):
        colour, marker = assign[digit]
        m = labels == digit
        ax.scatter(mu[m, 0], mu[m, 1], s=4, marker=marker, c=colour,
                   alpha=0.45, linewidths=0, rasterized=True)
        handles.append(Line2D([], [], marker=marker, color=colour, ls="",
                              ms=6, label=str(digit)))
    for digit, (cx, cy) in enumerate(centroids):
        # Direct label: the ring keeps it readable over any class underneath.
        ax.text(cx, cy, str(digit), fontsize=11, fontweight="bold",
                color=style.INK, ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.14", fc=style.SURFACE,
                          ec=style.GRID, lw=0.8, alpha=0.92))
    style.title(ax, "All 10 000 encoded test images",
                "each point is the mean of q(z|x); the ringed number is the "
                "class centroid")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.legend(handles=handles, title="digit", ncol=1, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), markerscale=1.2)

    # -- right: ten small multiples, one class against the rest in grey -------
    inner = gs[0, 1].subgridspec(2, 5, wspace=0.10, hspace=0.30)
    lo = float(np.percentile(mu, 0.2)) * 1.05
    hi = float(np.percentile(mu, 99.8)) * 1.05
    for digit in range(10):
        axd = fig.add_subplot(inner[digit // 5, digit % 5])
        colour, marker = assign[digit]
        axd.scatter(mu[:, 0], mu[:, 1], s=1.0, c=style.GRID, linewidths=0,
                    rasterized=True)
        m = labels == digit
        axd.scatter(mu[m, 0], mu[m, 1], s=1.6, marker=marker, c=colour,
                    linewidths=0, rasterized=True)
        axd.set_title(str(digit), fontsize=9.5, color=style.INK, pad=3)
        axd.set_xlim(lo, hi)
        axd.set_ylim(lo, hi)
        axd.set_xticks([])
        axd.set_yticks([])
        axd.grid(False)
    fig.text(0.575, 0.90, "The same data, one class at a time",
             fontsize=11, color=style.INK, fontweight="600")
    fig.text(0.575, 0.868, "the highlighted class against all other test "
                           "points in grey; shared axes",
             fontsize=8.5, color=style.INK_MUTED)

    fig.suptitle("Figure 4 — the d = 2 latent space of the test set",
                 x=0.008, ha="left", fontsize=12.5, color=style.INK)
    fig.subplots_adjust(top=0.80, left=0.055, right=0.985, bottom=0.085)
    fig.savefig(os.path.join(FIG, "fig4_latent.png"), dpi=200)
    plt.close(fig)
    return pairs, min_sep


def figure5(model, z_coords):
    """Decode a GRID x GRID plane of latent points on the Phi^-1 grid.

    ``z_coords`` are inverse-normal-CDF positions, so each cell of the picture
    covers the same amount of prior probability.
    """
    # The first tiled row is the largest z2, so the picture reads like an axis
    # that points up.
    zs = [[z1, z2] for z2 in reversed(z_coords) for z1 in z_coords]
    zs = torch.tensor(zs, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        imgs = model.decode(zs).view(GRID, GRID, 28, 28).numpy()

    # (row, col, 28, 28) -> one (15*28, 15*28) canvas
    canvas = imgs.transpose(0, 2, 1, 3).reshape(GRID * 28, GRID * 28)

    fig, ax = plt.subplots(figsize=(7.4, 7.8))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1,
              extent=[0, GRID, 0, GRID], interpolation="nearest")
    ax.grid(False)
    ticks = np.arange(GRID) + 0.5
    keep = [0, 3, 7, 11, 14]
    ax.set_xticks(ticks[keep])
    ax.set_xticklabels([f"{z_coords[i]:.2f}" for i in keep])
    ax.set_yticks(ticks[keep])
    ax.set_yticklabels([f"{z_coords[i]:.2f}" for i in keep])
    ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$")
    ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$")
    for spine in ax.spines.values():
        spine.set_visible(False)
    style.title(ax, "Figure 5 — the decoded plane",
                f"{GRID}x{GRID} latent points at quantiles q = {Q_LO} … "
                f"{Q_HI}, equal prior mass per cell")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_manifold.png"))
    plt.close(fig)
    return imgs


def figure6(model16, x_a, x_b, idx_a, idx_b, dig_a, dig_b, n=10):
    """Pixel-space vs latent-space interpolation between two test images."""
    ts = np.linspace(0.0, 1.0, n)

    # Top row: the straight average of the two images, pixel by pixel.
    pixel = [(1 - t) * x_a + t * x_b for t in ts]

    # Bottom row: average the two mu in the d = 16 latent space, then decode.
    model16.eval()
    with torch.no_grad():
        mu, _ = model16.encode(torch.stack([x_a, x_b]))
        latent = [model16.decode(((1 - t) * mu[0] + t * mu[1]).unsqueeze(0))[0]
                  for t in ts]

    fig, axes = plt.subplots(2, n, figsize=(10.5, 2.9))
    for col, t in enumerate(ts):
        for row, img in ((0, pixel[col]), (1, latent[col])):
            ax = axes[row][col]
            ax.imshow(img.view(28, 28).detach().numpy(), cmap="gray",
                      vmin=0, vmax=1)
            style.image_ax(ax)
        axes[0][col].set_title(f"t = {t:.2f}", fontsize=8,
                               color=style.INK_SOFT, pad=4)
    axes[0][0].set_ylabel("pixel space", fontsize=8.5, color=style.INK_SOFT)
    axes[1][0].set_ylabel("latent, d = 16", fontsize=8.5, color=style.INK_SOFT)
    fig.suptitle(f"Figure 6 — interpolating test image {idx_a} (a {dig_a}) "
                 f"into test image {idx_b} (a {dig_b})",
                 x=0.012, ha="left", fontsize=11, color=style.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(os.path.join(FIG, "fig6_interp.png"))
    plt.close(fig)

    # A number for the analysis: how much of each frame sits at mid-grey.  A
    # pixel near 0 or 1 is committed; one near 0.5 is a ghost.
    def midness(seq):
        v = np.stack([t.detach().numpy() for t in seq])
        return float(np.mean(np.minimum(v, 1.0 - v)))

    return (midness(pixel), midness(latent),
            midness(pixel[4:6]), midness(latent[4:6]))


# --------------------------------------------------------------------------- #
# A read-out probe (not part of the model)
# --------------------------------------------------------------------------- #

def knn_probe(mu_fit, y_fit, mu_eval, y_eval, k=10):
    """A plain k-NN on mu: how much class information survives in the code.

    Purely a read-out probe -- nothing here trains or changes the VAE, and it
    is fitted on the validation split, never on the test set.  It turns "which
    digits overlap" into a confusion count instead of an impression.
    """
    clf = KNeighborsClassifier(n_neighbors=k).fit(mu_fit, y_fit)
    pred = clf.predict(mu_eval)
    acc = float((pred == y_eval).mean())
    pairs = {}
    for true, got in zip(y_eval, pred):
        if true != got:
            key = tuple(sorted((int(true), int(got))))
            pairs[key] = pairs.get(key, 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:6]
    return acc, [{"pair": list(p), "count": c} for p, c in top]


# --------------------------------------------------------------------------- #

def main():
    style.apply_style()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    if not os.path.exists(CKPT):
        raise SystemExit(f"{CKPT} is missing — run `python vae_ex1.py` first, "
                         "the bottom row of Figure 6 needs the d = 16 model.")

    train_loader, val_loader, test_loader, test_set = load_mnist(BATCH)

    print(f"Training VAE with d = {D} — same data, seed, beta and budget as "
          f"Exercise 1")
    model, history, secs = train_vae(
        D, train_loader, val_loader, epochs=EPOCHS, lr=LR, beta=BETA, seed=SEED)
    print(f"  done in {secs:.1f} s")

    # First and only use of the test set for this model.
    test_recon, test_kl = evaluate(model, test_loader)
    print(f"TEST  recon {test_recon:.3f}  kl {test_kl:.3f}  "
          f"-ELBO {test_recon + test_kl:.3f}  nats/image")

    # ---- Figure 4: the encoded test set ----------------------------------- #
    x_test, y_test = test_matrix(test_set)
    model.eval()
    with torch.no_grad():
        mu_test, _ = model.encode(x_test)
    mu_np = mu_test.numpy()
    y_np = y_test.numpy()
    centroids = np.stack([mu_np[y_np == d].mean(axis=0) for d in range(10)])
    hue_pairs, hue_min_sep = figure4(mu_np, y_np, centroids)

    # ---- Figure 5: the decoded plane -------------------------------------- #
    q = np.linspace(Q_LO, Q_HI, GRID)
    z_coords = norm.ppf(q)                     # inverse CDF of N(0, 1)
    figure5(model, z_coords)

    # Which class does each of the 225 cells look like?  The nearest encoded
    # test image in latent space -- the honest way to say what sits where in
    # Figure 5, instead of squinting at it.
    cell_labels = []
    for z2 in reversed(z_coords):
        line = []
        for z1 in z_coords:
            dist = ((mu_np - np.array([z1, z2])) ** 2).sum(axis=1)
            line.append(int(y_np[int(dist.argmin())]))
        cell_labels.append(line)

    # ---- Figure 6: interpolation ------------------------------------------ #
    model16 = VAE(D16)
    model16.load_state_dict(torch.load(CKPT))

    # Two test images of different digits: the first 0 and the first 1 — about
    # the most dissimilar pair of shapes MNIST offers (a closed loop against a
    # single stroke), so the contrast between the two interpolations is as
    # visible as it gets.
    idx_a = int((y_test == 0).nonzero()[0].item())
    idx_b = int((y_test == 1).nonzero()[0].item())
    x_a, x_b = x_test[idx_a], x_test[idx_b]
    mid_px, mid_lat, mid_px_c, mid_lat_c = figure6(
        model16, x_a, x_b, idx_a, idx_b, 0, 1)

    # ---- probes ----------------------------------------------------------- #
    # Encode the validation split with both models and read class information
    # out of the two latent spaces, to put a number on what d = 2 costs.
    val_x, val_y = [], []
    for xb, yb in val_loader:
        val_x.append(xb.view(xb.shape[0], 784))
        val_y.append(yb)
    val_x = torch.cat(val_x)
    val_y = torch.cat(val_y).numpy()
    with torch.no_grad():
        mu_val2, _ = model.encode(val_x)
        mu_val16, _ = model16.encode(val_x)
        mu_test16, _ = model16.encode(x_test)
    acc2, pairs2 = knn_probe(mu_val2.numpy(), val_y, mu_np, y_np)
    acc16, pairs16 = knn_probe(mu_val16.numpy(), val_y, mu_test16.numpy(), y_np)
    print(f"k-NN on mu: d = 2 -> {acc2:.4f}   d = 16 -> {acc16:.4f}")

    with open(os.path.join(RES, "vae1.json"), encoding="utf-8") as f:
        ex1 = json.load(f)

    out = {
        "config": {"d": D, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH,
                   "beta": BETA, "seed": SEED, "grid": GRID,
                   "q_range": [Q_LO, Q_HI]},
        "train_seconds": secs,
        "history": history,
        "test": {"recon": test_recon, "kl": test_kl,
                 "neg_elbo": test_recon + test_kl},
        "val_kl_first_epoch": history[0]["val_kl"],
        "val_kl_last_epoch": history[-1]["val_kl"],
        "compare_d16": ex1["test"],
        "grid_q": q.tolist(),
        "grid_z": z_coords.tolist(),
        "cell_labels": cell_labels,
        "centroids": centroids.tolist(),
        "figure4_hue_pairs": [list(p) for p in hue_pairs],
        "figure4_hue_min_separation": hue_min_sep,
        "class_spread": {str(d): float(np.sqrt(
            ((mu_np[y_np == d] - centroids[d]) ** 2).sum(axis=1).mean()))
            for d in range(10)},
        "knn_probe": {"k": 10,
                      "acc_d2": acc2, "top_confusions_d2": pairs2,
                      "acc_d16": acc16, "top_confusions_d16": pairs16},
        "interpolation": {"idx_a": idx_a, "idx_b": idx_b,
                          "digit_a": 0, "digit_b": 1,
                          "midness_pixel": mid_px, "midness_latent": mid_lat,
                          "midness_pixel_centre": mid_px_c,
                          "midness_latent_centre": mid_lat_c},
    }
    with open(os.path.join(RES, "vae2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {os.path.join(RES, 'vae2.json')}")


if __name__ == "__main__":
    main()
