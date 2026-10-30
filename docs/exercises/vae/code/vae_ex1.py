"""Exercise 1 -- a VAE on MNIST with d = 16.

Produces Figures 1, 2 and 3, writes every computed number to
``results/vae1.json``, and saves the trained weights to ``vae_d16.pt`` so that
Exercise 2D can interpolate in this model's latent space.

    python vae_ex1.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import style
from mnist_data import first_index_per_digit, load_mnist
from vae import VAE, evaluate, n_params, train_vae

D = 16
EPOCHS = 20
LR = 1e-3
BATCH = 128
BETA = 1.0
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
CKPT = os.path.join(HERE, "vae_d16.pt")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def figure1(history):
    """Two panels: reconstruction x epoch and KL x epoch, train vs validation."""
    epochs = [h["epoch"] for h in history]
    series = [
        ("Reconstruction (BCE)", "train_recon", "val_recon",
         "Bernoulli negative log-likelihood, summed over the 784 pixels"),
        ("KL to the prior", "train_kl", "val_kl",
         f"closed-form KL, summed over the {D} latent dimensions"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, (name, k_tr, k_va, sub) in zip(axes, series):
        tr = [h[k_tr] for h in history]
        va = [h[k_va] for h in history]
        ax.plot(epochs, tr, color=style.SERIES[0], lw=2.0, label="train")
        ax.plot(epochs, va, color=style.SERIES[1], lw=2.0, label="validation")
        # The two curves converge and end on top of each other, so end-of-line
        # direct labels would collide: the legend carries identity here.
        style.title(ax, name, sub)
        ax.set_xlabel("epoch")
        ax.set_ylabel("nats per image")
        ax.set_xlim(0.6, EPOCHS + 0.4)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.legend(loc="center right" if "KL" in name else "upper right")
    fig.suptitle("Figure 1 — the two terms of the loss, logged separately",
                 x=0.008, ha="left", fontsize=12.5, color=style.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "fig1_curves.png"))
    plt.close(fig)


def figure2(model, test_set, indices):
    """Originals (top) and their reconstructions from mu (bottom)."""
    x = torch.stack([test_set[i][0].view(-1) for i in indices])
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x)
        # Decode mu, not a sampled z: the deterministic reconstruction.
        recon = model.decode(mu)

    fig, axes = plt.subplots(2, 10, figsize=(10.5, 2.5))
    for col, digit in enumerate(range(10)):
        for row, img in ((0, x[col]), (1, recon[col])):
            ax = axes[row][col]
            ax.imshow(img.view(28, 28).numpy(), cmap="gray", vmin=0, vmax=1)
            style.image_ax(ax)
        axes[0][col].set_title(str(digit), fontsize=9, color=style.INK, pad=4)
    axes[0][0].set_ylabel("original", fontsize=8.5, color=style.INK_SOFT)
    axes[1][0].set_ylabel(r"decode($\mu$)", fontsize=8.5, color=style.INK_SOFT)
    fig.suptitle("Figure 2 — first test image of each digit and its reconstruction",
                 x=0.012, ha="left", fontsize=11, color=style.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(FIG, "fig2_recon.png"))
    plt.close(fig)


def figure3(model):
    """8 x 8 images decoded from z ~ N(0, I), drawn after manual_seed(0)."""
    torch.manual_seed(0)          # the seed the statement prescribes here
    samples = model.sample(64)    # z ~ N(0, I), then decoded

    fig, axes = plt.subplots(8, 8, figsize=(6.4, 6.8))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(samples[k].view(28, 28).numpy(), cmap="gray", vmin=0, vmax=1)
        style.image_ax(ax)
    fig.suptitle(r"Figure 3 — 64 samples from the prior, $z \sim \mathcal{N}(0, I)$",
                 x=0.012, ha="left", fontsize=11, color=style.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(os.path.join(FIG, "fig3_prior.png"))
    plt.close(fig)
    return samples


# --------------------------------------------------------------------------- #

def main():
    style.apply_style()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    train_loader, val_loader, test_loader, test_set = load_mnist(BATCH)

    print(f"Training VAE with d = {D} for {EPOCHS} epochs "
          f"(Adam, lr = {LR}, batch = {BATCH}, beta = {BETA})")
    model, history, secs = train_vae(
        D, train_loader, val_loader, epochs=EPOCHS, lr=LR, beta=BETA, seed=SEED)
    print(f"  done in {secs:.1f} s")

    # The test set is touched here for the first time, for the final numbers.
    test_recon, test_kl = evaluate(model, test_loader)
    print(f"TEST  recon {test_recon:.3f}  kl {test_kl:.3f}  "
          f"-ELBO {test_recon + test_kl:.3f}  nats/image")

    torch.save(model.state_dict(), CKPT)

    indices = first_index_per_digit(test_set)
    figure1(history)
    figure2(model, test_set, indices)
    samples = figure3(model)

    # ---- two measurements for the analysis in 1E ------------------------- #
    # (i) Blur.  A committed pixel is near 0 or near 1; a blurry one sits in
    # between.  ``midness`` is the mean distance to the nearest extreme, and
    # ``frac_mid`` the share of pixels stranded in [0.1, 0.9] -- the second is
    # the sharper statistic, because the mean is dominated by the large black
    # background that every image gets right.
    def blur_stats(batch):
        v = batch.detach().numpy()
        return {"midness": float(np.mean(np.minimum(v, 1.0 - v))),
                "frac_mid": float(np.mean((v > 0.1) & (v < 0.9)))}

    x_recon = torch.stack([test_set[i][0].view(-1) for i in indices])
    with torch.no_grad():
        mu_r, _ = model.encode(x_recon)
        recon_imgs = model.decode(mu_r)

        # A larger, fairer sample for the blur comparison than the 10 digits
        # of Figure 2: the first 1 000 test images, reconstructed from mu.
        x_many = torch.stack([test_set[i][0].view(-1) for i in range(1000)])
        mu_many, _ = model.encode(x_many)
        recon_many = model.decode(mu_many)

        # (ii) Where prior samples land.  Encode the whole test set, then ask
        # how far a draw from N(0, I) falls from the nearest point the encoder
        # ever produced -- against the same distance measured between encoded
        # images themselves.  If the first is much larger, the prior is putting
        # mass where the decoder was never trained.
        mu_test, _ = model.encode(torch.stack(
            [test_set[i][0].view(-1) for i in range(10_000)]))
        torch.manual_seed(0)
        z_prior = torch.randn(1000, D)

        d_prior = torch.cdist(z_prior, mu_test).min(dim=1).values
        # For an encoded image, the nearest *other* encoded image: the diagonal
        # of the self-distance matrix is 0, so take the second smallest.
        d_enc = torch.cdist(mu_test[:1000], mu_test).topk(
            2, dim=1, largest=False).values[:, 1]

    landing = {
        "nearest_mu_from_prior_sample": float(d_prior.mean()),
        "nearest_other_mu_from_encoded": float(d_enc.mean()),
        "norm_prior_z": float(z_prior.norm(dim=1).mean()),
        "norm_encoded_mu": float(mu_test.norm(dim=1).mean()),
        "n_prior_draws": 1000,
    }
    print(f"landing: prior draw -> nearest mu {landing['nearest_mu_from_prior_sample']:.2f}"
          f"   mu -> nearest other mu {landing['nearest_other_mu_from_encoded']:.2f}"
          f"   ||z|| {landing['norm_prior_z']:.2f} vs ||mu|| {landing['norm_encoded_mu']:.2f}")

    out = {
        "config": {"d": D, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH,
                   "beta": BETA, "seed": SEED, "optimizer": "Adam",
                   "n_train": 50_000, "n_val": 10_000, "n_test": 10_000},
        "n_params": n_params(model),
        "train_seconds": secs,
        "history": history,
        "test": {"recon": test_recon, "kl": test_kl,
                 "neg_elbo": test_recon + test_kl},
        "val_kl_first_epoch": history[0]["val_kl"],
        "val_kl_last_epoch": history[-1]["val_kl"],
        "val_kl_max": max(h["val_kl"] for h in history),
        "val_kl_argmax_epoch": max(history, key=lambda h: h["val_kl"])["epoch"],
        "val_recon_first_epoch": history[0]["val_recon"],
        "val_recon_last_epoch": history[-1]["val_recon"],
        "figure2_test_indices": indices,
        "blur": {"prior_samples": blur_stats(samples),
                 "reconstructions_fig2": blur_stats(recon_imgs),
                 "reconstructions_1000": blur_stats(recon_many),
                 "originals_fig2": blur_stats(x_recon),
                 "originals_1000": blur_stats(x_many)},
        "landing": landing,
    }
    with open(os.path.join(RES, "vae1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {os.path.join(RES, 'vae1.json')}")


if __name__ == "__main__":
    main()
