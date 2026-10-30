"""The variational autoencoder itself: model, loss and training loop.

Everything the activity asks to be hand-written lives here and nowhere else:

* ``VAE.reparameterize`` -- the reparameterization trick, written out;
* ``vae_loss``          -- the Bernoulli reconstruction term and the closed-form
                           Gaussian KL, both written out, both summed over their
                           own axis so the two numbers are in *nats per image*;
* ``VAE.sample``        -- sampling from the prior.

No VAE class from a library and no ``torch.distributions.kl_divergence``:
``torch.nn`` is used only for the plain ``Linear`` layers and for ``Adam``.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PIXELS = 784  # 28 x 28, flattened


class VAE(nn.Module):
    """Encoder 784->512->256, two heads to d, decoder d->256->512->784.

    The architecture is the one fixed by the statement, so that the loss values
    can be compared against the reference numbers.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d

        # Encoder: two ReLU layers shared by both heads.
        self.enc = nn.Sequential(
            nn.Linear(PIXELS, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        # Two linear heads, 256 -> d.  log sigma^2 rather than sigma, so the
        # head can output any real number and the variance stays positive.
        self.head_mu = nn.Linear(256, d)
        self.head_logvar = nn.Linear(256, d)

        # Decoder: mirror of the encoder, sigmoid at the output so every pixel
        # is a Bernoulli probability -- which is what the BCE term expects.
        self.dec = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, PIXELS), nn.Sigmoid(),
        )

    def encode(self, x):
        """x: (N, 784) -> mu, logvar, both (N, d)."""
        h = self.enc(x)
        return self.head_mu(h), self.head_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps,  eps ~ N(0, I).  The reparameterization trick.

        Sampling z directly from N(mu, sigma^2) would put the randomness between
        the parameters and the loss, and the gradient could not pass through it.
        Here the randomness is an *input* (``eps``), independent of the
        parameters, and z is a deterministic, differentiable function of mu and
        sigma -- so dL/dmu and dL/dsigma exist and autograd can follow them.

        sigma comes from logvar as sigma = exp(logvar / 2), which is positive for
        any logvar the head happens to produce.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        """z: (N, d) -> x_hat: (N, 784), each entry in (0, 1)."""
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device="cpu"):
        """n images decoded from z ~ N(0, I) -- sampling from the prior."""
        z = torch.randn(n, self.d, device=device)
        return self.decode(z)


def vae_loss(x_hat, x, mu, logvar):
    """The two terms of the negative ELBO, separately, in nats per image.

    Reconstruction (Bernoulli likelihood, i.e. -log p(x|z)):

        -sum_j [ x_j log x_hat_j + (1 - x_j) log(1 - x_hat_j) ]

    summed over the 784 pixels and then averaged over the batch.  ``reduction=
    "sum"`` divided by the batch size is exactly that; ``reduction="mean"``
    would also divide by 784 and shrink this term 784x against the KL, which is
    the same as secretly training with beta ~= 784.

    KL, closed form for KL( N(mu, diag(sigma^2)) || N(0, I) ):

        1/2 sum_i ( mu_i^2 + sigma_i^2 - 1 - log sigma_i^2 )

    written out term by term, with sigma^2 = exp(logvar) and log sigma^2 =
    logvar, summed over the d latent dimensions and averaged over the batch.

    Returns the two terms unweighted; the caller forms ``recon + beta * kl``.
    """
    n = x.shape[0]

    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / n

    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / n

    return recon, kl


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Average reconstruction and KL over a loader, in nats per image.

    Single-sample estimate: z is sampled exactly as in training, because that is
    what the ELBO is an expectation over.  The averages are weighted by batch
    size, so a last short batch does not count as much as a full one.
    """
    model.eval()
    recon_sum = kl_sum = 0.0
    n_total = 0
    for x, _ in loader:
        x = x.view(x.shape[0], PIXELS).to(device)
        x_hat, mu, logvar = model(x)
        recon, kl = vae_loss(x_hat, x, mu, logvar)
        recon_sum += recon.item() * x.shape[0]
        kl_sum += kl.item() * x.shape[0]
        n_total += x.shape[0]
    return recon_sum / n_total, kl_sum / n_total


def train_vae(d, train_loader, val_loader, epochs=20, lr=1e-3, beta=1.0,
              seed=42, device="cpu", log=print):
    """Train a VAE and return (model, history).

    ``history`` is one dict per epoch with the two loss terms logged separately
    on the training set (averaged over the epoch) and on the validation set.
    The test set is not touched here -- it is only used for the final numbers.

    The seed is set immediately before the model is built, as the statement
    requires, so the weight initialisation (and the eps draws that follow) are
    reproducible.
    """
    torch.manual_seed(seed)
    model = VAE(d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        recon_sum = kl_sum = 0.0
        n_total = 0
        for x, _ in train_loader:
            x = x.view(x.shape[0], PIXELS).to(device)

            x_hat, mu, logvar = model(x)
            recon, kl = vae_loss(x_hat, x, mu, logvar)
            loss = recon + beta * kl  # beta = 1 for this activity

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # Accumulate the two terms separately -- a falling total says
            # nothing about which of the two is winning.
            recon_sum += recon.item() * x.shape[0]
            kl_sum += kl.item() * x.shape[0]
            n_total += x.shape[0]

        train_recon = recon_sum / n_total
        train_kl = kl_sum / n_total
        val_recon, val_kl = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_recon": train_recon, "train_kl": train_kl,
            "val_recon": val_recon, "val_kl": val_kl,
        })
        log(f"  epoch {epoch:2d}  train recon {train_recon:7.3f}  kl {train_kl:6.3f}"
            f"   |  val recon {val_recon:7.3f}  kl {val_kl:6.3f}")

    return model, history, time.perf_counter() - t0


def n_params(model):
    return sum(p.numel() for p in model.parameters())
