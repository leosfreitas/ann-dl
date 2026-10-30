# ann-dl

Exercises for **Artificial Neural Networks & Deep Learning** — Insper, 2026.2.

## Published pages

| Exercise | URL |
|---|---|
| Data | https://leosfreitas.github.io/ann-dl/exercises/data/ |
| Perceptron | https://leosfreitas.github.io/ann-dl/exercises/perceptron/ |
| MLP | https://leosfreitas.github.io/ann-dl/exercises/mlp/ |
| VAE | https://leosfreitas.github.io/ann-dl/exercises/vae/ |

## Exercise: Data

Three studies on how the shape of a dataset determines what a classifier can do.

1. **Point clouds in 2D** — four Gaussian classes at four spread factors; separation ratios, mixing rate, and
   piecewise-linear decision boundaries.
2. **Non-linearity in 5D** — shifted Gaussians versus concentric shells; PCA, centre distances, radius
   distributions, and why no linear boundary can separate the shells at any sample size.
3. **Spaceship Titanic** — a leakage-free preprocessing pipeline turning the raw Kaggle table into a numeric
   matrix bounded in `[-1, 1]` for a `tanh` network.

## Exercise: Perceptron

A perceptron written from scratch (NumPy only) on two 2000-point datasets.

1. **Separable data** — Gaussians at `[1.5, 1.5]` and `[5, 5]`, covariance `0.5I`. Converges on a
   zero-update epoch at epoch 26 with 100% accuracy; the `eta = 1.0` re-run and a zero-initialised run show
   that the learning rate sets the scale of `w`, not the boundary.
2. **Overlapping data** — Gaussians at `[3, 3]` and `[4, 4]`, covariance `1.5I`. Never converges: the
   final weights score 50.05% while the pocket holds 72.85%, against a 72.60% ceiling for any straight line.

## Exercise: VAE

A variational autoencoder on MNIST, with the two terms of the loss logged separately throughout.

1. **d = 16** -- encoder 784-512-256, two heads, mirrored decoder; Bernoulli reconstruction and the
   closed-form Gaussian KL written out by hand, trained 20 epochs with Adam. The KL *rises* over the
   first epochs while the total falls -- the optimiser buying latent capacity.
2. **d = 2** -- the same model with a latent space you can draw: the encoded test set, the decoded
   plane on an inverse-normal-CDF grid, and pixel-space against latent-space interpolation.

## Layout

MkDocs Material, published to GitHub Pages by `.github/workflows/deploy.yml` on every push.

```
mkdocs.yml
requirements.txt
docs/
  index.md                       landing page
  exercises/
    data/index.html              the data exercise page (static) + figures/
    perceptron/
      index.md                   the report, generated from results/
      code/                      the sources that produced it
        perceptron.py            the perceptron itself, from scratch
        perceptron_ex1.py        Figures 1, 2, 3     -> results/perceptron1.json
        perceptron_ex2.py        Figures 4, 5, 6, 6b -> results/perceptron2.json
        build_perceptron_report.py  renders ../index.md
        style.py                 shared matplotlib style
      figures/                   the figures the report shows
      results/                   every computed value, as emitted by the scripts
    mlp/
      index.md                 the report
      code/                    mlp.py (from scratch) + mlp_ex1-4.py + style.py
      figures/ results/
    vae/
      index.md                 the report, generated from results/
      code/
        vae.py                 model, reparameterization, both loss terms, training loop
        mnist_data.py          MNIST and the fixed 50k/10k split
        vae_ex1.py             Figures 1, 2, 3 -> results/vae1.json
        vae_ex2.py             Figures 4, 5, 6 -> results/vae2.json
        build_vae_report.py    renders ../index.md
        style.py               shared matplotlib style
      figures/ results/
code/                            the data exercise's scripts (ex1-3, build_report)
data/train.csv                   Spaceship Titanic training file (8,693 x 14)
results/*.json                   the data exercise's computed values
```

## Reproducing

```bash
pip install -r requirements.txt

# Data exercise
cd code
python ex1.py && python ex2.py && python ex3.py && python build_report.py

# Perceptron exercise (NumPy and Matplotlib only)
cd docs/exercises/perceptron/code
python perceptron_ex1.py && python perceptron_ex2.py && python build_perceptron_report.py

# VAE exercise (PyTorch; MNIST downloads itself on first run)
cd docs/exercises/vae/code
python vae_ex1.py && python vae_ex2.py && python build_vae_report.py

# Serve the site locally
mkdocs serve
```

The seed is `42` throughout, so runs are reproducible. The report page is *generated* from the JSON result
files rather than written by hand, so no number in the prose can drift from what the code computed.

The VAE exercise is the only one that uses PyTorch, and only for the plain `Linear` layers, `Adam`
and autograd: the reparameterization, both terms of the loss and the sampling are written out in
`docs/exercises/vae/code/vae.py`. No library VAE class and no `torch.distributions.kl_divergence`.

No third-party model is trained anywhere in this repository. The data exercise uses NumPy, pandas, Matplotlib
and scikit-learn restricted to `PCA`, `train_test_split`, `OneHotEncoder` and `MinMaxScaler`; the perceptron
exercise uses NumPy and Matplotlib only, with the model written out by hand in `docs/exercises/perceptron/code/perceptron.py`.
