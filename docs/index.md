# ann-dl

Coursework for **Artificial Neural Networks & Deep Learning** — Insper, 2026.2.
Written by Leonardo Sterman Freitas.

Each exercise is a self-contained study: the scripts that produced it live beside the report, the
figures are committed, and every number quoted in the prose is interpolated from the JSON those
scripts emit — so nothing on a page can drift away from what the code actually computed. The seed is
`42` throughout.

## Exercises

<div class="grid cards" markdown>

-   **[Data](exercises/data/index.html)**

    Point clouds and spread in 2D, non-linearity in 5D, and preparing the Spaceship Titanic table for
    a `tanh` network.

-   **[Perceptron](exercises/perceptron/index.md)**

    A perceptron written from scratch on separable and overlapping Gaussians: why one run stops by
    itself, why the other never can, and what the pocket algorithm is for.

-   **[MLP](exercises/mlp/index.md)**

    One backpropagation step by hand, then a from-scratch NumPy MLP on a problem no line can solve,
    the same code on three classes, and a controlled test of whether a second hidden layer pays off.

-   **[VAE](exercises/vae/index.md)**

    A variational autoencoder on MNIST, with the reconstruction and KL terms logged apart: what the
    prior buys, what a two-dimensional latent space costs, and what the decoded plane looks like.

</div>

## Reproducing

```bash
git clone https://github.com/leosfreitas/ann-dl
cd ann-dl
pip install -r requirements.txt
```

Each exercise folder carries its own `code/` directory with the scripts that generated it.
