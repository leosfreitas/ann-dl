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

</div>

## Reproducing

```bash
git clone https://github.com/leosfreitas/ann-dl
cd ann-dl
pip install -r requirements.txt
```

Each exercise folder carries its own `code/` directory with the scripts that generated it.
