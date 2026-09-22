# ann-dl

Exercises for **Artificial Neural Networks & Deep Learning** — Insper, 2026.2.

## Published pages

| Exercise | URL |
|---|---|
| Data | https://leosfreitas.github.io/ann-dl/exercises/data/ |
| Perceptron | https://leosfreitas.github.io/ann-dl/exercises/perceptron/ |

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

## Layout

```
code/                        analysis scripts
  style.py                   shared matplotlib style and validated palette
  report_kit.py              shared report stylesheet and HTML helpers

  ex1.py                     data: Figures 1, 1b, 2, 3   -> results/ex1.json
  ex2.py                     data: Figures 4, 5          -> results/ex2.json
  ex3.py                     data: Figures 6, 7          -> results/ex3.json
  build_report.py            renders exercises/data/index.html

  perceptron.py              the perceptron itself, from scratch
  perceptron_ex1.py          Figures 1, 2, 3      -> results/perceptron1.json
  perceptron_ex2.py          Figures 4, 5, 6, 6b  -> results/perceptron2.json
  build_perceptron_report.py renders exercises/perceptron/index.html

data/train.csv               Spaceship Titanic training file (8,693 x 14)
results/*.json               every computed value, as emitted by the scripts
exercises/<name>/            the published page and its figures
```

## Reproducing

```bash
pip install numpy pandas matplotlib scikit-learn

cd code

# Data exercise
python ex1.py && python ex2.py && python ex3.py && python build_report.py

# Perceptron exercise (NumPy and Matplotlib only)
python perceptron_ex1.py && python perceptron_ex2.py && python build_perceptron_report.py
```

The seed is `42` throughout, so runs are reproducible. The report page is *generated* from the JSON result
files rather than written by hand, so no number in the prose can drift from what the code computed.

No third-party model is trained anywhere in this repository. The data exercise uses NumPy, pandas, Matplotlib
and scikit-learn restricted to `PCA`, `train_test_split`, `OneHotEncoder` and `MinMaxScaler`; the perceptron
exercise uses NumPy and Matplotlib only, with the model written out by hand in `code/perceptron.py`.
