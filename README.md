# ann-dl

Exercises for **Artificial Neural Networks & Deep Learning** — Insper, 2026.2.

## Published pages

| Exercise | URL |
|---|---|
| Data | https://leosfreitas.github.io/ann-dl/exercises/data/ |

## Exercise: Data

Three studies on how the shape of a dataset determines what a classifier can do.

1. **Point clouds in 2D** — four Gaussian classes at four spread factors; separation ratios, mixing rate, and
   piecewise-linear decision boundaries.
2. **Non-linearity in 5D** — shifted Gaussians versus concentric shells; PCA, centre distances, radius
   distributions, and why no linear boundary can separate the shells at any sample size.
3. **Spaceship Titanic** — a leakage-free preprocessing pipeline turning the raw Kaggle table into a numeric
   matrix bounded in `[-1, 1]` for a `tanh` network.

## Layout

```
code/                  analysis scripts
  style.py             shared matplotlib style and validated palette
  ex1.py               Figures 1, 1b, 2, 3   -> results/ex1.json
  ex2.py               Figures 4, 5          -> results/ex2.json
  ex3.py               Figures 6, 7          -> results/ex3.json
  build_report.py      renders exercises/data/index.html from the JSON
data/train.csv         Spaceship Titanic training file (8,693 x 14)
results/*.json         every computed value, as emitted by the scripts
exercises/data/        the published page and its figures
```

## Reproducing

```bash
pip install numpy pandas matplotlib scikit-learn

cd code
python ex1.py
python ex2.py
python ex3.py
python build_report.py
```

The seed is `42` throughout, so runs are reproducible. The report page is *generated* from the JSON result
files rather than written by hand, so no number in the prose can drift from what the code computed.

Libraries are limited to NumPy, pandas, Matplotlib, and scikit-learn for `PCA`, `train_test_split`,
`OneHotEncoder` and `MinMaxScaler` only — no model is trained.
