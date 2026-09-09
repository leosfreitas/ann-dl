"""Exercise 3 -- Real-world data: Spaceship Titanic preprocessing for a tanh network."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from style import apply_style, title, SERIES, SURFACE, INK, INK_SOFT

apply_style()
pd.set_option("future.no_silent_downcasting", True)
FIG = "../exercises/data/figures"
SEED = 42
res = {}

df = pd.read_csv("../data/train.csv")
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUM = ["Age"] + SPEND
CAT = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP = ["PassengerId", "Cabin", "Name"]

# =========================================================== Part A: exploration
res["n_rows"], res["n_cols"] = int(df.shape[0]), int(df.shape[1])
vc = df["Transported"].value_counts()
res["target"] = {
    "n_true": int(vc.get(True, 0)), "n_false": int(vc.get(False, 0)),
    "pct_true": float(df["Transported"].mean() * 100),
    "pct_false": float((1 - df["Transported"].mean()) * 100),
}
miss = pd.DataFrame({
    "column": df.columns,
    "missing": df.isna().sum().values,
    "pct": (df.isna().sum().values / len(df) * 100),
})
res["missing"] = miss.to_dict("records")
res["missing_total"] = int(df.isna().sum().sum())
res["rows_any_missing"] = int(df.isna().any(axis=1).sum())
res["rows_any_missing_pct"] = float(df.isna().any(axis=1).mean() * 100)

res["spend_stats_full"] = {
    c: {"mean": float(df[c].mean()), "median": float(df[c].median()),
        "max": float(df[c].max()), "pct_zero": float((df[c] == 0).mean() * 100),
        "skew": float(df[c].skew())}
    for c in SPEND
}
res["age_stats_full"] = {"mean": float(df["Age"].mean()),
                         "median": float(df["Age"].median()),
                         "max": float(df["Age"].max())}
res["cat_levels"] = {c: [str(v) for v in sorted(df[c].dropna().unique().tolist(), key=str)]
                     for c in CAT}

# ============================================ Part B: stratified split, pre-transform
X_raw = df.drop(columns=["Transported"])
y = df["Transported"].astype(int)
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.20, random_state=SEED, stratify=y)
res["split"] = {
    "n_train": int(len(X_tr_raw)), "n_test": int(len(X_te_raw)),
    "pct_true_train": float(y_tr.mean() * 100),
    "pct_true_test": float(y_te.mean() * 100),
}

# FoodCourt on the TRAINING set, before any transformation (summary item 11)
res["foodcourt_train_raw"] = {
    "mean": float(X_tr_raw["FoodCourt"].mean()),
    "median": float(X_tr_raw["FoodCourt"].median()),
    "max": float(X_tr_raw["FoodCourt"].max()),
    "pct_zero": float((X_tr_raw["FoodCourt"] == 0).mean() * 100),
}

# ============================================================ Part C: preprocessing
X_tr = X_tr_raw.drop(columns=DROP).copy()
X_te = X_te_raw.drop(columns=DROP).copy()

# C1 -- imputation, every statistic learned on the training split only
num_medians = X_tr[NUM].median()
cat_modes = {c: X_tr[c].mode(dropna=True)[0] for c in CAT}
res["imputation"] = {
    "num_medians": {k: float(v) for k, v in num_medians.items()},
    "cat_modes": {k: str(v) for k, v in cat_modes.items()},
}
for d in (X_tr, X_te):
    d[NUM] = d[NUM].fillna(num_medians)
    for c in CAT:
        d[c] = d[c].fillna(cat_modes[c]).astype(str)

# C3 -- feature engineering (TotalSpend computed before the log transform)
for d in (X_tr, X_te):
    d["TotalSpend"] = d[SPEND].sum(axis=1)
HEAVY = SPEND + ["TotalSpend"]

res["totalspend_train_raw"] = {
    "mean": float(X_tr["TotalSpend"].mean()),
    "median": float(X_tr["TotalSpend"].median()),
    "max": float(X_tr["TotalSpend"].max()),
    "skew": float(X_tr["TotalSpend"].skew()),
}

# C4 -- log(1+x) on the heavy-tailed money columns
skew_before = {c: float(X_tr[c].skew()) for c in HEAVY}
fc_before = X_tr["FoodCourt"].to_numpy(copy=True)
for d in (X_tr, X_te):
    d[HEAVY] = np.log1p(d[HEAVY])
skew_after = {c: float(X_tr[c].skew()) for c in HEAVY}
res["skew"] = {"before": skew_before, "after": skew_after}
fc_after = X_tr["FoodCourt"].to_numpy(copy=True)
res["foodcourt_train_log"] = {"mean": float(fc_after.mean()),
                              "median": float(np.median(fc_after)),
                              "max": float(fc_after.max())}

# C2 -- one-hot encoding, fitted on train; unseen test categories map to all-zeros
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
enc.fit(X_tr[CAT])
cat_names = enc.get_feature_names_out(CAT).tolist()
NUM_FINAL = ["Age"] + HEAVY
Xtr_mat = np.hstack([X_tr[NUM_FINAL].to_numpy(float), enc.transform(X_tr[CAT])])
Xte_mat = np.hstack([X_te[NUM_FINAL].to_numpy(float), enc.transform(X_te[CAT])])
feature_names = NUM_FINAL + cat_names
res["features"] = {"names": feature_names, "n_numeric": len(NUM_FINAL),
                   "n_onehot": len(cat_names), "n_total": len(feature_names),
                   "onehot_names": cat_names}

# C5 -- scale to [-1, 1] for tanh, fitted on train only
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(Xtr_mat)
Xtr_s = scaler.transform(Xtr_mat)
Xte_s = scaler.transform(Xte_mat)

# ======================================================== Part D: verification
res["final"] = {
    "train_shape": list(Xtr_s.shape), "test_shape": list(Xte_s.shape),
    "train_min": float(Xtr_s.min()), "train_max": float(Xtr_s.max()),
    "test_min": float(Xte_s.min()), "test_max": float(Xte_s.max()),
    "train_nan": int(np.isnan(Xtr_s).sum()), "test_nan": int(np.isnan(Xte_s).sum()),
    "test_out_of_range": int(((Xte_s < -1) | (Xte_s > 1)).sum()),
    "test_out_of_range_rows": int(((Xte_s < -1) | (Xte_s > 1)).any(axis=1).sum()),
}
oor_cols = np.where(((Xte_s < -1) | (Xte_s > 1)).any(axis=0))[0]
res["final"]["test_out_of_range_features"] = [feature_names[i] for i in oor_cols]

# Illustration for the Part D reflection: where a typical spender lands on the
# scaled axis with and without the log step.
p75 = float(np.percentile(fc_before, 75))
raw_lo, raw_hi = float(fc_before.min()), float(fc_before.max())
log_lo, log_hi = float(fc_after.min()), float(fc_after.max())
res["log_illustration"] = {
    "p75_raw": p75,
    "raw_max": raw_hi,
    "scaled_without_log": -1 + 2 * (p75 - raw_lo) / (raw_hi - raw_lo),
    "scaled_with_log": -1 + 2 * (float(np.log1p(p75)) - log_lo) / (log_hi - log_lo),
    "pct_at_floor": float((fc_before == 0).mean() * 100),
}

# ---------------------------------------------------------------- Figure 6
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
axes[0].hist(fc_before, bins=50, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
title(axes[0], "Before - raw FoodCourt",
      "skew {:.2f} | median {:.0f} | max {:.0f}".format(
          skew_before["FoodCourt"], np.median(fc_before), fc_before.max()))
axes[0].set_xlabel("FoodCourt (raw units)")
axes[0].set_yscale("log")
axes[0].set_ylabel("count (log scale)")

axes[1].hist(fc_after, bins=50, color=SERIES[1], edgecolor=SURFACE, linewidth=0.6)
title(axes[1], "After - log(1 + FoodCourt)",
      "skew {:.2f} | median {:.2f} | max {:.2f}".format(
          skew_after["FoodCourt"], np.median(fc_after), fc_after.max()))
axes[1].set_xlabel("log(1 + FoodCourt)")
axes[1].set_ylabel("count")
fig.suptitle(
    "Figure 6 - Log transform pulls the FoodCourt tail into a usable range (training split)",
    x=0.045, ha="left", fontsize=12, color=INK, weight="600")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIG + "/fig6_logtransform.png")
plt.close(fig)

# ------------------------------------------------- Figure 7: post-scaling ranges
fig, ax = plt.subplots(figsize=(9.2, 5.6))
pos = np.arange(len(feature_names))
ax.barh(pos, Xtr_s.max(0) - Xtr_s.min(0), left=Xtr_s.min(0), height=0.6,
        color=SERIES[0], alpha=0.85, label="train range")
ax.scatter(Xte_s.min(0), pos, s=30, color=SERIES[1], marker="|", linewidths=2.0,
           label="test min / max", zorder=4)
ax.scatter(Xte_s.max(0), pos, s=30, color=SERIES[1], marker="|", linewidths=2.0, zorder=4)
for v in (-1, 1):
    ax.axvline(v, color=INK_SOFT, lw=1.0, ls="--", zorder=2)
ax.set_yticks(pos)
ax.set_yticklabels(feature_names, fontsize=8)
ax.invert_yaxis()
title(ax, "Figure 7 - Every feature lands inside the tanh-friendly band",
      "dashed lines mark -1 and +1; test values may sit just outside the train range")
ax.set_xlabel("scaled value")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2)
ax.grid(axis="y", visible=False)
fig.savefig(FIG + "/fig7_ranges.png")
plt.close(fig)

json.dump(res, open("../results/ex3.json", "w"), indent=2)

print("rows={} cols={}".format(res["n_rows"], res["n_cols"]))
print("target: {:.2f}% True / {:.2f}% False".format(
    res["target"]["pct_true"], res["target"]["pct_false"]))
print("missing cells={}  rows with >=1 NaN={} ({:.2f}%)".format(
    res["missing_total"], res["rows_any_missing"], res["rows_any_missing_pct"]))
print("\nmissing by column:")
for r in res["missing"]:
    if r["missing"]:
        print("  {:<15} {:>4}  {:.2f}%".format(r["column"], r["missing"], r["pct"]))
print("\nspend stats (full file):")
for c, v in res["spend_stats_full"].items():
    print("  {:<13} mean={:8.2f} median={:6.1f} max={:8.0f} zero={:.1f}% skew={:.2f}".format(
        c, v["mean"], v["median"], v["max"], v["pct_zero"], v["skew"]))
print("\nsplit: train={} test={} ({:.2f}% / {:.2f}% positive)".format(
    res["split"]["n_train"], res["split"]["n_test"],
    res["split"]["pct_true_train"], res["split"]["pct_true_test"]))
print("FoodCourt train raw: mean={:.2f} median={:.1f} max={:.0f}".format(
    res["foodcourt_train_raw"]["mean"], res["foodcourt_train_raw"]["median"],
    res["foodcourt_train_raw"]["max"]))
print("\nimputation medians:", res["imputation"]["num_medians"])
print("imputation modes  :", res["imputation"]["cat_modes"])
print("\nfeatures: {} ({} numeric + {} one-hot)".format(
    res["features"]["n_total"], res["features"]["n_numeric"], res["features"]["n_onehot"]))
print("one-hot:", cat_names)
print("\nskew before -> after:")
for c in HEAVY:
    print("  {:<13} {:6.2f} -> {:5.2f}".format(c, skew_before[c], skew_after[c]))
f = res["final"]
print("\nfinal train shape {}, test shape {}".format(
    tuple(f["train_shape"]), tuple(f["test_shape"])))
print("train range [{:.4f}, {:.4f}]  NaN={}".format(
    f["train_min"], f["train_max"], f["train_nan"]))
print("test  range [{:.4f}, {:.4f}]  NaN={}".format(
    f["test_min"], f["test_max"], f["test_nan"]))
print("test values outside [-1,1]: {} cells in {} rows, features={}".format(
    f["test_out_of_range"], f["test_out_of_range_rows"],
    f["test_out_of_range_features"]))
