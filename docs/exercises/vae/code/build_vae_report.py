"""Render ``docs/exercises/vae/index.md`` from the two JSON result files.

Every number quoted in the report comes from ``results/vae1.json`` or
``results/vae2.json``, substituted into the template below, so no figure in the
prose can drift away from what the scripts actually computed.

    python vae_ex1.py && python vae_ex2.py && python build_vae_report.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "index.md")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def fmt_row(cells):
    """A run of Figure-5 cell labels, e.g. '4, 4, 4, 9, 9'."""
    return ", ".join(str(c) for c in cells)


def signed(value):
    """Format a grid coordinate, without printing a negative zero."""
    return f"{value + 0.0:+.2f}".replace("-0.00", "0.00")


def nearest_index(values, target):
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


TEMPLATE = """---
exercise: vae
ai_use: "Claude (Anthropic), via Claude Code, wrote the VAE implementation and the two exercise scripts, produced the six figures, and drafted this report."
---

# Variational Autoencoder

A VAE pays for two things at once: reconstructing its input, and keeping its latent space close to a
prior you can sample from. This report keeps those two terms apart from the first epoch to the last,
because a total that falls says nothing about which of the two is winning. First a 16-dimensional
latent space, then a 2-dimensional one you can actually look at --- and a measurement of what that
second dimension-count costs.

Seed `42` throughout: `torch.manual_seed(42)` runs immediately before each model is built, and the
50 000 / 10 000 split uses its own generator seeded with 42. The runs are deterministic --- the
d = 16 model was trained twice while this page was being written and reported the same test
reconstruction and KL to three decimals both times.

!!! info "Reproducing"
    ```bash
    git clone https://github.com/leosfreitas/ann-dl
    cd ann-dl
    pip install -r requirements.txt
    cd docs/exercises/vae/code
    python vae_ex1.py          # d = 16: Figures 1, 2, 3 -> results/vae1.json
    python vae_ex2.py          # d = 2:  Figures 4, 5, 6 -> results/vae2.json
    python build_vae_report.py # renders ../index.md from the two JSON files
    ```
    MNIST downloads itself into `code/data/` on the first run. `vae_ex1.py` saves the trained
    d = 16 weights to `code/vae_d16.pt`, which `vae_ex2.py` reloads for the bottom row of Figure 6.
    Training took @@T1@@ s (d = 16) and @@T2@@ s (d = 2) on a laptop CPU.

    PyTorch supplies the `Linear` layers, `Adam` and autograd. The reparameterization, both terms of
    the loss and the sampling are written out in `code/vae.py`: no VAE class from a library and no
    `torch.distributions.kl_divergence`.

## Exercise 1 --- A VAE on MNIST

### A --- Load and split the data

MNIST comes from `torchvision` with pixels left in `[0, 1]` by `transforms.ToTensor()`, flattened to
784-vectors by the training loop. The 60 000 training images are split with the generator the
statement prescribes:

```python
train, val = random_split(train_full, [50_000, 10_000],
                          generator=torch.Generator().manual_seed(42))
```

That gives **50 000** training images, **10 000** validation images, and the **10 000** official test
images as the test set. Every decision during training reads the validation split; the test set is
read exactly once per model, to produce the final numbers in 1C and 2A.

??? note "`mnist_data.py`"
    ```python
    --8<-- "docs/exercises/vae/code/mnist_data.py"
    ```

### B --- Build the model

| Part | Layers |
| :--- | :--- |
| Encoder | 784 -> 512 -> 256, ReLU after each |
| Heads | two linear layers 256 -> d, one for `mu`, one for `log sigma^2` |
| Decoder | d -> 256 -> 512 -> 784, ReLU after the hidden layers, sigmoid at the output |

With d = 16 that is **@@NPARAMS@@ parameters**. The loss per image, summed over the 784 pixels and
over the d latent dimensions, with `beta = 1`:

```text
L = - sum_j [ x_j log x̂_j + (1 - x_j) log(1 - x̂_j) ]            <- reconstruction (BCE)
    + beta * 1/2 sum_i ( mu_i² + sigma_i² - 1 - log sigma_i² )   <- KL
```

averaged over the images of the batch. Three pieces are written out by hand:

**The reparameterization.** `z = mu + sigma * eps` with `eps ~ N(0, I)` and `sigma = exp(logvar / 2)`.
Drawing `z` straight from `N(mu, sigma²)` would put the randomness between the parameters and the
loss, and no gradient could pass through it. Written this way the randomness is an *input*,
independent of the parameters, and `z` is a differentiable function of `mu` and `logvar`. The head
emits `log sigma²` rather than `sigma` so that it can output any real number and the variance stays
positive.

**The reconstruction term.** `F.binary_cross_entropy(..., reduction="sum")` divided by the batch
size --- summed over pixels, averaged over images. `reduction="mean"` would also average over the 784
pixels and shrink this term 784-fold against the KL, which is the same as secretly training with
`beta ≈ 784`. A Bernoulli likelihood is the right one here because the sigmoid output is a
per-pixel probability of being on, and MNIST pixels are nearly all 0 or 1.

**The KL.** The closed form for `KL( N(mu, diag(sigma²)) || N(0, I) )`, written term by term with
`sigma² = exp(logvar)` and `log sigma² = logvar`, summed over the d latent dimensions.

??? note "`vae.py` --- the model, the loss and the training loop"
    ```python
    --8<-- "docs/exercises/vae/code/vae.py"
    ```

### C --- Train

Adam, learning rate `1e-3`, batch size 128, 20 epochs. Both terms are recorded separately at the end
of each epoch: the training average over the epoch, and a pass over the validation split.

On the test set, in nats per image:

| Term | Value |
| :--- | ---: |
| Reconstruction (BCE) | **@@E1_RECON@@** |
| KL to the prior | **@@E1_KL@@** |
| Negative ELBO (their sum) | **@@E1_ELBO@@** |

which sits on the reference values the statement gives (around 81 nats of reconstruction and 20 of
KL).

![Figure 1](figures/fig1_curves.png)

**Figure 1.** The two terms across the 20 epochs, train and validation. Reconstruction falls
monotonically, from @@E1_VREC_FIRST@@ to @@E1_VREC_LAST@@ nats on validation. The KL does the
opposite: it *rises*, from @@E1_VKL_FIRST@@ to @@E1_VKL_LAST@@ nats, steeply over the first five
epochs and then almost flat. The train and validation curves stay together throughout --- with
50 000 images and 20 epochs this model is nowhere near overfitting.

??? note "`vae_ex1.py` --- Figures 1, 2 and 3"
    ```python
    --8<-- "docs/exercises/vae/code/vae_ex1.py"
    ```

### D --- Reconstructions and samples

![Figure 2](figures/fig2_recon.png)

**Figure 2.** The first test image of each digit 0--9 (top) and its reconstruction (bottom),
decoded from `mu` rather than from a sampled `z`. Test indices @@IDX_LIST@@. The reconstructions keep
identity and rough style but lose the sharp stroke edges; the 8 at index @@IDX8@@ is the one clear
failure, coming back with its upper loop broken open.

![Figure 3](figures/fig3_prior.png)

**Figure 3.** 64 images decoded from `z ~ N(0, I)`, drawn after `torch.manual_seed(0)`. Most are
readable digits; a handful are smeared shapes that are no digit at all.

### E --- Analysis

#### Why would the optimizer choose to pay more KL?

Because the KL is not a penalty the model is trying to dodge --- it is the price of the information
the code carries, and early in training that information is very cheap relative to what it buys.

At initialisation both heads sit near zero, so `mu ≈ 0` and `log sigma² ≈ 0`, i.e. `q(z|x) ≈ N(0, I)`
for *every* image. The KL is then near zero (the training average over epoch 1 is only
@@E1_TKL_FIRST@@ nats) --- and so is the usefulness of the code: the decoder receives essentially the
same noise whatever the input, so the best it can do is output something like the dataset average,
which is what the @@E1_TREC_FIRST@@-nat reconstruction of epoch 1 reflects. The only way to cut
reconstruction is to make `q(z|x)` genuinely depend on `x`: push `mu(x)` away from the origin so
different images get different codes, and shrink `sigma` below 1 so the draws do not blur into each
other. The first move is charged by the `mu²` term, the second by `sigma² - 1 - log sigma²`. Both
cost KL.

So the optimizer is buying latent capacity. The KL *is* an upper bound on how much information, in
nats, the code carries about the input, so "paying more KL" and "letting more of the image through
the bottleneck" are the same sentence. Over the first five epochs the exchange rate is excellent:
validation KL rises @@E1_VKL_FIRST@@ -> @@E1_VKL_E5@@ (+@@KL_GAIN_5@@ nats) while validation
reconstruction falls @@E1_VREC_FIRST@@ -> @@E1_VREC_E5@@ (-@@REC_GAIN_5@@ nats) --- about
@@EXCHANGE@@ nats of reconstruction for every nat of KL. The total falls sharply even though one of
its two terms is climbing. After that the cheap gains are gone and the KL curve flattens: from epoch
5 to epoch 20 it adds only @@KL_GAIN_LATE@@ nats while reconstruction falls a further
@@REC_GAIN_LATE@@.

#### Why Figure 3 looks worse than Figure 2

**Where the prior samples land.** The decoder is only ever trained at points the encoder produced.
Those points do not fill `N(0, I)` evenly: they leave gaps between the class clusters, and in 16
dimensions the prior's mass sits in a thin shell the encoded codes occupy only partly. Measured on
this model: the mean distance from a draw `z ~ N(0, I)` to the *nearest* of the 10 000 encoded test
`mu` is **@@D_PRIOR@@**, while the mean distance from an encoded `mu` to its nearest neighbouring
`mu` is **@@D_ENC@@** --- a prior sample typically lands @@D_RATIO@@x farther from any code the
decoder has ever seen than encoded images lie from one another. The norms differ in the same
direction: `E||z|| = @@N_PRIOR@@` against `E||mu|| = @@N_ENC@@`. A `z` in one of those gaps is a
point the decoder has no training signal for, so what it produces there is an extrapolation --- a
blend of whichever classes surround the gap. That is where Figure 3's non-digits come from. Figure 2
cannot fail this way by construction: every `z` it decodes is a `mu` the encoder has just produced
from a real image.

**What the decoder outputs for a given z.** The decoder does not return a sample from `p(x|z)`; it
returns that Bernoulli's *mean*, one probability per pixel. The reconstruction term is minimised by
predicting the conditional mean of every image consistent with `z`, and since `q(z|x)` has finite
width, many different images map near the same `z`. Averaging digits whose strokes are not aligned
leaves grey wherever the strokes disagree --- the lecture's blur. In numbers, @@MID_PRIOR@@% of the
pixels in Figure 3 sit in the uncommitted band `[0.1, 0.9]`, against @@MID_ORIG@@% for real MNIST
images.

One honest caveat, which the measurement forces. The second effect is *not* what separates the two
figures: reconstructions from `mu` sit at @@MID_RECON@@% mid-band pixels, essentially as soft as the
prior samples at @@MID_PRIOR@@%. Both figures are blurry, for the second reason. Only Figure 3 also
contains shapes that are not digits at all, and that is the first reason. The blur is the cost of a
Gaussian posterior and a mean-valued decoder; the nonsense is the cost of sampling somewhere the
decoder was never taught.

#### Why decode `mu` and not a sampled `z` in Figure 2?

Because Figure 2 asks what survives the bottleneck, and `mu` answers that and nothing else. It is the
centre of `q(z|x)`, the single code the encoder assigns to this image, so `decode(mu)` is the
reconstruction the model is aiming at. Sampling `z = mu + sigma * eps` would mix in a second,
unrelated source of variation --- the draw itself --- so the figure would change every time it was
rendered, and a reader could not tell genuine loss of information from an unlucky `eps`. Averaging
several draws would not repair it either: the mean of the decoded samples is not the decode of the
mean, because the decoder is non-linear.

The figure and the metric deliberately answer different questions. The test numbers in 1C *do* sample
`z`, because the ELBO is an expectation over `q(z|x)` and the single-sample estimate is what training
actually optimises. The figure is a portrait of the code; the number is an estimate of a bound.

## Exercise 2 --- A latent space you can see

### A --- Train a 2-D VAE

The same model, the same data, the same seed, the same `beta` and the same 20-epoch budget, with
`d = 2` in place of `d = 16`. On the test set, in nats per image:

| Term | d = 16 | d = 2 | Difference |
| :--- | ---: | ---: | ---: |
| Reconstruction (BCE) | @@E1_RECON@@ | **@@E2_RECON@@** | +@@DIFF_RECON@@ |
| KL to the prior | @@E1_KL@@ | **@@E2_KL@@** | @@DIFF_KL@@ |
| Negative ELBO | @@E1_ELBO@@ | **@@E2_ELBO@@** | +@@DIFF_ELBO@@ |

Both figures land near the statement's reference values (roughly 135 and 7). The two terms move in
opposite directions, which is the whole point: a 2-dimensional code *cannot hold* 20 nats of
information about its input, so it carries about a third as much --- and the decoder pays
@@DIFF_RECON@@ extra nats of reconstruction for the detail it is no longer told about.

### B --- The encoded test set

![Figure 4](figures/fig4_latent.png)

**Figure 4.** `mu` for all 10 000 test images, coloured by digit, with a legend. Ten classes is more
than colour alone can carry, so identity is encoded three times over: hue, marker shape, and a direct
label at each class centroid. Which two digits share a hue is not arbitrary --- the obvious pairing
(digit `d` with `d + 5`) would put 4 with 9 and 3 with 8, precisely the pairs that land on top of each
other here, so the shape encoding would buy nothing where it is needed most. Instead all 945 ways of
splitting ten digits into five pairs are enumerated and the one whose closest pair of centroids is
farthest apart wins: @@HUE_PAIRS@@, with a minimum centroid separation of @@HUE_SEP@@. The right-hand
panel repeats the same data one class at a time, which is the readable view.

The classes form wedges around the origin rather than isolated islands: 1 occupies the whole upper
left, 0 the lower middle, 2 the left, 7 the right, and 3, 5 and 8 pile up near the centre.

### C --- The decoded plane

![Figure 5](figures/fig5_manifold.png)

**Figure 5.** A 15x15 grid of latent points decoded and tiled in the same layout, with `z1`
increasing to the right and `z2` upward. The grid is *not* linearly spaced: `z = Phi^-1(q)` for 15
evenly spaced `q` in `[0.05, 0.95]`, giving the coordinates @@Z_FIRST@@ … @@Z_LAST@@ --- tightly
packed near the origin, stretched out in the tails.

**Why this spacing gives every cell the same prior mass.** Because `Phi^-1` is the inverse CDF of
`N(0, 1)`, the prior probability of the interval between two consecutive grid values is exactly the
gap between their quantiles --- a constant `0.90 / 14 ≈ @@Q_STEP@@` in each coordinate --- so every
cell of the picture covers the same amount of the prior's mass, and the plot gives equal area to
equally likely regions instead of wasting most of it on the tails.

### D --- Interpolation

![Figure 6](figures/fig6_interp.png)

**Figure 6.** Interpolation between test image **@@IDX_A@@** (a @@DIG_A@@) and test image
**@@IDX_B@@** (a @@DIG_B@@), at ten values of `t` from 0 to 1. Top row, in pixel space:
`(1 - t) x_a + t x_b`. Bottom row, in the latent space of the **d = 16** model from Exercise 1:
decode of `(1 - t) mu_a + t mu_b`.

### E --- Analysis

#### Which digits overlap, and the same transition in Figure 5

From Figure 4 the collisions are around the centre and to the right: 3, 5 and 8 sit almost on top of
each other near the origin, and 4 and 9 share a single lobe. To replace squinting with a count, a
plain 10-nearest-neighbour classifier fitted on the *validation* `mu` and evaluated on the test `mu`
reaches **@@KNN2@@%**, and its mistakes are concentrated exactly there:

| Confused pair | Test errors (d = 2) | Same pair at d = 16 |
| :--- | ---: | ---: |
@@CONF_TABLE@@

4 against 9 alone accounts for @@CONF49_SHARE@@% of all @@ERR2@@ errors the 2-D code produces.

The same transition is in Figure 5 on the right-hand side. Along the row `z2 ≈ @@ROW49_Z@@`, as `z1`
runs from @@ROW49_Z1LO@@ to @@ROW49_Z1HI@@, the cells read @@ROW49@@. The in-between images are a 4
whose two upper strokes bend toward each other and close: the open wedge at the top narrows into a
small loop while the long right-hand stroke stays where it is, so the same picture can be read as a 4
with a nearly-shut top or a 9 with a squarish head. There is no cell where one digit stops and the
other starts --- which is exactly why no classifier can separate them. The 3/5/8 pile-up shows the
same way just left of centre: along `z2 ≈ @@ROW358_Z@@`, as `z1` runs from @@ROW358_Z1LO@@
to @@ROW358_Z1HI@@, the cells read @@ROW358@@, alternating with no clean boundary --- the lower bowl of
the 3 closing into the 8's bottom loop, the top bar of the 5 rounding off.

#### What two numbers per image force the model to throw away

Style, independently of identity.

Two numbers are enough to place a digit on a sheet of prototypes, and that is precisely what Figure 5
is: at every point of the plane there is one canonical shape, and moving continuously changes it
continuously. What the plane has no room for is two different versions of the same digit. You cannot
find a narrow 8 and a round 8 in Figure 5, because every 8-looking cell belongs to the same small
region and the only way to move away from it is to stop being an 8. Writer-specific variation ---
slant, stroke thickness, how closed the loops are, how long the tail of a 2 runs --- has to share its
two axes with digit identity, so the model spends them on identity and discards the rest.

The reconstruction gap is what that costs: **@@DIFF_RECON@@ nats per image** (@@E2_RECON@@ against
@@E1_RECON@@), which is the decoder being asked to draw a digit it has only been told the rough shape
of. The k-NN probe measures the other half --- how much *identity* also fails to fit: @@KNN2@@% from
the 2-D code against **@@KNN16@@%** from the 16-D code. Even the information the 2-D space does
prioritise is incomplete, because 10 classes plus their styles will not fit in a plane: the model
keeps what reduces the loss most and drops 4-versus-9 along with everything else.

#### The two rows of Figure 6

The bottom row passes through images that look like digits. The top row does not.

Pixel-space interpolation averages two pictures with ink in different places, so for every `t`
strictly between 0 and 1 both digits are present at once at reduced brightness --- the 0's ring and
the 1's stroke superimposed, a double exposure. Nothing in that row is a digit, because the set of
digit images is not convex: the straight line between two of its members leaves it immediately. The
frames are measurably more indeterminate, too --- at the two central values of `t` the mean distance
of a pixel to the nearest of 0 or 1 is @@MIDC_PIXEL@@ for the pixel row against @@MIDC_LATENT@@ for
the latent row, @@MIDC_RATIO@@% more of the frame sitting at non-committal grey.

The bottom row interpolates between `mu_a` and `mu_b` in the latent space and then decodes, so every
frame is the decoder's output for some `z` --- and the decoder only ever learned to produce
digit-shaped images. The straight segment between two encoded images runs through a region the
encoder populates, so the frames are plausible codes rather than extrapolations, and the picture
changes by *morphing*: the 0's ring narrows, passes through something like an 8 or a filled-head 9
around `t = 0.44`--`0.56`, then thins into the 1's single stroke. One shape turns into the other
instead of fading into it. The qualification Figure 3 already supplies: the decoder's output is
always digit-*like*, not necessarily a particular digit --- but it is never a superposition of two.

??? note "`vae_ex2.py` --- Figures 4, 5 and 6"
    ```python
    --8<-- "docs/exercises/vae/code/vae_ex2.py"
    ```

## Results summary

| # | Quantity | Value |
| ---: | :--- | :--- |
| 1 | Ex. 1 --- test reconstruction, KL and negative ELBO (d = 16) | reconstruction **@@E1_RECON@@**, KL **@@E1_KL@@**, negative ELBO **@@E1_ELBO@@** nats per image |
| 2 | Ex. 1 --- validation KL at the first and at the last epoch | **@@E1_VKL_FIRST@@** -> **@@E1_VKL_LAST@@** nats per image (it rises) |
| 3 | Ex. 2 --- test reconstruction and KL (d = 2) | reconstruction **@@E2_RECON@@**, KL **@@E2_KL@@** nats per image (negative ELBO @@E2_ELBO@@) |
| 4 | Ex. 2 --- the two test indices interpolated | test index **@@IDX_A@@** (a @@DIG_A@@) and test index **@@IDX_B@@** (a @@DIG_B@@) |
"""


def main():
    e1 = load("vae1.json")
    e2 = load("vae2.json")

    h1, h2 = e1["history"], e2["history"]
    t1, t2 = e1["test"], e2["test"]
    land = e1["landing"]
    blur = e1["blur"]
    knn = e2["knn_probe"]
    interp = e2["interpolation"]

    # --- the epoch-5 exchange rate quoted in 1E -------------------------- #
    kl_1, kl_5, kl_20 = h1[0]["val_kl"], h1[4]["val_kl"], h1[-1]["val_kl"]
    rec_1, rec_5, rec_20 = h1[0]["val_recon"], h1[4]["val_recon"], h1[-1]["val_recon"]

    # --- the two runs of Figure-5 cells quoted in 2E --------------------- #
    z = e2["grid_z"]
    cells = e2["cell_labels"]          # cells[0] is the largest z2
    rows_top_down = list(reversed(z))

    r49 = nearest_index(rows_top_down, 0.33)
    c49_lo, c49_hi = nearest_index(z, 0.0), nearest_index(z, 1.2)
    row49 = cells[r49][c49_lo:c49_hi + 1]

    r358 = nearest_index(rows_top_down, 0.16)
    c358_lo, c358_hi = nearest_index(z, -0.92), nearest_index(z, 0.0)
    row358 = cells[r358][c358_lo:c358_hi + 1]

    # --- the confusion table --------------------------------------------- #
    d16_counts = {tuple(c["pair"]): c["count"]
                  for c in knn["top_confusions_d16"]}
    lines = []
    for c in knn["top_confusions_d2"][:5]:
        a, b = c["pair"]
        same = d16_counts.get((a, b))
        lines.append(f"| {a} vs {b} | {c['count']} | "
                     f"{same if same is not None else '< 26'} |")
    conf_table = "\n".join(lines)

    err2 = round((1 - knn["acc_d2"]) * 10_000)
    conf49 = knn["top_confusions_d2"][0]["count"]

    subs = {
        "T1": f"{e1['train_seconds']:.0f}",
        "T2": f"{e2['train_seconds']:.0f}",
        "NPARAMS": f"{e1['n_params']:,}".replace(",", " "),

        "E1_RECON": f"{t1['recon']:.2f}",
        "E1_KL": f"{t1['kl']:.2f}",
        "E1_ELBO": f"{t1['neg_elbo']:.2f}",
        "E2_RECON": f"{t2['recon']:.2f}",
        "E2_KL": f"{t2['kl']:.2f}",
        "E2_ELBO": f"{t2['neg_elbo']:.2f}",
        "DIFF_RECON": f"{t2['recon'] - t1['recon']:.2f}",
        "DIFF_KL": f"{t2['kl'] - t1['kl']:.2f}",
        "DIFF_ELBO": f"{t2['neg_elbo'] - t1['neg_elbo']:.2f}",

        "E1_VKL_FIRST": f"{kl_1:.2f}",
        "E1_VKL_LAST": f"{kl_20:.2f}",
        "E1_VKL_E5": f"{kl_5:.2f}",
        "E1_VREC_FIRST": f"{rec_1:.2f}",
        "E1_VREC_LAST": f"{rec_20:.2f}",
        "E1_VREC_E5": f"{rec_5:.2f}",
        "E1_TKL_FIRST": f"{h1[0]['train_kl']:.2f}",
        "E1_TREC_FIRST": f"{h1[0]['train_recon']:.0f}",
        "KL_GAIN_5": f"{kl_5 - kl_1:.2f}",
        "REC_GAIN_5": f"{rec_1 - rec_5:.2f}",
        "EXCHANGE": f"{(rec_1 - rec_5) / (kl_5 - kl_1):.1f}",
        "KL_GAIN_LATE": f"{kl_20 - kl_5:.2f}",
        "REC_GAIN_LATE": f"{rec_5 - rec_20:.2f}",

        "IDX_LIST": ", ".join(str(i) for i in e1["figure2_test_indices"]),
        "IDX8": str(e1["figure2_test_indices"][8]),

        "D_PRIOR": f"{land['nearest_mu_from_prior_sample']:.2f}",
        "D_ENC": f"{land['nearest_other_mu_from_encoded']:.2f}",
        "D_RATIO": f"{land['nearest_mu_from_prior_sample'] / land['nearest_other_mu_from_encoded']:.1f}",
        "N_PRIOR": f"{land['norm_prior_z']:.2f}",
        "N_ENC": f"{land['norm_encoded_mu']:.2f}",
        "MID_PRIOR": f"{blur['prior_samples']['frac_mid'] * 100:.1f}",
        "MID_RECON": f"{blur['reconstructions_1000']['frac_mid'] * 100:.1f}",
        "MID_ORIG": f"{blur['originals_1000']['frac_mid'] * 100:.1f}",

        "HUE_PAIRS": ", ".join(f"({a}, {b})"
                               for a, b in e2["figure4_hue_pairs"]),
        "HUE_SEP": f"{e2['figure4_hue_min_separation']:.2f}",

        "Z_FIRST": f"{z[0]:.2f}",
        "Z_LAST": f"{z[-1]:.2f}",
        "Q_STEP": f"{(0.95 - 0.05) / 14:.3f}",

        "KNN2": f"{knn['acc_d2'] * 100:.2f}",
        "KNN16": f"{knn['acc_d16'] * 100:.2f}",
        "CONF_TABLE": conf_table,
        "ERR2": str(err2),
        "CONF49_SHARE": f"{conf49 / err2 * 100:.0f}",

        "ROW49_Z": signed(rows_top_down[r49]),
        "ROW49_Z1LO": signed(z[c49_lo]),
        "ROW49_Z1HI": signed(z[c49_hi]),
        "ROW49": fmt_row(row49),
        "ROW358_Z": signed(rows_top_down[r358]),
        "ROW358_Z1LO": signed(z[c358_lo]),
        "ROW358_Z1HI": signed(z[c358_hi]),
        "ROW358": fmt_row(row358),

        "IDX_A": str(interp["idx_a"]),
        "IDX_B": str(interp["idx_b"]),
        "DIG_A": str(interp["digit_a"]),
        "DIG_B": str(interp["digit_b"]),
        "MIDC_PIXEL": f"{interp['midness_pixel_centre']:.3f}",
        "MIDC_LATENT": f"{interp['midness_latent_centre']:.3f}",
        "MIDC_RATIO": f"{(interp['midness_pixel_centre'] / interp['midness_latent_centre'] - 1) * 100:.0f}",
    }

    text = TEMPLATE
    for key, value in subs.items():
        text = text.replace("@@" + key + "@@", value)

    leftover = [tok for tok in text.split("@@") if tok.isupper() and tok.strip()]
    if leftover:
        raise SystemExit(f"unsubstituted placeholders: {sorted(set(leftover))}")

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {OUT}  ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
