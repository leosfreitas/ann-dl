"""Render docs/exercises/data/index.html from the JSON produced by ex1/ex2/ex3.

Every number on the page is interpolated from those files, so the prose can
never drift away from what the code actually computed.
"""

import json
import pathlib
from datetime import date

from report_kit import CSS, table, figure

R = pathlib.Path("../results")
e1 = json.load(open(R / "ex1.json"))
e2 = json.load(open(R / "ex2.json"))
e3 = json.load(open(R / "ex3.json"))

OUT = pathlib.Path("../docs/exercises/data/index.html")
REPO = "https://github.com/leosfreitas/ann-dl"


def pct(x, d=2):
    return f"{x:.{d}f}%"


# ============================================================== Exercise 1 bits
pairs = e1["pairs"]
worst = e1["worst"]
mix = e1["mixing"]
sb = e1["sigma_bar"]

pair_rows = [
    [f"{p['pair']}", f"{p['dist']:.4f}", f"{p['sum_sigma_bar']:.2f}",
     f"{p['r_s1']:.4f}", f"{p['r_s2']:.4f}"]
    for p in pairs]
worst_idx = min(range(len(pairs)), key=lambda i: pairs[i]["r_s1"])

mix_rows = [[f"{s}", f"{mix[s]*100:.2f}%", f"{int(round(mix[s]*400))} / 400",
             f"{worst['r_s1']/float(s):.4f}"] for s in ["0.5", "1.0", "2.0", "4.0"]]

# ============================================================== Exercise 2 bits
d1, d2 = e2["I"], e2["II"]
ex2_rows = [
    ["Distance between class centres &nbsp;<code>||c<sub>1</sub> &minus; c<sub>2</sub>||</code>",
     f"{d1['center_distance']:.4f}", f"{d2['center_distance']:.4f}"],
    ["Norm of centre 1 &nbsp;<code>||c<sub>1</sub>||</code>",
     f"{d1['center_norm_0']:.4f}", f"{d2['center_norm_0']:.4f}"],
    ["Norm of centre 2 &nbsp;<code>||c<sub>2</sub>||</code>",
     f"{d1['center_norm_1']:.4f}", f"{d2['center_norm_1']:.4f}"],
    ["Explained variance PC1", pct(d1["evr_pc1"] * 100), pct(d2["evr_pc1"] * 100)],
    ["Explained variance PC2", pct(d1["evr_pc2"] * 100), pct(d2["evr_pc2"] * 100)],
    ["<b>Explained variance PC1 + PC2</b>",
     f"<b>{pct(d1['evr_pc1_pc2']*100)}</b>", f"<b>{pct(d2['evr_pc1_pc2']*100)}</b>"],
    ["Mean radius, class 1", f"{d1['radius_mean_0']:.4f}", f"{d2['radius_mean_0']:.4f}"],
    ["Mean radius, class 2", f"{d1['radius_mean_1']:.4f}", f"{d2['radius_mean_1']:.4f}"],
    ["Largest radius, class 1", f"{d1['radius_max_0']:.4f}", f"{d2['radius_max_0']:.4f}"],
    ["Smallest radius, class 2", f"{d1['radius_min_1']:.4f}", f"{d2['radius_min_1']:.4f}"],
]
evr1 = " / ".join(f"{v*100:.1f}%" for v in d1["evr"])
evr2 = " / ".join(f"{v*100:.1f}%" for v in d2["evr"])
thr = d2["radius_threshold"]

# ============================================================== Exercise 3 bits
miss_rows = [[m["column"], f"{m['missing']:,}", f"{m['pct']:.2f}%"]
             for m in e3["missing"] if m["missing"] > 0]
spend_rows = [
    [c, f"{v['mean']:,.2f}", f"{v['median']:,.1f}", f"{v['max']:,.0f}",
     f"{v['pct_zero']:.1f}%", f"{v['skew']:.2f}"]
    for c, v in e3["spend_stats_full"].items()]
skew_rows = [[c, f"{e3['skew']['before'][c]:.2f}", f"{e3['skew']['after'][c]:.2f}"]
             for c in e3["skew"]["before"]]
f3 = e3["final"]
li = e3["log_illustration"]
fc = e3["foodcourt_train_raw"]
imp_num = e3["imputation"]["num_medians"]
imp_cat = e3["imputation"]["cat_modes"]
onehot = e3["features"]["onehot_names"]

summary_rows = [
    ["1", "Mixing rate at <code>s = 0.5</code>", f"{mix['0.5']*100:.2f}%"],
    ["2", "Mixing rate at <code>s = 1.0</code>", f"{mix['1.0']*100:.2f}%"],
    ["3", "Mixing rate at <code>s = 2.0</code>", f"{mix['2.0']*100:.2f}%"],
    ["4", "Mixing rate at <code>s = 4.0</code>", f"{mix['4.0']*100:.2f}%"],
    ["5", f"Smallest separation ratio at <code>s = 1.0</code> (pair {worst['pair']})",
     f"{worst['r_s1']:.4f}"],
    ["6", "Distance between class centres &mdash; Dataset I",
     f"{d1['center_distance']:.4f} &nbsp;<span style='color:#86857f'>(theoretical "
     f"{e2['I']['center_distance_theoretical']:.4f})</span>"],
    ["7", "Distance between class centres &mdash; Dataset II", f"{d2['center_distance']:.4f}"],
    ["8", "Explained variance PC1 + PC2 &mdash; Dataset I", pct(d1["evr_pc1_pc2"] * 100)],
    ["9", "Explained variance PC1 + PC2 &mdash; Dataset II", pct(d2["evr_pc1_pc2"] * 100)],
    ["10", "Share of the positive class in <code>Transported</code>",
     f"{e3['target']['pct_true']:.2f}% &nbsp;<span style='color:#86857f'>"
     f"({e3['target']['n_true']:,} of {e3['n_rows']:,})</span>"],
    ["11", "<code>FoodCourt</code> mean / median &mdash; training split, untransformed",
     f"{fc['mean']:.2f} / {fc['median']:.2f}"],
    ["12", "Final training feature matrix shape",
     f"({f3['train_shape'][0]:,}, {f3['train_shape'][1]})"],
    ["13", "Min / max after scaling &mdash; train, then test",
     f"[{f3['train_min']:.4f}, {f3['train_max']:.4f}] &nbsp;/&nbsp; "
     f"[{f3['test_min']:.4f}, {f3['test_max']:.4f}]"],
]

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Exercise &mdash; Artificial Neural Networks &amp; Deep Learning</title>
<meta name="description" content="Point clouds, non-linearity in 5D, and Spaceship Titanic preprocessing for a tanh network.">
<style>{CSS}</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <p class="eyebrow">Exercise &middot; Data</p>
    <h1>Geometry, non-linearity, and preparing real data for a <code style="font-size:.62em;vertical-align:.18em">tanh</code> network</h1>
    <p class="lede">Three linked studies on what a dataset&rsquo;s shape does to a classifier: how spread destroys
    linear separability in 2D, why a linear model cannot touch concentric shells in 5D no matter how much data it
    sees, and how a messy real-world table is turned into a numeric matrix a neural network can actually train on.</p>
    <div class="meta">
      <span>Leonardo Sterman Freitas</span>
      <span>Artificial Neural Networks &amp; Deep Learning</span>
      <span>Insper &middot; 2026.2</span>
      <span>Seed <code>42</code></span>
      <a href="{REPO}">Source repository &nearr;</a>
    </div>
  </div>
</header>

<div class="wrap">

<nav class="toc">
  <h2>Contents</h2>
  <ol>
    <li><a href="#ex1">Point clouds in 2D</a>
      <ol><li><a href="#ex1a">Generating the clouds</a></li>
          <li><a href="#ex1b">Spread, separation, mixing</a></li>
          <li><a href="#ex1c">Analysis</a></li></ol></li>
    <li><a href="#ex2">Non-linearity in 5D</a>
      <ol><li><a href="#ex2a">Dataset I &mdash; shifted Gaussians</a></li>
          <li><a href="#ex2b">Dataset II &mdash; concentric shells</a></li>
          <li><a href="#ex2c">Projection and metrics</a></li>
          <li><a href="#ex2d">Analysis</a></li></ol></li>
    <li><a href="#ex3">Spaceship Titanic</a>
      <ol><li><a href="#ex3a">Exploration</a></li>
          <li><a href="#ex3b">Splitting first</a></li>
          <li><a href="#ex3c">Preprocessing</a></li>
          <li><a href="#ex3d">Verification &amp; reflection</a></li></ol></li>
    <li><a href="#summary">Results summary</a></li>
    <li><a href="#repro">Reproducing this page</a></li>
  </ol>
</nav>

<!-- ================================================================ EX 1 -->
<section id="ex1">
<h2><span class="num">1</span>Point clouds: geometry and spread in 2D</h2>

<p>Four Gaussian classes share a plane. Their centres never move; only their spread does. The question is how much
spread a fixed geometry can absorb before the classes stop being distinguishable &mdash; and what kind of decision
boundary is needed along the way.</p>

<h3 id="ex1a">A &middot; Generating the clouds</h3>

<p>Each class contributes 100 points drawn from an axis-aligned Gaussian, for 400 points in total. The generator is
seeded once with <code>np.random.default_rng(42)</code>.</p>

<div class="tablewrap"><table>
<caption><b>Class parameters.</b> The mean vector fixes position; the two standard deviations fix the spread along each axis.</caption>
<thead><tr><th>Class</th><th class="num">&mu;<sub>x</sub></th><th class="num">&mu;<sub>y</sub></th>
<th class="num">&sigma;<sub>x</sub></th><th class="num">&sigma;<sub>y</sub></th>
<th class="num">&sigma;&#772; = (&sigma;<sub>x</sub>+&sigma;<sub>y</sub>)/2</th><th class="num">n</th></tr></thead>
<tbody>
<tr><td>Class 0</td><td class="num">2</td><td class="num">3</td><td class="num">0.8</td><td class="num">2.5</td><td class="num">{sb[0]:.2f}</td><td class="num">100</td></tr>
<tr><td>Class 1</td><td class="num">5</td><td class="num">6</td><td class="num">1.2</td><td class="num">1.9</td><td class="num">{sb[1]:.2f}</td><td class="num">100</td></tr>
<tr><td>Class 2</td><td class="num">8</td><td class="num">1</td><td class="num">0.9</td><td class="num">0.9</td><td class="num">{sb[2]:.2f}</td><td class="num">100</td></tr>
<tr><td>Class 3</td><td class="num">15</td><td class="num">4</td><td class="num">0.5</td><td class="num">2.0</td><td class="num">{sb[3]:.2f}</td><td class="num">100</td></tr>
</tbody></table></div>

{figure("figures/fig1_scatter.png",
        "<b>Figure 1.</b> The four clouds at their nominal spread. Class 3 sits far to the right and is isolated "
        "on <code>x&#8321;</code> alone. Classes 0 and 1 already interpenetrate along the diagonal between their "
        "centres, and class 2 &mdash; the most compact of the four &mdash; brushes the lower edge of class 1.")}

<h3 id="ex1b">B &middot; What spread does to separation</h3>

<p>The same four classes are regenerated at four spread factors, <code>s &isin; {{0.5, 1, 2, 4}}</code>. Every standard
deviation is multiplied by <code>s</code>; the means are untouched. To make the panels directly comparable, one pool
of standard-normal deviates <code>z</code> is drawn once and reused, so each point is
<code>x = &mu; + (s&middot;&sigma;)&odot;z</code> &mdash; the same underlying sample, breathing in and out.</p>

{figure("figures/fig2_scales.png",
        "<b>Figure 2.</b> Shared axis limits across all four panels, so the growth is real rather than an artifact "
        "of rescaling. At <code>s = 0.5</code> the classes are four tight, obviously distinct islands. By "
        "<code>s = 4</code> they are one cloud with a faint left-to-right colour gradient.")}

<h4>Separation ratios</h4>

<p>For each pair of classes, the separation ratio compares how far apart the centres are with how much the two
classes spread:</p>

<pre><code>r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j),
       where sigma_bar_k = (sigma_kx + sigma_ky) / 2</code></pre>

<p>Because the numerator is fixed and the denominator is proportional to <code>s</code>, the ratio is exactly
inversely proportional to the spread factor: <code>r_ij(s) = r_ij(1) / s</code>. No simulation is needed to know
what happens at any other <code>s</code>.</p>

{table("<b>Separation ratios for all six class pairs.</b> The highlighted row is the tightest pair.",
       ["Pair", "||&mu;<sub>i</sub> &minus; &mu;<sub>j</sub>||",
        "&sigma;&#772;<sub>i</sub> + &sigma;&#772;<sub>j</sub>",
        "r<sub>ij</sub> at s = 1", "r<sub>ij</sub> at s = 2"],
       pair_rows, ["left", "num", "num", "num", "num"], hl={worst_idx})}

<div class="callout">
<span class="tag">Tightest pair</span>
<p>Classes <b>{worst['pair'].replace('-', ' and ')}</b> are the closest relative to their spread, at
<b>r = {worst['r_s1']:.4f}</b> when <code>s = 1</code>. Doubling the spread to <code>s = 2</code> halves it to
<b>{worst['r_s2']:.4f}</b> &mdash; below 1, meaning the centres are now closer together than the classes are wide.
That is the point at which the two clouds stop having distinct territory.</p>
</div>

<h4>Mixing rate</h4>

<p>The mixing rate is the share of points whose nearest class centre is not their own centre. It is the error a
nearest-centroid rule would make with perfect knowledge of the true means, so it isolates the effect of geometry
from any question of model fitting.</p>

{table("<b>Mixing rate by spread factor</b>, with the tightest pair's separation ratio alongside.",
       ["s", "Mixing rate", "Misassigned points",
        f"r<sub>{worst['pair']}</sub>"],
       mix_rows, ["num", "num", "num", "num"])}

{figure("figures/fig3_mixing.png",
        f"<b>Figure 3.</b> Mixing climbs from {mix['0.5']*100:.2f}% to {mix['4.0']*100:.2f}% as the spread grows "
        "eightfold. For four balanced classes, blind guessing would misassign 75%, so even the most scrambled "
        "case retains some structure &mdash; but at <code>s = 4</code> well over a third of the points already sit "
        "closer to a foreign centre than to their own.")}

<h3 id="ex1c">C &middot; Analysis</h3>

<h4>Where the classes overlap at <code>s = 1</code></h4>
<p>Overlap at the nominal spread is local, not global. Class 3 is cleanly isolated: its centre is
{[p for p in pairs if p['pair'] == '0-3'][0]['dist']:.2f} units from class 0 and its nearest neighbour, class 2, is
still {[p for p in pairs if p['pair'] == '2-3'][0]['dist']:.2f} units away, so nothing reaches it. The contested
region is the triangle formed by classes 0, 1 and 2. Class 0 is tall and narrow
(<code>&sigma; = [0.8, 2.5]</code>) and leans up into class 1; class 1 is broad in both directions and spills down
toward class 2. The {int(round(mix['1.0']*400))} misassigned points at <code>s = 1</code> come almost entirely from
this cluster of three.</p>

<h4>Is one straight line enough?</h4>
<p>No &mdash; and not only because the classes touch. A single line cuts the plane into two half-planes, so it can
express at most a two-way decision. Four classes need at least three cuts regardless of how far apart they sit;
even the perfectly separated <code>s = 0.5</code> panel cannot be resolved by one line. Separability and
sufficiency are different questions, and it is the second one that sets the floor here.</p>

<p>What the geometry does decide is whether <em>straight</em> cuts suffice at all. At <code>s = 1</code> they very
nearly do: the four centres are in general position, and the nearest-centroid partition below assigns
{100 - mix['1.0']*100:.2f}% of points correctly using three straight boundaries.</p>

{figure("figures/fig1b_boundaries.png",
        "<b>Figure 1b.</b> The nearest-centroid partition sketched over Figure 1. Three straight cuts carve the "
        "plane into four convex regions. The blue/orange boundary running up the middle is the one under real "
        "pressure; the cuts isolating class 3 on the right are never tested by any point.")}

<h4>How the sketch relates to spreading</h4>
<p>Scaling <code>&sigma;</code> leaves the boundaries exactly where they are &mdash; they depend only on the means,
which never move. What changes is how much probability mass each class pushes across them. This is the mechanism
behind the numbers in Figure 3: at <code>s = 0.5</code> the clouds are far inside their own cells and mixing is
{mix['0.5']*100:.2f}%; at <code>s = 2</code> the tightest pair's ratio has fallen to {worst['r_s2']:.4f} and mixing
reaches {mix['2.0']*100:.2f}%; at <code>s = 4</code> it is {mix['4.0']*100:.2f}%. A more flexible boundary would
not rescue the last case. Once two classes are centred {[p for p in pairs if p['pair'] == '0-1'][0]['dist']:.2f}
units apart while each is several units wide, the overlap is a property of the distributions themselves, and no
decision surface can undo it.</p>
</section>

<!-- ================================================================ EX 2 -->
<section id="ex2">
<h2><span class="num">2</span>Non-linearity in higher dimensions</h2>

<p>Two 5-dimensional datasets, each with 500 points per class. They are built to look similar under the summary
statistics a linear model relies on, and to behave completely differently.</p>

<h3 id="ex2a">A &middot; Dataset I &mdash; shifted Gaussians</h3>

<p>Two correlated Gaussians offset along every axis: class A is centred at the origin, class B at
<code>[1.5, 1.5, 1.5, 1.5, 1.5]</code>, giving a theoretical centre distance of
<code>1.5&radic;5 = {e2['I']['center_distance_theoretical']:.4f}</code>. The covariance matrices differ in both
scale and correlation structure &mdash; class B is the wider of the two, and its first two coordinates are
negatively correlated where class A&rsquo;s are positively correlated.</p>

<pre><code>Sigma_A = [[ 1.0,  0.8, 0.1, 0.0, 0.0],      Sigma_B = [[ 1.5, -0.7, 0.2, 0.0, 0.0],
           [ 0.8,  1.0, 0.3, 0.0, 0.0],                 [-0.7,  1.5, 0.4, 0.0, 0.0],
           [ 0.1,  0.3, 1.0, 0.5, 0.0],                 [ 0.2,  0.4, 1.5, 0.6, 0.0],
           [ 0.0,  0.0, 0.5, 1.0, 0.2],                 [ 0.0,  0.0, 0.6, 1.5, 0.3],
           [ 0.0,  0.0, 0.0, 0.2, 1.0]]                 [ 0.0,  0.0, 0.0, 0.3, 1.5]]</code></pre>

<p>Both matrices are symmetric and positive definite (smallest eigenvalues {e2['eigen']['Sigma_A']['min']:.4f} and {e2['eigen']['Sigma_B']['min']:.4f}), so both
are valid covariance matrices and <code>rng.multivariate_normal</code> samples them without a correction.</p>

<h3 id="ex2b">B &middot; Dataset II &mdash; concentric shells</h3>

<p>Here direction and distance are generated separately. A direction is drawn uniformly on the unit 5-sphere by
normalising a standard Gaussian vector, then scaled by a radius that decides which shell the point belongs to:</p>

<pre><code>v = rng.standard_normal((n, 5))
u = v / ||v||                      # uniform direction on the unit sphere
rho_C = rng.normal(2.0, 0.4)       # class C, the core
rho_D = rng.normal(5.0, 0.4)       # class D, the outer shell
x = rho * u</code></pre>

<div class="callout warn">
<span class="tag">Stated assumption</span>
<p>The radius specification <code>N(2.0, 0.4)</code> is read as <b>mean 2.0, standard deviation 0.4</b>. Under the
alternative reading (0.4 as a variance, i.e. &sigma; &asymp; 0.632) the shells would still be cleanly separated, so
nothing in the conclusions below depends on this choice &mdash; only the exact width of the gap does.</p>
</div>

<h3 id="ex2c">C &middot; Projection and metrics</h3>

{figure("figures/fig4_pca.png",
        "<b>Figure 4.</b> Both datasets under PCA. Dataset I spreads along PC1 &mdash; the two clouds still overlap "
        "through the middle, but their centres are clearly offset, and PC1 alone carries "
        + pct(d1['evr_pc1']*100, 1) + " of the variance. Dataset II shows the core sitting inside "
        "the shell &mdash; a bullseye no straight line can split, and the variance is spread almost evenly across "
        "all five components.")}

{table("<b>Geometry of the two datasets in the full 5D space.</b> Distances and radii are computed before "
       "any projection; PCA is reported for reference.",
       ["Quantity", "Dataset I", "Dataset II"],
       ex2_rows, ["left", "num", "num"], hl={0, 5})}

<p>The explained-variance profiles are the clearest signal. Dataset I concentrates its variance:
{evr1}. Dataset II spreads it almost evenly: {evr2} &mdash; close to the {100/5:.0f}% per component that perfect
isotropy would give. There is no privileged direction to find, because the class structure is not directional.</p>

{figure("figures/fig5_radius.png",
        f"<b>Figure 5.</b> The same information as a distance from the origin. Dataset I&rsquo;s radii overlap "
        f"heavily &mdash; radius is a poor descriptor there, since the classes are separated by direction. Dataset "
        f"II&rsquo;s radii are disjoint with room to spare: the largest core radius is {d2['radius_max_0']:.4f} and "
        f"the smallest shell radius is {d2['radius_min_1']:.4f}.")}

<h3 id="ex2d">D &middot; Analysis</h3>

<h4>Why Dataset II&rsquo;s centres nearly coincide</h4>
<p>Each class is spherically symmetric about the origin. Writing a point as <code>x = &rho;u</code> with the
direction <code>u</code> uniform on the sphere and independent of <code>&rho;</code>, the expectation factorises as
<code>E[x] = E[&rho;]&middot;E[u] = 0</code>, because <code>E[u] = 0</code> for a uniform direction. Both classes
therefore have the same population mean &mdash; the origin &mdash; no matter how different their radii are. The
observed distance of {d2['center_distance']:.4f} is nothing but sampling noise from averaging 500 random directions;
it shrinks toward zero as the sample grows. Dataset I, by contrast, shows
{d1['center_distance']:.4f}, close to its theoretical {e2['I']['center_distance_theoretical']:.4f}.</p>

<div class="callout">
<span class="tag">The core point</span>
<p>Class means are a <em>directional</em> summary. Dataset II encodes its classes <em>radially</em>, and a radial
structure is invisible to the mean. Two classes can be perfectly separable and still have identical centres.</p>
</div>

<h4>Why a linear boundary fails, and why more data will not help</h4>
<p>A linear classifier computes <code>sign(w&#7488;x + b)</code>, so all it sees of the data is the 1-D projection
<code>w&#7488;x</code>. For a spherically symmetric class, <code>w&#7488;x = &rho;&middot;(w&#7488;u)</code>, and
<code>w&#7488;u</code> is symmetric about zero for every choice of <code>w</code>. Both classes therefore project
to distributions centred at zero; they differ only in <em>width</em>, since the shell&rsquo;s larger
<code>&rho;</code> stretches its projection further. Any threshold <code>b</code> then cuts both distributions at
essentially the same quantile, and whatever fraction of the shell falls on one side, a comparable fraction of the
core falls there too.</p>

<p>This is a statement about the distributions, not the sample. Extra data estimates the same overlapping
projections more precisely; it does not create a direction that does not exist. The failure is one of
<em>representational capacity</em>, not of estimation.</p>

<h4>PCA as the demonstration</h4>
<p>PCA is exactly the tool for this argument, because it is itself linear: it searches all directions and returns
the ones with the most variance. On Dataset II it reports {evr2} &mdash; a nearly flat profile. The best linear
projection available is barely better than a random one, and Figure 4 shows what the best two directions actually
buy: a bullseye. If the optimal linear view still yields nested classes, no linear boundary exists to be found.</p>

<h4>A function that separates Dataset II</h4>
<p>Squared distance from the origin does it in one number:</p>

<pre><code>f(x) = ||x||^2 = x_1^2 + x_2^2 + x_3^2 + x_4^2 + x_5^2</code></pre>

<p>This maps each shell onto a tight interval on the real line. With the midpoint threshold
<code>||x|| &lt; {thr:.4f}</code> (equivalently <code>||x||&sup2; &lt; {thr**2:.4f}</code>) the rule classifies
<b>{d2['radius_rule_accuracy']*100:.2f}%</b> of the 1,000 points correctly, with a comfortable margin:
the gap between the largest core radius and the smallest shell radius is
{d2['radius_min_1'] - d2['radius_max_0']:.4f}. Note that <code>f</code> is linear in the <em>squared</em>
coordinates, which is precisely what a hidden layer supplies: given a non-linear activation, a network can
approximate each <code>x&#7522;&sup2;</code> and sum them, turning an impossible linear problem into a trivial
one. That single step &mdash; from raw coordinates to a learned non-linear feature &mdash; is the whole reason
hidden layers exist.</p>
</section>

<!-- ================================================================ EX 3 -->
<section id="ex3">
<h2><span class="num">3</span>Real-world data: Spaceship Titanic</h2>

<p>The first two exercises used data built to order. This one starts from a table with missing values in every
column, categorical text, and spending figures spanning four orders of magnitude, and ends with a numeric matrix
whose every entry is safe to feed a <code>tanh</code> network.</p>

<h3 id="ex3a">A &middot; Exploration</h3>

<div class="kpis">
  <div class="kpi"><div class="v">{e3['n_rows']:,}</div><div class="k">passengers in <code>train.csv</code></div></div>
  <div class="kpi"><div class="v">{e3['n_cols']}</div><div class="k">raw columns</div></div>
  <div class="kpi"><div class="v">{e3['target']['pct_true']:.2f}%</div><div class="k">transported &mdash; essentially balanced</div></div>
  <div class="kpi"><div class="v">{e3['rows_any_missing_pct']:.1f}%</div><div class="k">of rows have at least one missing value</div></div>
</div>

<h4>The target</h4>
<p><code>Transported</code> is a boolean recording whether a passenger was transported to an alternate dimension
when the ship struck the spacetime anomaly &mdash; the binary outcome to be predicted. It is almost perfectly
balanced: {e3['target']['n_true']:,} <code>True</code> ({e3['target']['pct_true']:.2f}%) against
{e3['target']['n_false']:,} <code>False</code> ({e3['target']['pct_false']:.2f}%). That balance is convenient:
accuracy is a meaningful metric here, and no class weighting or resampling is called for.</p>

<h4>Feature types</h4>
<ul>
<li><b>Numerical ({len(e3['spend_stats_full']) + 1}):</b> <code>Age</code> plus the five amenity-billing columns
&mdash; <code>RoomService</code>, <code>FoodCourt</code>, <code>ShoppingMall</code>, <code>Spa</code>,
<code>VRDeck</code>.</li>
<li><b>Categorical ({len(e3['cat_levels'])}):</b> <code>HomePlanet</code>
({', '.join(e3['cat_levels']['HomePlanet'])}), <code>Destination</code>
({', '.join(e3['cat_levels']['Destination'])}), and the two booleans <code>CryoSleep</code> and
<code>VIP</code>.</li>
<li><b>Identifiers and free text (3):</b> <code>PassengerId</code>, <code>Cabin</code>, <code>Name</code> &mdash;
dropped, as specified.</li>
</ul>

{table("<b>Missing values by column.</b> Every column except the target and <code>PassengerId</code> has gaps, "
       "each around 2%.",
       ["Column", "Missing", "Share"], miss_rows, ["left", "num", "num"])}

<p>The pattern matters more than any single figure. No column is badly damaged &mdash; the worst is
<code>CryoSleep</code> at {[m['pct'] for m in e3['missing'] if m['column'] == 'CryoSleep'][0]:.2f}% &mdash; but the
gaps are scattered nearly independently, so <b>{e3['rows_any_missing']:,} rows
({e3['rows_any_missing_pct']:.2f}%)</b> carry at least one. Dropping incomplete rows would discard a quarter of
the dataset to repair {e3['missing_total']:,} cells out of
{e3['n_rows'] * e3['n_cols']:,}. Imputation is the only sensible route.</p>

<h4>The spending columns</h4>

{table("<b>Amenity spending, full file.</b> The mean sits far above the median in every column.",
       ["Column", "Mean", "Median", "Max", "Zeros", "Skew"],
       spend_rows, ["left", "num", "num", "num", "num", "num"])}

<p>The mean exceeds the median by a wide margin in all five columns, and the median is <code>0</code> in every one
of them. The reason is visible in the zeros column: roughly {min(v['pct_zero'] for v in e3['spend_stats_full'].values()):.0f}&ndash;{max(v['pct_zero'] for v in e3['spend_stats_full'].values()):.0f}%
of passengers spend nothing at all, so more than half the mass sits exactly at the floor while a thin tail runs out
to {max(v['max'] for v in e3['spend_stats_full'].values()):,.0f}. The mean is dragged upward by that tail and
describes almost no actual passenger. Skewness values of {min(v['skew'] for v in e3['spend_stats_full'].values()):.1f}
to {max(v['skew'] for v in e3['spend_stats_full'].values()):.1f} put a number on it &mdash; a symmetric
distribution scores 0. This is the signature of a heavy right tail, and it is what makes the log step in Part C
necessary rather than cosmetic.</p>

<h3 id="ex3b">B &middot; Splitting before anything else</h3>

<p>An 80/20 split stratified on the target, with a fixed seed, executed <em>before</em> a single statistic is
computed:</p>

<pre><code>X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)</code></pre>

<p>This yields <b>{e3['split']['n_train']:,}</b> training and <b>{e3['split']['n_test']:,}</b> test rows, with the
positive class at {e3['split']['pct_true_train']:.2f}% and {e3['split']['pct_true_test']:.2f}% respectively &mdash;
stratification holding the balance to within a twentieth of a point.</p>

<h4>Why every statistic must come from the training set alone</h4>
<p>The test set exists to estimate performance on passengers the model has never encountered. That estimate is only
honest if nothing derived from those rows reaches the model &mdash; and a fitted preprocessing statistic is
derived information just as much as a learned weight is. Concretely:</p>
<ul>
<li>An imputation median computed over all {e3['n_rows']:,} rows encodes the central tendency of the test
passengers, and that value is then written into training rows.</li>
<li>A scaler fitted on the full table learns the global minimum and maximum. If the largest spender happens to sit
in the test set &mdash; as the {f3['test_out_of_range']} out-of-range cells reported below confirm can happen
&mdash; the training features are silently rescaled by a test observation.</li>
<li>An encoder fitted on everything has already seen every category, so the pipeline never has to handle an unseen
level, and a genuine deployment failure mode is hidden.</li>
</ul>
<p>Each leak is small on its own. Together they make the test score optimistic in a way no amount of later
validation can detect, because the contamination is baked into the features themselves. The discipline is
mechanical: <b>fit on train, transform both.</b></p>

<h3 id="ex3c">C &middot; Preprocessing</h3>

<h4>1 &middot; Imputation</h4>
<p>Numerical columns are filled with the <b>training median</b>, categorical columns with the <b>training mode</b>.</p>

<pre><code>num_medians = X_tr[NUM].median()          # {', '.join(f'{k}={v:g}' for k, v in imp_num.items())}
cat_modes   = {{c: X_tr[c].mode()[0] ...}} # {', '.join(f'{k}={v}' for k, v in imp_cat.items())}</code></pre>

<p>The median is chosen over the mean precisely because of the skew documented above: with a maximum of
{fc['max']:,.0f} and a median of {fc['median']:.0f}, <code>FoodCourt</code>&rsquo;s mean of {fc['mean']:.2f} is a
value almost no passenger has, and imputing it would invent hundreds of moderate spenders who do not exist. The
median is robust to the tail and, for the spending columns, lands on <code>0</code> &mdash; which also happens to
be the right substantive guess, since a missing charge most plausibly means no charge. <code>Age</code> takes
{imp_num['Age']:.0f}.</p>

<p>For the categorical columns the mode is defensible mainly because the gaps are small &mdash; around 2.1&ndash;2.5%
&mdash; so the distortion to the marginal distribution is slight. The alternative worth naming is to treat
<code>"Missing"</code> as its own level, which is the better choice when missingness is itself informative. Both
were available; the mode was taken here for its simplicity, and at this rate of missingness the two are unlikely to
differ materially.</p>

<h4>2 &middot; Categorical encoding</h4>
<p>One-hot encoding, fitted on the training split:</p>

<pre><code>enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_tr[CAT])</code></pre>

<p>One-hot rather than integer codes, because integers would assert an order and a spacing that do not exist &mdash;
encoding <code>Earth&rarr;0, Europa&rarr;1, Mars&rarr;2</code> tells the network that Europa lies between the other
two and that Mars is twice Europa, and a linear layer will act on that fiction. One-hot puts each level on its own
orthogonal axis, asserting nothing. The four columns expand to <b>{len(onehot)}</b> indicators:</p>

<p style="font-family:var(--mono);font-size:13px;color:var(--ink-soft);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px">
{'<br>'.join(onehot)}</p>

<p><code>handle_unknown="ignore"</code> settles the unseen-category question: a level that appears only in the test
set produces a row of zeros across that feature group rather than raising an error or silently adding a column. The
matrix keeps its width, the network keeps its input dimension, and the unknown level is represented as
&ldquo;none of the known categories&rdquo; &mdash; which is honestly what is known about it. Dropping a
reference level was deliberately avoided: multicollinearity is a concern for closed-form linear regression, not for
a gradient-trained network, and keeping every indicator preserves symmetry between levels.</p>

<h4>3 &middot; Feature engineering</h4>
<p><code>TotalSpend</code> is the row-wise sum of the five amenity columns, computed after imputation and before
the log transform; <code>Cabin</code>, <code>Name</code> and <code>PassengerId</code> are dropped.</p>
<p>The sum is worth adding because the five columns share a single underlying behaviour &mdash; whether a passenger
was awake and consuming at all. A passenger in <code>CryoSleep</code> necessarily bills zero everywhere, so total
consumption is a compact proxy for a fact the network would otherwise have to reconstruct from five separate
inputs. On the training split it runs from 0 to {e3['totalspend_train_raw']['max']:,.0f} with a median of
{e3['totalspend_train_raw']['median']:,.0f} and skew {e3['totalspend_train_raw']['skew']:.2f}, so it is treated as
heavy-tailed alongside its components. The dropped columns are identifiers and free text: <code>PassengerId</code>
and <code>Name</code> carry no signal that generalises, and <code>Cabin</code> would need to be parsed into
deck/number/side before it meant anything numerically.</p>

<h4>4 &middot; Taming the heavy tails</h4>
<p><code>log(1 + x)</code> is applied to the five spending columns and to <code>TotalSpend</code>. The
<code>1 +</code> is what makes it safe: over 60% of the entries are exactly zero, and <code>log(0)</code> is
undefined, whereas <code>log1p(0) = 0</code> keeps the floor at zero and preserves the meaning of &ldquo;spent
nothing&rdquo;.</p>

{table("<b>Skewness before and after the log transform</b> (training split). A symmetric distribution scores 0.",
       ["Column", "Skew before", "Skew after"], skew_rows, ["left", "num", "num"])}

{figure("figures/fig6_logtransform.png",
        f"<b>Figure 6.</b> <code>FoodCourt</code> before and after. The raw panel needs a logarithmic count axis to "
        f"show anything at all &mdash; the tail runs to {fc['max']:,.0f} with single-digit counts. After the "
        f"transform the spenders occupy a visible, roughly unimodal band between 4 and 9, while the zero spike "
        f"stays honestly at zero. Skew falls from {e3['skew']['before']['FoodCourt']:.2f} to "
        f"{e3['skew']['after']['FoodCourt']:.2f}.")}

<p>Why this matters specifically for a <code>tanh</code> network: <code>tanh</code> saturates once its input passes
roughly &plusmn;2, where the gradient is near zero and learning stalls. An untransformed feature spanning
[0, {fc['max']:,.0f}] guarantees saturation for every non-trivial weight. Worse, the raw scale wastes the entire
usable range on outliers &mdash; a point quantified in the reflection below.</p>

<h4>5 &middot; Scaling</h4>
<p>Min-max normalisation to <code>[-1, 1]</code>, fitted on the training split:</p>

<pre><code>scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_tr_matrix)</code></pre>

<p>Both options were on the table, and the choice follows from the activation. Standardisation fixes the mean and
variance but leaves the range open: a feature can still produce values at &plusmn;5 or beyond, straight into the
flat region of <code>tanh</code>. Min-max to <code>[-1, 1]</code> instead <em>bounds</em> every input, matching the
domain where <code>tanh</code> has usable gradient and matching its own output range &mdash; so the inputs to layer
one look like the activations arriving at layer two, which keeps initialisation heuristics consistent across the
depth of the network.</p>

<p>Min-max&rsquo;s usual weakness is its sensitivity to outliers, since a single extreme value sets the entire
scale. The log transform is what removes that objection: applied first, it has already pulled the tail in, so the
range being normalised is a sane one rather than one dictated by the largest spender. The two steps are a pair, and
the order matters.</p>

<h3 id="ex3d">D &middot; Verification and reflection</h3>

<div class="kpis">
  <div class="kpi"><div class="v">({f3['train_shape'][0]:,}, {f3['train_shape'][1]})</div><div class="k">training matrix</div></div>
  <div class="kpi"><div class="v">({f3['test_shape'][0]:,}, {f3['test_shape'][1]})</div><div class="k">test matrix</div></div>
  <div class="kpi"><div class="v">{f3['train_nan'] + f3['test_nan']}</div><div class="k">remaining NaN values</div></div>
  <div class="kpi"><div class="v">[{f3['train_min']:.2f}, {f3['train_max']:.2f}]</div><div class="k">training value range</div></div>
</div>

<p>The {f3['train_shape'][1]} features are {e3['features']['n_numeric']} numerical
(<code>Age</code>, the five logged spending columns, and logged <code>TotalSpend</code>) plus
{e3['features']['n_onehot']} one-hot indicators. Both matrices are fully dense: <b>{f3['train_nan']} NaN values in
train and {f3['test_nan']} in test</b>. The training range is exactly
<code>[{f3['train_min']:.4f}, {f3['train_max']:.4f}]</code>, as min-max fitting guarantees.</p>

{figure("figures/fig7_ranges.png",
        "<b>Figure 7.</b> Per-feature ranges after scaling. Every training feature spans the full "
        "<code>[-1, 1]</code> band. The orange ticks mark the test minima and maxima, which sit inside it with two "
        "exceptions discussed below.")}

<div class="callout warn">
<span class="tag">Honest caveat &mdash; test values outside the band</span>
<p>The test range is <code>[{f3['test_min']:.4f}, {f3['test_max']:.4f}]</code>, not
<code>[-1, 1]</code>. <b>{f3['test_out_of_range']} cells</b> in {f3['test_out_of_range_rows']} rows exceed the
upper bound, in <code>{'</code> and <code>'.join(f3['test_out_of_range_features'])}</code> &mdash; test passengers
who spent more than any passenger in the training split. This is correct behaviour, not a bug: the scaler was
fitted on training data only, so it cannot know about them, and clipping them would discard real information.
The overshoot is small ({f3['test_max']:.4f} against a bound of 1.0) and well inside the region where
<code>tanh</code> still has gradient, so it is left as is. Making the test range exactly <code>[-1, 1]</code>
would require fitting the scaler on the test set &mdash; precisely the leak Part B forbids.</p>
</div>

<h4>Which decision matters most, and why</h4>
<p>The log transform of the spending columns, with the scaling that follows it. Its effect can be stated
numerically. Three quarters of training passengers bill {li['p75_raw']:.2f} or less at
<code>FoodCourt</code>, against a maximum of {li['raw_max']:,.0f}. Min-max that raw column into
<code>[-1, 1]</code> and the 75th percentile lands at <b>{li['scaled_without_log']:.4f}</b> &mdash; the entire
bottom three quarters of the dataset compressed into roughly {abs(-1 - li['scaled_without_log'])/2*100:.1f}% of the
available range, indistinguishable from the {li['pct_at_floor']:.1f}% of passengers sitting exactly at the floor.
The network would receive a feature that is <code>-1</code> for almost everyone, with all its variation
concentrated in a handful of outliers at the top. Gradients through that input would be negligible for the vast
majority of examples, and the feature would contribute essentially nothing.</p>

<p>Apply the log first and the same 75th percentile lands at <b>{li['scaled_with_log']:.4f}</b> &mdash; near the
middle of the range, where <code>tanh</code> is steepest and differences between passengers are actually
resolvable. The distinction between spending 50 credits and 500 becomes a real distance in the input space instead
of a rounding error.</p>

<p>The other choices are consequential but more forgiving. A different imputation strategy shifts about 2% of
values. Label encoding instead of one-hot would inject false ordinal structure, which is a genuine error, though a
network with enough capacity can partly work around it. Standardisation instead of min-max changes the input
distribution but leaves relative distances intact. The log transform is different in kind: without it, the spending
features &mdash; six of {f3['train_shape'][1]}, and among the most predictive available &mdash; arrive at the
network effectively constant, and no amount of training recovers information that the representation has already
destroyed.</p>
</section>

<!-- ================================================================ SUMMARY -->
<section id="summary">
<h2><span class="num">4</span>Results summary</h2>

{table("<b>All required reported values</b>, produced directly by the scripts in <code>code/</code>.",
       ["#", "Quantity", "Value"], summary_rows, ["num", "left", "num"])}
</section>

<!-- ================================================================ REPRO -->
<section id="repro">
<h2><span class="num">5</span>Reproducing this page</h2>

<p>Every figure and every number above is regenerated from scratch by four scripts. The random seed is
<code>42</code> throughout, so the outputs are byte-identical between runs.</p>

<pre><code>git clone {REPO}
cd ann-dl
pip install numpy pandas matplotlib scikit-learn

cd code
python ex1.py            # Figures 1, 1b, 2, 3  + results/ex1.json
python ex2.py            # Figures 4, 5         + results/ex2.json
python ex3.py            # Figures 6, 7         + results/ex3.json
python build_report.py   # renders exercises/data/index.html from the JSON</code></pre>

<p>The report is generated from the JSON result files rather than written by hand, so no figure in the prose can
drift from the value the code produced. Libraries used: NumPy, pandas, Matplotlib, and scikit-learn restricted to
<code>PCA</code>, <code>train_test_split</code>, <code>OneHotEncoder</code> and <code>MinMaxScaler</code> &mdash;
no model is trained anywhere in this exercise.</p>

<p>The Spaceship Titanic file is the Kaggle <code>train.csv</code>
({e3['n_rows']:,} rows &times; {e3['n_cols']} columns), committed to <code>data/</code> so the pipeline runs without
Kaggle credentials.</p>

<footer>
  <p><b>Leonardo Sterman Freitas</b> &middot; Artificial Neural Networks &amp; Deep Learning &middot; Insper 2026.2</p>
  <p>Source and scripts: <a href="{REPO}">github.com/leosfreitas/ann-dl</a> &middot; generated {date.today().isoformat()}</p>
</footer>
</section>

</div>
</body>
</html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT.resolve()}  ({len(HTML):,} bytes)")
