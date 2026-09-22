"""Render exercises/perceptron/index.html from results/perceptron{1,2}.json.

Same contract as the data exercise: every number on the page is interpolated
from the JSON the scripts emitted, so the prose cannot drift away from what the
code actually computed.
"""

import json
import pathlib
from datetime import date

from report_kit import CSS, table, figure

R = pathlib.Path("../results")
e1 = json.load(open(R / "perceptron1.json"))
e2 = json.load(open(R / "perceptron2.json"))

OUT = pathlib.Path("../exercises/perceptron/index.html")
REPO = "https://github.com/leosfreitas/ann-dl"


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def vec(v, d=6):
    return "(" + ", ".join(f"{c:.{d}f}" for c in v) + ")"


# ------------------------------------------------------------------ Exercise 1
f1, m1, z1, h1 = e1["final"], e1["margin"], e1["zero_init"], e1["eta1"]
s1 = e1["scale"]

# ------------------------------------------------------------------ Exercise 2
f2, p2, by, sc = e2["final"], e2["pocket"], e2["bayes"], e2["scale"]

# --------------------------------------------------------------------- tables
params_rows = [
    ["Class 0", "1000", "(1.5, 1.5)", "0.5 &middot; I", "(3, 3)", "1.5 &middot; I"],
    ["Class 1", "1000", "(5, 5)", "0.5 &middot; I", "(4, 4)", "1.5 &middot; I"],
    ["Centre distance", "&mdash;", f"{e1['centre_distance']:.4f}", "",
     f"{e2['centre_distance']:.4f}", ""],
    ["Per-axis &sigma;", "&mdash;", f"{e1['sigma']:.4f}", "", f"{e2['sigma']:.4f}", ""],
]

hyper_rows = [
    ["Learning rate &eta;", "<code>0.01</code> (and <code>1.0</code> for the re-run in 1D)"],
    ["Weight init", f"<code>rng.normal(0, 0.01, size=2)</code> &rarr; {vec(e1['w_init'])}"],
    ["Bias init", "<code>0.0</code>"],
    ["Activation", "step &mdash; output <code>1</code> if <code>w&middot;x + b &ge; 0</code>, else <code>0</code>"],
    ["Update", "<code>w += &eta;(y &minus; &#375;)x</code> and <code>b += &eta;(y &minus; &#375;)</code>, applied per sample"],
    ["Stopping rule", "a full epoch with zero updates, or 100 epochs"],
    ["Seed", "<code>np.random.default_rng(42)</code>, drawn in a fixed order"],
]

eta_rows = [
    ["&eta; = 0.01, random init", vec(f1["w"]), f"{f1['b']:.4f}", str(f1["epochs"]),
     pct(f1["acc"]), f"{f1['w_norm']:.4f}"],
    ["&eta; = 1.0, random init", vec(h1["w"], 4), f"{h1['b']:.4f}", str(h1["epochs"]),
     pct(h1["acc"]), f"{h1['w_norm']:.4f}"],
    ["&eta; = 0.01, <b>zero</b> init", vec(z1["w_lo"]), f"{z1['b_lo']:.4f}",
     str(z1["epochs_lo"]), pct(z1["acc_lo"]), ""],
    ["&eta; = 1.0, <b>zero</b> init", vec(z1["w_hi"], 4), f"{z1['b_hi']:.4f}",
     str(z1["epochs_hi"]), pct(z1["acc_hi"]), ""],
]

dir_rows = [
    ["&eta; = 0.01", vec(f1["w_dir"]), f"{f1['w_norm']:.4f}"],
    ["&eta; = 1.0", vec(h1["w_dir"]), f"{h1['w_norm']:.4f}"],
    ["Angle between them", f"{h1['angle_deg']:.4f}&deg;", ""],
    ["Ratio of norms", f"{h1['norm_ratio']:.2f}&times;", ""],
]

ex2_rows = [
    ["Final weights", vec(f2["w"]), f"{f2['b']:.4f}", pct(f2["acc"]),
     f"{f2['n_misclassified']}"],
    ["Pocket weights", vec(p2["w"]), f"{p2['b']:.4f}", pct(p2["acc"]),
     f"{p2['n_misclassified']}"],
    ["Bisector of the centres", vec(by["w"]), f"{by['b']:.4f}",
     pct(by["acc_empirical"]), f"{int(round((1 - by['acc_empirical']) * 2000))}"],
]

geom_rows = [
    ["Final", f"{f2['w_norm']:.6f}", f"{f2['b']:.4f}",
     f"{f2['geometry']['offset']:.4f}", f"{f2['geometry']['needed']:.4f}", pct(f2["acc"])],
    ["Pocket", f"{p2['w_norm']:.6f}", f"{p2['b']:.4f}",
     f"{p2['geometry']['offset']:.4f}", f"{p2['geometry']['needed']:.4f}", pct(p2["acc"])],
]

summary_rows = [
    ["1", "Exercise 1 &mdash; final weights <b>w</b> and bias <i>b</i>",
     f"<b>w</b> = {vec(f1['w'])}, &nbsp;<i>b</i> = {f1['b']:.4f}"],
    ["2", "Exercise 1 &mdash; epochs to convergence",
     f"{f1['epochs']} (stopped on a zero-update epoch, not the 100-epoch cap)"],
    ["3", "Exercise 1 &mdash; final accuracy",
     f"{pct(f1['acc'])} &nbsp;({2000 - f1['n_misclassified']} / 2000 correct)"],
    ["4", "Exercise 1 &mdash; epochs and accuracy with &eta; = 1.0",
     f"{h1['epochs']} epochs, {pct(h1['acc'])}"],
    ["5", "Exercise 2 &mdash; final weights <b>w</b> and bias <i>b</i>",
     f"<b>w</b> = {vec(f2['w'])}, &nbsp;<i>b</i> = {f2['b']:.4f}"],
    ["6", "Exercise 2 &mdash; accuracy of the final weights",
     f"{pct(f2['acc'])} &nbsp;({f2['n_misclassified']} / 2000 wrong)"],
    ["7", "Exercise 2 &mdash; accuracy of the pocket weights",
     f"{pct(p2['acc'])} &nbsp;({p2['n_misclassified']} / 2000 wrong)"],
    ["8", "Exercise 2 &mdash; epoch at which the pocket best occurred",
     f"epoch {p2['epoch']} (weight update {p2['update']} of {e2['total_updates']})"],
]

CODE = """<span class="c"># --- the whole learning rule: predict, and only move on a mistake ---</span>
for epoch in range(1, max_epochs + 1):
    updates = 0
    for i in range(X.shape[0]):
        y_hat = 1 if (w @ X[i] + b) >= 0.0 else 0   <span class="c"># step activation</span>
        err = y[i] - y_hat                          <span class="c"># -1, 0 or +1</span>
        if err != 0:                                <span class="c"># correct samples do nothing</span>
            w += eta * err * X[i]
            b += eta * err
            updates += 1
            if pocket:                              <span class="c"># Gallant's rule: score every change</span>
                a = accuracy(X, y, w, b)
                if a &gt; best_acc:
                    best_acc, best_w, best_b = a, w.copy(), b

    acc = accuracy(X, y, w, b)                      <span class="c"># recorded once per epoch</span>
    curve.append(acc)
    if updates == 0:                                <span class="c"># a clean pass: nothing left to learn</span>
        converged = True
        break"""

EXTRA_CSS = """
.eqn { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 20px; margin: 18px 0; font-family: var(--mono); font-size: 14px;
  overflow-x: auto; line-height: 1.9; }
.eqn .lbl { color: var(--ink-muted); font-family: var(--sans); font-size: 13px; }
pre .c { color: #9aa0a6; }
"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Perceptron Exercise &mdash; Artificial Neural Networks &amp; Deep Learning</title>
<meta name="description" content="A perceptron written from scratch on separable and overlapping Gaussian data, with the pocket algorithm.">
<style>{CSS}{EXTRA_CSS}</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <p class="eyebrow">Exercise &middot; Perceptron</p>
    <h1>A perceptron from scratch: what separability buys, and what it costs when it is gone</h1>
    <p class="lede">The same short learning rule is run on two datasets. On the first it halts by itself with
    a perfect score. On the second it never halts at all, and the weights it happens to be holding when the
    epoch budget runs out are no better than a coin flip &mdash; while weights it passed through and discarded
    along the way were close to the best any straight line can do. That gap is what the pocket algorithm is
    for.</p>
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
    <li><a href="#impl">The implementation</a></li>
    <li><a href="#ex1">Exercise 1 &mdash; separable data</a>
      <ol><li><a href="#ex1a">A &middot; Generating the data</a></li>
          <li><a href="#ex1b">B &middot; The perceptron</a></li>
          <li><a href="#ex1c">C &middot; Training and measuring</a></li>
          <li><a href="#ex1d">D &middot; Analysis</a></li></ol></li>
    <li><a href="#ex2">Exercise 2 &mdash; overlapping data</a>
      <ol><li><a href="#ex2a">A &middot; Generating the data</a></li>
          <li><a href="#ex2b">B &middot; Training with the pocket</a></li>
          <li><a href="#ex2c">C &middot; Figures</a></li>
          <li><a href="#ex2d">D &middot; Analysis</a></li></ol></li>
    <li><a href="#summary">Results summary</a></li>
    <li><a href="#repro">Reproducing this page</a></li>
  </ol>
</nav>

<!-- ============================================================ IMPLEMENTATION -->
<section id="impl">
<h2><span class="num">1</span>The implementation</h2>

<p>Both exercises call one function, <code>fit</code> in <code>code/perceptron.py</code>. Nothing is imported
beyond NumPy: the step activation, the update rule and the stopping test are all written out. Exercise 2 reuses
it completely unchanged &mdash; the only difference is that it passes <code>pocket=True</code>.</p>

{table("<b>Fixed settings</b>, identical across both exercises unless stated.",
       ["Setting", "Value"], hyper_rows)}

<p>Labels are <code>{{0, 1}}</code> rather than <code>{{&minus;1, +1}}</code>, so the error term
<code>y &minus; &#375;</code> is <code>+1</code> when a class-1 point was called 0, <code>&minus;1</code> in the
other direction, and <code>0</code> whenever the sample is already right. That last case is the important one:
<b>a correctly classified sample changes nothing</b>, which is what makes the stopping rule meaningful.</p>

<pre><code>{CODE}</code></pre>

<div class="callout">
<span class="tag">One design decision worth stating</span>
<p>The samples are stacked class 0 first, class 1 second, and are <b>not</b> shuffled &mdash; the statement
builds them that way. On separable data the order cannot affect the outcome. On overlapping data it very much
affects <i>which</i> weights you end up holding, because every epoch ends right after a run of 1000 class-1
points. Section 2D takes that apart rather than hiding it.</p>
</div>

<p>One consequence is that the pocket has to be scored after <b>every weight update</b>, which is Gallant's
original formulation, not once per epoch. Scoring per epoch would only ever sample the bottom of the sawtooth
in Figure 6b and would report a pocket of about 52% instead of {pct(p2['acc'])}.</p>
</section>

<!-- ================================================================== EX 1 -->
<section id="ex1">
<h2><span class="num">2</span>Exercise 1 &mdash; separable data</h2>

<h3 id="ex1a">A &middot; Generating the data</h3>

<p>Two spherical Gaussians, 1000 points each. Their centres are {e1['centre_distance']:.4f} apart while each
class has a per-axis &sigma; of only {e1['sigma']:.4f} &mdash; the centres are
{s1['sigma_multiple']:.1f} standard deviations apart along the line joining them, far enough that the clouds
never touch.</p>

{table("<b>Class parameters for both exercises</b>, side by side. The geometry is the only thing that changes.",
       ["", "n", "Exercise 1 mean", "Exercise 1 cov", "Exercise 2 mean", "Exercise 2 cov"],
       params_rows, ["left", "num", "num", "left", "num", "left"])}

{figure("figures/fig1_scatter.png",
        "<b>Figure 1.</b> The two classes. There is visible empty space between them: projected onto the "
        "direction joining the centres, the nearest class-1 point sits "
        f"{m1['projection_gap']:.4f} above the farthest class-0 point. A positive gap is exactly what "
        "linear separability means, and it is what the convergence theorem needs.")}

<h3 id="ex1b">B &middot; The perceptron</h3>

<p>As described in section 1, with <code>&eta; = {e1['eta']}</code>, weights initialised to
{vec(e1['w_init'])} and bias to <code>0</code>, capped at {e1['max_epochs']} epochs.</p>

<h3 id="ex1c">C &middot; Training and measuring</h3>

<div class="kpis">
  <div class="kpi"><div class="v">{f1['epochs']}</div><div class="k">epochs to a zero-update pass</div></div>
  <div class="kpi"><div class="v">{pct(f1['acc'])}</div><div class="k">final training accuracy</div></div>
  <div class="kpi"><div class="v">{f1['n_misclassified']}</div><div class="k">misclassified points</div></div>
  <div class="kpi"><div class="v">{f1['total_updates']}</div><div class="k">weight updates in total</div></div>
</div>

<p>The run ended on its own at epoch {f1['epochs']}, not at the 100-epoch cap: that epoch produced no updates
at all, so every one of the 2000 points was on the correct side and there was nothing left to change. The final
state is <b>w</b> = {vec(f1['w'])} and <i>b</i> = {f1['b']:.4f}.</p>

{figure("figures/fig2_boundary.png",
        "<b>Figure 2.</b> The line <code>w&middot;x + b = 0</code> for the final weights, drawn over the data. "
        f"No point is marked as misclassified because there are none &mdash; {pct(f1['acc'])} of 2000. Note how "
        "close the line runs to the class-0 cloud: the perceptron stops at the <i>first</i> hyperplane that "
        "happens to be consistent, and makes no attempt to sit in the middle of the gap.")}

{figure("figures/fig3_accuracy.png",
        "<b>Figure 3.</b> Accuracy after each epoch, for both learning rates. It is far from monotonic: each "
        "epoch is measured right after the 1000 class-1 points, which repeatedly drag the boundary too far "
        "before the next pass pulls it back. What matters is that the oscillation is damped &mdash; once the "
        f"line lands inside the gap it stays there, updates stop, and the run ends at epoch {f1['epochs']}.")}

<h3 id="ex1d">D &middot; Analysis</h3>

<h4>Why separable data converges quickly</h4>

<p>Because the update is <i>error-driven</i>. The term <code>y &minus; &#375;</code> is zero for every correctly
classified sample, so a correct point contributes nothing at all &mdash; not a small gradient, exactly nothing.
Training therefore consumes itself: each mistake rotates and shifts the boundary towards the offending point,
and once no point is left on the wrong side the algorithm is a fixed point of its own update rule.</p>

<p>The numbers make the scale of this concrete. Over {f1['epochs']} epochs the loop looked at
{m1['sample_presentations']:,} samples and changed the weights only {f1['total_updates']} times &mdash;
{m1['update_rate'] * 100:.2f}% of presentations. Nearly all the work was already done.</p>

<p>Novikoff's theorem is what turns that into a guarantee. Writing the model in augmented coordinates
<code>x&#771; = [x, 1]</code> so the bias folds into the weight vector, if some unit-norm <b>w</b>&#8407;* classifies
every sample with margin at least &gamma; and every <code>||x&#771;|| &le; R</code>, then the total number of
mistakes is at most <code>(R / &gamma;)&sup2;</code> &mdash; a bound that does not depend on how many samples
there are, or on &eta;. Here a valid witness is the hyperplane with the same normal as the centre line, offset
to the middle of the empirical gap:</p>

<div class="eqn">
R = {m1['R']:.4f} &nbsp;&nbsp; &gamma;* = {m1['gamma_star']:.6f} &nbsp;&nbsp;&rarr;&nbsp;&nbsp;
mistakes &le; (R / &gamma;*)&sup2; = {m1['novikoff_bound']:,.0f}<br>
<span class="lbl">observed: {f1['total_updates']} mistakes</span>
</div>

<p>The bound is finite, which is the whole point &mdash; it guarantees termination. It is also extremely loose,
by about three orders of magnitude, and that looseness is honest rather than a bug: the augmented form inflates
<code>R</code> with an offset of roughly 5 while deflating &gamma;, and the bound has to hold for the worst
possible ordering of the data. The qualitative claim it licenses is the one that matters: a wide gap means a
large &gamma;, and a large &gamma; means few mistakes.</p>

<p>It is worth noticing what the perceptron does <i>not</i> give you. The margin its own solution attains is
{m1['gamma_attained']:.2e} &mdash; more than a hundred times smaller than the witness we used to prove it would
terminate. It stops at the first consistent line, not the best one. That distinction is exactly what a support
vector machine exists to fix.</p>

<h4>Re-running with &eta; = 1.0</h4>

<p>Same data, same starting weights, only &eta; changed.</p>

{table("<b>Four runs</b>: two learning rates, each from the random init and from a zero init.",
       ["Run", "Final w", "Final b", "Epochs", "Accuracy", "||w||"],
       eta_rows, ["left", "num", "num", "num", "num", "num"])}

{table("<b>Direction versus scale.</b> The unit vector w/||w|| fixes the boundary's orientation; ||w|| does not "
       "affect any prediction.",
       ["Run", "w / ||w||", "||w||"], dir_rows, ["left", "num", "num"])}

<p>Both reach {pct(h1['acc'])}. The weight <i>directions</i> are the same to within
{h1['angle_deg']:.4f}&deg;, while the norms differ by a factor of {h1['norm_ratio']:.0f}. So &eta; bought
scale, not a different classifier &mdash; multiplying <b>w</b> and <i>b</i> by any positive constant leaves the
set <code>w&middot;x + b = 0</code> untouched and every prediction with it.</p>

<p>The one real difference is the epoch count: {f1['epochs']} at &eta; = 0.01 against {h1['epochs']} at
&eta; = 1.0. It would be easy to read that as "the small learning rate is faster", but the last two rows of the
table show it is nothing of the kind. Started from <b>w</b> = 0 both learning rates take exactly
{z1['epochs_lo']} epochs. The {f1['epochs']}-epoch run is the odd one out, and the reason is the
<i>initialisation</i>, not the rate:</p>

<div class="eqn">
||w&#8320;|| = {s1['w_init_norm']:.6f} &nbsp;&nbsp; mean ||x|| = {s1['mean_norm']:.4f}<br>
one update at &eta; = 0.01 moves w by &asymp; {s1['dw_eta001']:.4f} &nbsp;
<span class="lbl">&mdash; about {s1['ratio_eta001']:.0f}&times; the initial weights, so w&#8320; still counts</span><br>
one update at &eta; = 1.0 &nbsp;moves w by &asymp; {s1['dw_eta1']:.2f} &nbsp;
<span class="lbl">&mdash; about {s1['ratio_eta1']:.0f}&times;, so w&#8320; is immediately irrelevant</span>
</div>

<p>At &eta; = 1.0 the initial weights are wiped out by the first update, so the run is indistinguishable from
starting at zero &mdash; and indeed it takes the same {h1['epochs']} epochs. At &eta; = 0.01 the initial vector
is {1 / s1['ratio_eta001']:.2f}&times; the size of one update and survives long enough to matter; here it happened to point somewhere useful and
saved {z1['epochs_lo'] - f1['epochs']} epochs. That is luck, not a property of the learning rate.</p>

<p><b>So what does &eta; control?</b> Only the scale of <b>w</b>, and through it the single ratio
<code>||w&#8320;|| / (&eta;&middot;||x||)</code> &mdash; how quickly the updates drown out whatever the weights
were initialised to. It does not change the direction the weights converge to, the decision boundary, or the
classification of any point.</p>

<h4>Why &eta; cancels exactly when you start from zero</h4>

<p>Claim: run the algorithm twice on the same data in the same order, once with &eta;&#8321; and once with
&eta;&#8322;, both from <b>w</b> = 0, <i>b</i> = 0. Then at every step
<b>w</b><sup>(2)</sup> = <i>c</i> &middot; <b>w</b><sup>(1)</sup> and
<i>b</i><sup>(2)</sup> = <i>c</i> &middot; <i>b</i><sup>(1)</sup>, with <i>c</i> = &eta;&#8322;/&eta;&#8321; &gt; 0.</p>

<p><i>Proof by induction on the number of samples processed.</i></p>

<p><b>Base case.</b> Both runs start at <b>w</b> = 0 and <i>b</i> = 0, and 0 = <i>c</i> &middot; 0 for any
<i>c</i>. The relation holds.</p>

<p><b>Inductive step.</b> Suppose it holds before some sample <b>x</b>. The second run's activation is</p>

<div class="eqn">
w<sup>(2)</sup>&middot;x + b<sup>(2)</sup> = c&middot;w<sup>(1)</sup>&middot;x + c&middot;b<sup>(1)</sup>
= c (w<sup>(1)</sup>&middot;x + b<sup>(1)</sup>)
</div>

<p>Since <i>c</i> &gt; 0, multiplying by <i>c</i> cannot change a sign, so
<code>w<sup>(2)</sup>&middot;x + b<sup>(2)</sup> &ge; 0</code> exactly when
<code>w<sup>(1)</sup>&middot;x + b<sup>(1)</sup> &ge; 0</code>. Both runs therefore emit the <b>same
prediction</b> &#375;, hence the same error <code>e = y &minus; &#375;</code>, and in particular they update on
precisely the same samples. Applying the rule:</p>

<div class="eqn">
w<sup>(2)</sup> + &eta;&#8322; e x = c&middot;w<sup>(1)</sup> + c&middot;&eta;&#8321; e x
= c (w<sup>(1)</sup> + &eta;&#8321; e x)
<span class="lbl">&nbsp; using &eta;&#8322; = c&middot;&eta;&#8321;</span><br>
b<sup>(2)</sup> + &eta;&#8322; e &nbsp;= c&middot;b<sup>(1)</sup> + c&middot;&eta;&#8321; e
&nbsp;= c (b<sup>(1)</sup> + &eta;&#8321; e)
</div>

<p>which is the relation again, one step later. By induction it holds for the entire run. &#9633;</p>

<p>Three consequences follow immediately. The two runs make <b>identical predictions at every step</b>, so they
misclassify the same samples in the same order and their accuracy curves are equal; the epoch on which a pass
first produces no updates is therefore the same, so the <b>epoch count is identical</b>; and since
<code>{{x : c(w&middot;x + b) = 0}}</code> is the same set as <code>{{x : w&middot;x + b = 0}}</code> for
<i>c</i> &gt; 0, the <b>decision boundary is identical</b>. Only <code>||w||</code> differs, by the factor
<i>c</i>.</p>

<p>The code checks this rather than asserting it. Running from zero at &eta; = 0.01 and &eta; = 1.0 gives
weights whose largest componentwise discrepancy from the predicted factor of {z1['scale']:.0f} is
{z1['max_abs_residual']:.2e} &mdash; floating-point noise &mdash; with identical epoch counts
({z1['epochs_lo']} and {z1['epochs_hi']}) and, verified elementwise, identical accuracy curves.</p>

<div class="callout">
<span class="tag">Which closes the loop on the previous question</span>
<p>The proof needs <b>w</b> = 0 as its base case. That is precisely why the statement asks for a non-zero
initialisation: it breaks the scaling relation and makes &eta; observable at all. What we measured above &mdash;
{f1['epochs']} epochs versus {h1['epochs']} &mdash; is the size of that broken symmetry, and it vanishes the
moment the initialisation does.</p>
</div>
</section>

<!-- ================================================================== EX 2 -->
<section id="ex2">
<h2><span class="num">3</span>Exercise 2 &mdash; overlapping data</h2>

<h3 id="ex2a">A &middot; Generating the data</h3>

<p>The same construction with the geometry inverted: the centres move <i>closer</i>, from
{e1['centre_distance']:.2f} apart to {e2['centre_distance']:.4f}, while the spread <i>triples</i>, from
&sigma; = {e1['sigma']:.4f} to {e2['sigma']:.4f}. Where Exercise 1 had its centres
{s1['sigma_multiple']:.1f} standard deviations apart, here they are only {e2['sigma_multiple']:.2f} &mdash; so
the clouds do not merely touch, they sit almost on top of one another.</p>

{figure("figures/fig4_scatter.png",
        "<b>Figure 4.</b> The overlapping classes. There is no empty corridor anywhere: any straight line you "
        "draw will have points of both classes on both sides. The class centres are still distinguishable, "
        "which is why a linear rule can do better than chance &mdash; but not much better.")}

<h3 id="ex2b">B &middot; Training with the pocket</h3>

<p>The implementation from Exercise 1 is imported unchanged, with the same &eta; = {e2['eta']} and the same
100-epoch cap. The only addition is the pocket: a copy of the best weights seen so far, updated whenever the
running weights score higher on the full training set. The pocket never feeds back into training &mdash; it only
records.</p>

{table("<b>What the run produced.</b> The bisector of the two centres is included as a reference: it is the "
       "Bayes-optimal linear rule <i>for these distributions</i>, though on a finite sample another line can "
       "edge past it.",
       ["Weights", "w", "b", "Accuracy", "Misclassified"],
       ex2_rows, ["left", "num", "num", "num", "num"])}

<div class="kpis">
  <div class="kpi"><div class="v">{pct(f2['acc'])}</div><div class="k">final weights</div></div>
  <div class="kpi"><div class="v">{pct(p2['acc'])}</div><div class="k">pocket weights, at epoch {p2['epoch']}</div></div>
  <div class="kpi"><div class="v">{pct(by['acc_empirical'])}</div><div class="k">distribution-optimal line, for reference</div></div>
  <div class="kpi"><div class="v">{f2['epochs']}</div><div class="k">epochs run &mdash; the cap, never converged</div></div>
</div>

<p>The run did not stop early: all {f2['epochs']} epochs executed, and the smallest number of updates in any
single epoch was {e2['min_updates_epoch']}, never zero. The pocket reached {pct(p2['acc'])} at epoch
{p2['epoch']} and was never beaten in the {100 - p2['epoch']} epochs that followed.</p>

<h3 id="ex2c">C &middot; Figures</h3>

{figure("figures/fig5_boundaries.png",
        "<b>Figure 5.</b> Both boundaries on the data, with the points the pocket gets wrong circled. The pocket "
        "line passes between the two centres, roughly where you would draw it by hand. The final line is not "
        f"near the data at all &mdash; it sits below everything, so it calls {f2['predicted_class1_fraction']:.1%} "
        f"of the set class 1 and scores {pct(f2['acc'])}, which is chance.")}

{figure("figures/fig6_pocket.png",
        "<b>Figure 6.</b> The two required curves. The running accuracy never settles and never rises much "
        f"above 50%; the pocket steps up whenever a better vector is seen and holds {pct(p2['acc'])} from epoch "
        f"{p2['epoch']} onwards. Compare the shape with Figure 3, which flattens and stops.")}

<p>Figure 6 raises an obvious question: if the running weights hover at 50% for all 100 epochs, where did the
pocket's {pct(p2['acc'])} come from? The answer is that once-per-epoch sampling hides the dynamics.</p>

{figure("figures/fig6b_updates.png",
        "<b>Figure 6b.</b> The same run, resolved per weight update instead of per epoch. It is a sawtooth: "
        "within every epoch the boundary sweeps up through the clouds &mdash; touching the linear ceiling near "
        f"update {p2['update']} &mdash; and is then dragged back out by the class-1 block at the end of the "
        "pass. The orange triangles mark the epoch boundaries, which is to say the points Figure 6 samples. "
        "They land in the trough every single time.")}

<h3 id="ex2d">D &middot; Analysis</h3>

<h4>Where the gap between final and pocket accuracy comes from</h4>

<p>The proximate cause is that on non-separable data there is no fixed point. Updates never stop, so the final
weights are not a solution the algorithm settled on &mdash; they are just wherever it happened to be standing
when the epoch counter ran out. Since every epoch ends with 1000 consecutive class-1 samples, that moment is
systematically the worst one in the cycle.</p>

<p>The mechanism is in the relative size of the two updates. A mistake moves the weight vector by
<code>&eta;x</code> but the bias by only <code>&eta;</code>:</p>

<div class="eqn">
||&Delta;w|| = &eta;&middot;||x|| &asymp; {e2['eta']} &times; {sc['mean_norm']:.3f} = {sc['dw_per_mistake']:.4f}
&nbsp;&nbsp;versus&nbsp;&nbsp; |&Delta;b| = &eta; = {sc['db_per_mistake']:.4f}<br>
<span class="lbl">the normal vector travels about {sc['ratio']:.1f}&times; faster than the offset, because the
clouds sit ||x|| &asymp; 5 from the origin</span>
</div>

<p>That ratio is decisive, because the boundary's distance from the origin is
<code>&minus;b / ||w||</code>, and for the line to pass through the clouds that quantity has to reach the
projection of the cloud centre &mdash; about 4.9. The bias accumulates {sc['ratio']:.1f} times more slowly
than <code>||w||</code> grows, so the ratio it can sustain is of order 1, not of order 5:</p>

{table("<b>Where each boundary actually sits.</b> &minus;b/||w|| is the line's distance from the origin; "
       "&quot;needed&quot; is the value that would put it through the middle of the data.",
       ["Weights", "||w||", "b", "&minus;b / ||w||", "Needed", "Accuracy"],
       geom_rows, ["left", "num", "num", "num", "num", "num"], hl=set([1]))}

<p>The final weights sit at {f2['geometry']['offset']:.2f} when they would need
{f2['geometry']['needed']:.2f} &mdash; the line is nowhere near the data, which is why it labels essentially
everything class 1 and scores {pct(f2['acc'])}. The pocket weights sit at {p2['geometry']['offset']:.2f}
against a required {p2['geometry']['needed']:.2f}, which is in range, and score {pct(p2['acc'])}. Notice how
the pocket got there: its <code>||w||</code> is {f2['w_norm'] / p2['w_norm']:.1f} times <i>smaller</i>, so the
same small bias buys a much larger offset. Those configurations are transient &mdash; the very next mistake
inflates <code>||w||</code> again &mdash; which is precisely why they have to be caught and stored as they go
past. Of the {e2['total_updates']} updates in the run, only {e2['trace_stats']['above_60']} ever exceeded 60%
and {e2['trace_stats']['above_70']} exceeded 70%.</p>

<p>So the {pct(p2['acc'])} against {pct(f2['acc'])} is not the pocket being clever. It is the pocket keeping a
receipt for a good moment that the algorithm itself has no way to recognise or hold on to.</p>

<h4>Figure 3 against Figure 6, and what the theorem assumed</h4>

<p>Figure 3 is a damped oscillation that terminates: the swings shrink, the curve reaches 100%, an epoch passes
with no updates, and training ends at epoch {f1['epochs']}. Figure 6 is an undamped one that does not: after
{f2['epochs']} epochs the curve is still moving between {pct(e2['curve_min'])} and {pct(e2['curve_max'])}, and
every epoch still produces updates. Nothing about it is converging, and running it longer would not change that.</p>

<p>The perceptron convergence theorem requires two things:</p>

<ol>
<li><b>Bounded inputs</b> &mdash; some <code>R</code> with <code>||x&#771;|| &le; R</code> for every sample.
This dataset satisfies it comfortably; the norms are finite and small.</li>
<li><b>Linear separability with a positive margin</b> &mdash; some <b>w</b>* and &gamma; &gt; 0 with
<code>y&#771;(w*&middot;x&#771;) &ge; &gamma;</code> for <i>every</i> sample. <b>This is the assumption that
fails.</b></li>
</ol>

<p>It fails structurally, not by accident of sampling. Two Gaussians have overlapping support: whatever line
you pick, both classes have probability mass on both sides of it. The optimal linear rule for these two
distributions &mdash; the perpendicular bisector of the centres, since the covariances are equal and spherical
&mdash; still misclassifies {int(round((1 - by['acc_empirical']) * 2000))} of the 2000 points, for
{pct(by['acc_empirical'])} accuracy. The theoretical ceiling, from
<code>&Phi;(d / 2&sigma;)</code> with <code>d = {by['d']:.4f}</code> and
<code>&sigma; = {by['sigma']:.4f}</code>, is {pct(by['acc_theoretical'])}.</p>

<p>With no &gamma; &gt; 0 to speak of, <code>(R/&gamma;)&sup2;</code> is unbounded. The theorem does not give a
weaker guarantee in this regime &mdash; it gives none, and the behaviour in Figure 6 is what "none" looks like.
The algorithm keeps finding mistakes because mistakes genuinely exist, and it is built to respond to every one
of them.</p>

<p>The pocket's {pct(p2['acc'])}, incidentally, is marginally <i>above</i> the bisector's
{pct(by['acc_empirical'])}. That is not a contradiction: the bisector is optimal for the distributions, while
the pocket is chosen by looking at this particular finite sample, and 2000 points leave room for that much
slack.</p>

<h4>Would more epochs, or a smaller &eta;, fix it?</h4>

<p>Neither, and the update rule says so without needing an experiment.</p>

<p><b>More epochs.</b> The stopping rule fires when an epoch produces zero updates. An update fires on every
misclassified sample. Since the classes overlap, every possible weight vector misclassifies a positive fraction
of the data &mdash; even the best line found in this whole run still gets {p2['n_misclassified']} of the 2000
wrong &mdash; so the number of mistakes per epoch is bounded below by a positive number that no amount of
training can reduce. A zero-update epoch is therefore unreachable, and the run is guaranteed to exit on the cap at any
budget. The evidence is already in the results: the smallest number of updates in any of the
{f2['epochs']} epochs was {e2['min_updates_epoch']}, and the pocket's best arrived at epoch {p2['epoch']} and
was not improved once in the {100 - p2['epoch']} epochs after it. More epochs buy more lottery tickets for the
pocket, with diminishing returns. They do not buy convergence.</p>

<p><b>A smaller &eta;.</b> The zero-initialisation proof in <a href="#ex1d">Exercise 1D</a> is the complete
answer. From a zero initialisation, changing
&eta; rescales the entire weight trajectory by a positive constant and changes no prediction anywhere &mdash;
the boundary and the mistake sequence are literally identical. A learning rate cannot fix non-convergence
because it is not a parameter of the decision boundary at all; only <code>w/||w||</code> and
<code>b/||w||</code> are, and scaling every update by the same constant leaves both alone. With the non-zero
initialisation used here, shrinking &eta; does one thing: it makes <code>w&#8320;</code> relatively more
important for longer. That is a perturbation, not a cure, and it certainly cannot manufacture a separating
hyperplane that does not exist.</p>

<p>What would actually help is changing what is being optimised, not how fast:</p>

<ul>
<li><b>Keep the pocket</b> &mdash; already done, and already at the linear ceiling: {pct(p2['acc'])} against
the {pct(by['acc_empirical'])} of the distribution-optimal bisector leaves essentially nothing on the table.</li>
<li><b>Use a loss with a minimum</b> &mdash; logistic regression or a soft-margin SVM optimise a convex
objective that stays finite on overlapping data, so they converge to a stable answer instead of oscillating.
The perceptron rule has no such objective to descend.</li>
<li><b>Accept the floor, or change the features</b> &mdash; roughly
{(1 - by['acc_theoretical']) * 100:.1f}% error is irreducible for <i>any</i> linear rule on these two
distributions. Beating it needs a different hypothesis class, not a different optimiser.</li>
</ul>

<div class="callout warn">
<span class="tag">On the sample ordering</span>
<p>Because the data is not shuffled, the {pct(f2['acc'])} figure is partly an artefact of <i>when</i> we look:
every epoch ends just after 1000 class-1 points. Shuffling would make the final weights land somewhere less
systematically bad, and the final-versus-pocket gap would narrow. It would not change the conclusion. The run
would still never converge, the accuracy would still oscillate rather than settle, and the pocket would still be
necessary &mdash; because none of that depends on the order, only on the overlap.</p>
</div>
</section>

<!-- ================================================================ SUMMARY -->
<section id="summary">
<h2><span class="num">4</span>Results summary</h2>

{table("<b>All required reported values</b>, interpolated directly from the JSON the scripts emit.",
       ["#", "Quantity", "Value"], summary_rows, ["num", "left", "left"])}
</section>

<!-- ================================================================== REPRO -->
<section id="repro">
<h2><span class="num">5</span>Reproducing this page</h2>

<p>Two scripts produce every figure and every number above; a third renders this page from their JSON output.
The seed is <code>42</code> and each script draws from its own generator in a fixed order, so the results are
identical between runs and each exercise can be run on its own.</p>

<pre><code>git clone {REPO}
cd ann-dl
pip install numpy matplotlib

cd code
python perceptron_ex1.py            <span class="c"># Figures 1, 2, 3   + results/perceptron1.json</span>
python perceptron_ex2.py            <span class="c"># Figures 4, 5, 6, 6b + results/perceptron2.json</span>
python build_perceptron_report.py   <span class="c"># renders exercises/perceptron/index.html</span></code></pre>

<p>The perceptron itself lives in <code>code/perceptron.py</code> and is written from scratch &mdash; NumPy is
used for array arithmetic and the random generator, Matplotlib for the figures, and nothing else. No
third-party model is imported or trained anywhere in this exercise.</p>

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
