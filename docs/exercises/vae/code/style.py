"""Shared plotting style and palette for the ann-dl exercises.

Palette slots come from the validated categorical theme (blue / orange / aqua /
violet).  That 4-slot subset was checked with the palette validator under
``--pairs all`` (scatter uses every pair, not just adjacent ones) on the light
surface: CVD worst pair dE 9.2, normal-vision worst pair dE 16.3.  Aqua sits at
2.74:1 contrast, just under the 3:1 gate, so every figure that uses it also
carries direct labels or an accompanying table -- the documented relief rule.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#86857f"
GRID = "#e3e2dd"

# Categorical slots, fixed order -- never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

# Marker shapes double as a secondary (non-colour) identity encoding.
MARKERS = ["o", "s", "^", "D"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


def title(ax, text, subtitle=None):
    """Left-aligned title, with an optional recessive subtitle underneath."""
    ax.set_title(text, loc="left", color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=INK_MUTED, ha="left", va="bottom")


# ---------------------------------------------------------------------------
# Ten-class scheme, for the digit-coloured latent scatter (Figure 4).
#
# Ten series is past what colour alone can carry: on the validated ramps no
# ten-hue ordering clears the all-pairs gates a scatter is measured against
# (every pair is on screen at once, not just adjacent ones), and inventing two
# extra hues is not allowed.  So identity is encoded twice over.
#
# Five hues were chosen by running the palette validator over all 56 five-hue
# subsets of the 8-slot categorical theme under ``--pairs all`` on the light
# surface and keeping the best: blue / yellow / magenta / green / violet, which
# passes every hard gate (worst all-pairs CVD dE 13.0, normal-vision dE 16.3).
# Each hue then carries two digits, separated by marker shape -- so a same-hue
# pair is told apart by shape and a different-hue pair by hue.  Yellow (2.11:1)
# and magenta (2.62:1) sit under the 3:1 contrast gate, so the documented relief
# rule applies: every class is also labelled directly at its centroid.
DIGIT_HUES = ["#2a78d6", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
DIGIT_MARKERS = ["o", "^"]


def _perfect_matchings(items):
    """Every way of splitting an even-length list into unordered pairs."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for i in range(len(rest)):
        for tail in _perfect_matchings(rest[:i] + rest[i + 1:]):
            yield [(first, rest[i])] + tail


def pair_digits_by_distance(centroids):
    """Hand the five hues to the ten digits so that hue-mates sit far apart.

    Which two digits share a hue is not cosmetic.  The naive pairing (d with
    d + 5) puts 4 with 9 and 3 with 8 -- precisely the pairs that land on top
    of each other in a MNIST latent space, so the shape encoding buys nothing
    exactly where it is needed and the two classes become indistinguishable.

    So all 945 ways of splitting ten digits into five pairs are enumerated and
    the one whose *closest* pair of centroids is as far apart as possible wins
    (a max-min matching).  Overlapping classes then always differ in hue, and
    two digits only share a hue when they are far enough apart that position
    already separates them.  It is a deterministic function of the trained
    model, so the figure is reproducible.

    Returns ``({digit: (colour, marker)}, pairs, min_separation)``.
    """
    def dist(a, b):
        dx = centroids[a][0] - centroids[b][0]
        dy = centroids[a][1] - centroids[b][1]
        return (dx * dx + dy * dy) ** 0.5

    best, best_score = None, -1.0
    for matching in _perfect_matchings(list(range(10))):
        score = min(dist(a, b) for a, b in matching)
        if score > best_score:
            best_score, best = score, matching

    assign = {}
    # Sorted by the lower digit of each pair, so the hue order follows the
    # legend's reading order rather than the order the search happened to find.
    pairs = sorted(best)
    for slot, (a, b) in enumerate(pairs):
        assign[a] = (DIGIT_HUES[slot], DIGIT_MARKERS[0])
        assign[b] = (DIGIT_HUES[slot], DIGIT_MARKERS[1])
    # Plain Python types: centroids usually arrive as a NumPy array, and a
    # stray float32 here is not JSON-serialisable downstream.
    return assign, [(int(a), int(b)) for a, b in pairs], float(best_score)


def image_ax(ax):
    """Strip ticks, grid and spines: for a panel that shows a raster image."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
