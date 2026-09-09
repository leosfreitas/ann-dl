"""Shared plotting style and palette for the ann-dl data exercise.

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
