"""Shared style for every ER=EPR figure. No colour is defined outside this module.

Usage in a figure script:

    from style import *
    fig, ax = new_fig("half")
    ...
    finish(fig, FIG_ID)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- palette (the deck's) ------------------------------------------------
INK   = "#161F38"   # primary text / axes
MUTED = "#5A6478"   # secondary text, tick labels
NAVY  = "#1B2A4A"   # primary data series
AMBER = "#E9A13B"   # accent / "our result"
TEAL  = "#3E8E7E"   # positive / pass
CLAY  = "#C05746"   # negative / fail
ICE   = "#CBD9F0"   # light fills
LIFT  = "#24406B"   # secondary dark
GRID  = "#E3E8F0"   # gridlines
CARD  = "#F1F4FA"   # panel fills

SIZES = {"half": (5.6, 4.2), "full": (11.5, 4.6), "square": (5.0, 5.0)}

HERE = Path(__file__).resolve().parent
OUT = HERE
PROV = HERE / "provenance.txt"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "text.usetex": False,
    "axes.labelsize": 11, "axes.labelcolor": INK,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.fontsize": 10, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "figure.dpi": 110, "savefig.dpi": 300,
    "axes.titlesize": 11,          # titles are never set; guard only
})


def new_fig(kind="half", ncols=1, nrows=1, **kw):
    """Figure + axes with the deck's geometry. kind in {half, full, square}."""
    figsize = kw.pop("figsize", SIZES[kind])
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize,
                           constrained_layout=True, **kw)
    for a in np.atleast_1d(np.asarray(ax, dtype=object)).ravel():
        style_axes(a)
    return fig, ax


def style_axes(ax, grid_axis="y"):
    """Spines left+bottom only; gridlines beneath the data on one axis."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return ax


def blank_axes(ax):
    """A drawing canvas: no spines, no ticks, no grid (for schematics)."""
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_aspect("equal")
    return ax


def finish(fig, fig_id, name=None):
    """Save PDF (white bg, vector) and PNG (transparent, 300 dpi)."""
    stem = f"fig_{fig_id:02d}_{name}" if name else f"fig_{fig_id:02d}"
    pdf, png = OUT / f"{stem}.pdf", OUT / f"{stem}.png"
    fig.savefig(pdf, facecolor="white", edgecolor="none")
    fig.savefig(png, transparent=True, dpi=300)
    plt.close(fig)
    return stem


def provenance(fig_id, text):
    """Print and record where a [DATA] figure's numbers came from."""
    line = f"fig_{fig_id:02d}: {text}"
    print(line)
    with open(PROV, "a") as fh:
        fh.write(line + "\n")
    return line


# --- schematic primitives ------------------------------------------------
def box(ax, xy, w, h, label, fc=CARD, ec=NAVY, tc=INK, fs=10, lw=1.4, r=0.02,
        weight="normal", zorder=2):
    """Rounded box with centred wrapped label. xy is the lower-left corner."""
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((xy[0], xy[1]), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(p)
    if label:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center",
                fontsize=fs, color=tc, zorder=zorder + 1, weight=weight)
    return p


def arrow(ax, p0, p1, color=MUTED, lw=1.6, style="-|>", ls="-", rad=0.0,
          zorder=3, mutation=14):
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                        linestyle=ls, mutation_scale=mutation, zorder=zorder,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    return a


def note(ax, xy, text, color=MUTED, fs=10, ha="center", va="center",
         weight="normal", zorder=5, style="normal"):
    return ax.text(xy[0], xy[1], text, ha=ha, va=va, fontsize=fs, color=color,
                   zorder=zorder, weight=weight, style=style)


def circle(ax, xy, r, fc=ICE, ec=NAVY, lw=1.5, alpha=1.0, zorder=2):
    from matplotlib.patches import Circle
    c = Circle(xy, r, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(c)
    return c


def wavy(ax, p0, p1, amp=0.05, n=220, color=AMBER, lw=2.0, cycles=6, zorder=3):
    """A wavy connector (entanglement link)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n)
    d = p1 - p0
    L = np.hypot(*d)
    perp = np.array([-d[1], d[0]]) / L
    pts = p0[None, :] + t[:, None] * d[None, :] \
        + (amp * np.sin(2 * np.pi * cycles * t))[:, None] * perp[None, :]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=zorder,
            solid_capstyle="round")
    return pts


def bar_h(ax, y, x0, x1, color=NAVY, lw=3.2, zorder=3, alpha=1.0):
    """One persistence bar."""
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt",
            zorder=zorder, alpha=alpha)


def legend_below(ax, ncol=3, y=-0.16, **kw):
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     **kw)
