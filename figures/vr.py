"""Vietoris-Rips helpers for the topology schematics.

The complexes drawn in figures 14, 18, 19, 20 are computed from real point
coordinates at real thresholds, so the topology on the slide is the topology
the rule actually produces. Betti numbers come from ripser, not from the
author's expectation.
"""
from itertools import combinations

import numpy as np


def pdist(P):
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return D


def edges(P, t):
    D = pdist(P)
    return [(i, j) for i, j in combinations(range(len(P)), 2) if D[i, j] <= t]


def triangles(P, t):
    """Clique rule: a triple is filled iff all three pairs are within t."""
    D = pdist(P)
    return [(i, j, k) for i, j, k in combinations(range(len(P)), 3)
            if D[i, j] <= t and D[i, k] <= t and D[j, k] <= t]


def betti(P, t, maxdim=1):
    """(β₀, β₁) of the VR complex at threshold t, via ripser."""
    from ripser import ripser
    dgms = ripser(pdist(P), distance_matrix=True, maxdim=maxdim,
                  thresh=float(t))["dgms"]
    b = []
    for d in dgms:
        if d.size == 0:
            b.append(0)
            continue
        alive = (d[:, 0] <= t) & (d[:, 1] > t)
        b.append(int(alive.sum()))
    return tuple(b)


def draw(ax, P, t, style, discs=True, pt_ms=6, tri_alpha=0.28, lw=1.5):
    """Draw the VR complex at threshold t: discs of radius t/2, edges, and
    filled 2-simplices."""
    from matplotlib.patches import Circle, Polygon
    if discs:
        for p in P:
            ax.add_patch(Circle(p, t / 2, fc=style["ICE"], ec="none",
                                alpha=0.45, zorder=1))
    for (i, j, k) in triangles(P, t):
        ax.add_patch(Polygon(P[[i, j, k]], closed=True, fc=style["NAVY"],
                             ec="none", alpha=tri_alpha, zorder=2))
    for (i, j) in edges(P, t):
        ax.plot(P[[i, j], 0], P[[i, j], 1], color=style["NAVY"], lw=lw,
                zorder=3)
    ax.plot(P[:, 0], P[:, 1], "o", color=style["INK"], ms=pt_ms, zorder=4)


def ring(n=12, r=1.0, jitter=0.06, seed=5):
    """A noisy ring — the standard example of a point cloud with one loop."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.c_[r * np.cos(th), r * np.sin(th)]
    return P + rng.normal(0, jitter, P.shape)
