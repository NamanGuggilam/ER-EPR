#!/usr/bin/env python3
"""Rung 1 of the RTN rebuild: persistent homology of a random tensor network.

MODEL
-----
A 1D open chain of n_sites = 6 boundary sites.  Each site carries one tensor
with a dangling (physical) boundary leg; neighbouring tensors are contracted
through a bond of dimension D.  The bulk geometry IS the bond graph:

    site:    1 --- 2 --- 3 --- 4 --- 5 --- 6
    bond:       b1    b2    b3    b4    b5

WHY AN RTN
----------
For a random tensor network the Ryu-Takayanagi formula is a theorem
(Hayden-Nezami-Qi-Thomas-Walter-Yang 2016), not a conjecture: averaged over
the Haar-random tensors, the entanglement entropy of a boundary region is the
minimal cut through the bond graph,

    S(A) = min_cut(A) * log(D)                            [RT, exact for RTNs]

so the entanglement (EPR) side and the geometric (ER) side can be computed
INDEPENDENTLY and then compared, with no numerical diagonalization and no
appeal to a conjectured duality.

WHAT THIS SCRIPT TESTS
----------------------
The claim under test is that a single linear rescaling phi carries the
boundary distance matrix onto the bulk one,

    phi * D_boundary  ==  D_bulk        with the prediction phi = log(D),

which would make the two persistence barcodes isomorphic.  Nothing here is
tuned toward that answer: both matrices are built from their own definitions
and the rescaling is then MEASURED pair by pair and reported, matching or not.

The script is a report, not a search: no threshold, factor, or bond dimension
is adjusted to improve agreement.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # headless: write PNGs, open no window
import matplotlib.pyplot as plt
import numpy as np
from persim import wasserstein
from ripser import ripser

HERE = Path(__file__).resolve().parent

#: Bond dimensions for the sweep.
DEFAULT_BOND_DIMS = [2, 4, 8, 16, 32]

#: Sites are labelled 1..n_sites throughout (physics convention); numpy
#: indices are therefore site-1.  Every helper takes 1-based site labels.


# ==========================================================================
# 1. ENTANGLEMENT (EPR) SIDE: RT entropy from the minimal cut
# ==========================================================================

def min_cut(i, j, n_sites):
    """Bonds that must be cut to separate the contiguous interval [i..j].

    On an OPEN chain the interval's complement sits on at most two sides, so
    the cut is just its two endpoints' outward bonds:

      * interval strictly inside the chain (1 < i and j < n)  -> 2 bonds
      * interval touching exactly one end of the chain        -> 1 bond
      * the whole chain [1..n]                                -> 0 bonds

    The last case is not an edge case to be papered over: the global state of
    an RTN is pure, so S = 0 is the correct physical answer for A = everything.
    """
    if not 1 <= i <= j <= n_sites:
        raise ValueError(f"need 1 <= i <= j <= n_sites, got i={i}, j={j}")
    return (1 if i > 1 else 0) + (1 if j < n_sites else 0)


def rt_entropy(i, j, n_sites, bond_dim):
    """RT entropy of the interval [i..j]: S = min_cut * log(D).

    Exact for a random tensor network -- this is the theorem, not a model.
    """
    return min_cut(i, j, n_sites) * np.log(bond_dim)


# ==========================================================================
# 2. BOUNDARY DISTANCE MATRIX
# ==========================================================================

def boundary_distance_matrix(n_sites, bond_dim):
    """d_boundary(i,j) = 1 / S({i..j}), with d(i,i) = 0.

    Returns (D, n_singular) where n_singular counts the pairs whose interval
    has S = 0 and therefore 1/S = +inf.  Those entries are left as np.inf
    rather than clipped: in a Vietoris-Rips filtration an infinite distance is
    meaningful (the edge is never born), so the honest encoding of "these two
    sites are never linked" is +inf, and substituting a large finite number
    would invent a birth scale that the construction does not contain.
    """
    D = np.zeros((n_sites, n_sites), dtype=float)
    n_singular = 0
    for i in range(1, n_sites + 1):
        for j in range(1, n_sites + 1):
            if i == j:
                continue                               # diagonal stays 0
            lo, hi = min(i, j), max(i, j)
            S = rt_entropy(lo, hi, n_sites, bond_dim)
            if S == 0.0:
                D[i - 1, j - 1] = np.inf
                n_singular += 1
            else:
                D[i - 1, j - 1] = 1.0 / S
    return D, n_singular


# ==========================================================================
# 3. BULK (ER) SIDE: geodesic distance in the bond graph
# ==========================================================================

def bond_position(k):
    """Position of bond b_k = (k, k+1) along the chain: midway between sites."""
    return k + 0.5


def bulk_distance_matrix(n_sites):
    """d_bulk(i,j) = |position(b_i) - position(b_j)| = |i - j|.

    This is the formula the construction specifies operationally, and on a
    chain it is just the graph geodesic between the two sites' outward bonds.

    NOTE -- an inconsistency in the two definitions offered for this side.
    "The shortest path between the two minimal cut points" is undefined for an
    interval that touches an end of the chain, because that interval's minimal
    cut has only ONE bond, not two (see min_cut above): there is no second cut
    point to measure to.  Of the 15 unordered pairs on a 6-site chain, 9 touch
    an end, so the cut-point reading leaves most of the matrix undefined while
    the bond-position reading fills it in.  The bond-position formula is used
    here and the discrepancy is reported rather than silently resolved.
    """
    D = np.zeros((n_sites, n_sites), dtype=float)
    for i in range(1, n_sites + 1):
        for j in range(1, n_sites + 1):
            if i == j:
                continue
            D[i - 1, j - 1] = abs(bond_position(i) - bond_position(j))
    return D


def n_cut_points(i, j, n_sites):
    """How many cut points the interval [i..j] actually has (0, 1, or 2)."""
    return min_cut(min(i, j), max(i, j), n_sites)


# ==========================================================================
# 4. METRIC VALIDATION
# ==========================================================================

def validate_metric(D, name, tol=1e-12):
    """Check the distance-matrix axioms and report every violation found.

    Returns a list of human-readable strings (empty list = a valid metric on
    the finite entries).  Triangle inequalities involving an infinite entry
    are skipped: they hold vacuously.
    """
    problems = []
    n = D.shape[0]
    if not np.allclose(np.diag(D), 0, atol=tol):
        problems.append("diagonal is not zero")
    if not np.allclose(D, D.T, atol=tol, equal_nan=True):
        problems.append("matrix is not symmetric")

    off = ~np.eye(n, dtype=bool)
    finite_off = D[off][np.isfinite(D[off])]
    if np.any(finite_off < -tol):
        problems.append(f"negative off-diagonal entries: min {finite_off.min():.6g}")
    n_zero = int(np.sum(np.abs(finite_off) <= tol))
    if n_zero:
        problems.append(f"{n_zero} zero off-diagonal entries "
                        f"(distinct sites at distance 0 -> not a metric, "
                        f"only a pseudmetric)")
    n_inf = int(np.sum(~np.isfinite(D[off])))
    if n_inf:
        problems.append(f"{n_inf} infinite off-diagonal entries "
                        f"(pairs never joined in the filtration)")

    n_tri = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i in (j, k) or j == k:
                    continue
                a, b, c = D[i, j], D[i, k], D[k, j]
                if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
                    continue
                if a > b + c + tol:
                    n_tri += 1
    if n_tri:
        problems.append(f"{n_tri} triangle-inequality violations")

    # How much distance information does the matrix actually carry?
    distinct = sorted({round(float(v), 10) for v in finite_off})
    problems.append(f"INFO: {len(distinct)} distinct finite off-diagonal "
                    f"value(s): {[f'{v:.6g}' for v in distinct]}")
    return problems


# ==========================================================================
# 5. PERSISTENT HOMOLOGY
# ==========================================================================

def barcodes(D, maxdim=1):
    """Ripser on a precomputed distance matrix.  Returns the diagram list."""
    return ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"]


def finite_bars(dgm):
    """Drop infinite bars -- persim's Wasserstein needs finite diagrams."""
    if dgm is None or len(dgm) == 0:
        return np.empty((0, 2))
    dgm = np.asarray(dgm, dtype=float)
    return dgm[np.isfinite(dgm[:, 1])]


def describe_barcode(dgm, label, indent="    "):
    """Print one dimension's bars, longest first."""
    dgm = np.asarray(dgm, dtype=float) if len(dgm) else np.empty((0, 2))
    if len(dgm) == 0:
        print(f"{indent}{label}: (no bars)")
        return
    n_inf = int(np.sum(~np.isfinite(dgm[:, 1])))
    fin = finite_bars(dgm)
    print(f"{indent}{label}: {len(dgm)} bars ({n_inf} infinite)")
    order = np.argsort(-(fin[:, 1] - fin[:, 0])) if len(fin) else []
    for idx in order:
        b, d = fin[idx]
        print(f"{indent}  [{b:.6f}, {d:.6f}]   persistence {d - b:.6f}")


def wasserstein_safe(dgm_a, dgm_b):
    """Wasserstein distance between two diagrams, with the empty cases named.

    Returns (value, note).  Two empty diagrams are at distance 0 by
    definition; one empty is still well defined (everything matches to the
    diagonal), so both are computed rather than skipped.
    """
    a, b = finite_bars(dgm_a), finite_bars(dgm_b)
    if len(a) == 0 and len(b) == 0:
        return 0.0, "both diagrams empty (no finite bars) -> 0 by definition"
    try:
        return float(wasserstein(a, b)), ""
    except Exception as exc:                    # pragma: no cover - diagnostic
        return float("nan"), f"persim failed: {type(exc).__name__}: {exc}"


# ==========================================================================
# 6. THE MAP phi
# ==========================================================================

def measure_phi(D_boundary, D_bulk):
    """Measure the per-pair rescaling phi(i,j) = d_bulk / d_boundary.

    If a single linear rescaling carries one matrix onto the other, every
    finite pair must return the SAME phi.  The spread across pairs is the
    whole test, so it is reported rather than averaged away.

    Returns dict with the ratio matrix and its statistics.
    """
    n = D_boundary.shape[0]
    off = ~np.eye(n, dtype=bool)
    usable = off & np.isfinite(D_boundary) & np.isfinite(D_bulk) & (D_boundary > 0)
    ratios = np.full((n, n), np.nan)
    ratios[usable] = D_bulk[usable] / D_boundary[usable]
    vals = ratios[usable]
    return {
        "ratios": ratios,
        "n_usable": int(usable.sum()),
        "n_skipped": int(off.sum() - usable.sum()),
        "min": float(vals.min()), "max": float(vals.max()),
        "mean": float(vals.mean()), "std": float(vals.std()),
        "consistent": bool(np.allclose(vals, vals[0], rtol=1e-9, atol=1e-12)),
        "spread_ratio": float(vals.max() / vals.min()),
    }


def rescale(D, factor):
    """Multiply the finite entries of a distance matrix by `factor`."""
    out = D.copy()
    finite = np.isfinite(out)
    out[finite] = out[finite] * factor
    return out


# ==========================================================================
# 7. ONE BOND DIMENSION
# ==========================================================================

def analyse(n_sites, bond_dim, maxdim=1, verbose=True):
    """Full rung-1 analysis at one bond dimension."""
    logD = float(np.log(bond_dim))
    D_bd, n_singular = boundary_distance_matrix(n_sites, bond_dim)
    D_bk = bulk_distance_matrix(n_sites)

    dgms_bd = barcodes(D_bd, maxdim)
    dgms_bk = barcodes(D_bk, maxdim)

    # The RT prediction: phi = log(D) applied to the boundary side.
    D_bd_rescaled = rescale(D_bd, logD)
    dgms_bd_rescaled = barcodes(D_bd_rescaled, maxdim)

    w_raw, note_raw = {}, {}
    w_scaled, note_scaled = {}, {}
    for dim in range(maxdim + 1):
        w_raw[dim], note_raw[dim] = wasserstein_safe(dgms_bd[dim], dgms_bk[dim])
        w_scaled[dim], note_scaled[dim] = wasserstein_safe(
            dgms_bd_rescaled[dim], dgms_bk[dim])

    return {
        "bond_dim": bond_dim, "logD": logD,
        "D_boundary": D_bd, "D_bulk": D_bk,
        "D_boundary_rescaled": D_bd_rescaled,
        "n_singular": n_singular,
        "dgms_boundary": dgms_bd, "dgms_bulk": dgms_bk,
        "dgms_boundary_rescaled": dgms_bd_rescaled,
        "phi": measure_phi(D_bd, D_bk),
        "w_raw": w_raw, "w_raw_notes": note_raw,
        "w_scaled": w_scaled, "w_scaled_notes": note_scaled,
    }


# ==========================================================================
# 8. PLOTS
# ==========================================================================

def plot_barcodes(res, out_path, n_sites):
    """Barcodes for both sides (and the rescaled boundary) side by side."""
    panels = [("Boundary (EPR)\n$d=1/S$", res["dgms_boundary"]),
              (f"Boundary rescaled\n$\\phi=\\log D={res['logD']:.4f}$",
               res["dgms_boundary_rescaled"]),
              ("Bulk (ER)\n$d=$ bond geodesic", res["dgms_bulk"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=False)
    colors = {0: "#2b6cb0", 1: "#c05621"}
    for ax, (title, dgms) in zip(axes, panels):
        y = 0
        for dim, dgm in enumerate(dgms):
            dgm = np.asarray(dgm, dtype=float)
            if len(dgm) == 0:
                continue
            fin = dgm[np.isfinite(dgm[:, 1])]
            inf = dgm[~np.isfinite(dgm[:, 1])]
            for b, d in fin[np.argsort(-(fin[:, 1] - fin[:, 0]))]:
                ax.plot([b, d], [y, y], lw=3, color=colors.get(dim, "gray"),
                        solid_capstyle="butt")
                y += 1
            # infinite bars: drawn to the right edge with an arrow cap
            lim = np.nanmax([np.nanmax(fin[:, 1]) if len(fin) else 1.0, 1.0])
            for b, _ in inf:
                ax.plot([b, lim * 1.12], [y, y], lw=3, ls=":",
                        color=colors.get(dim, "gray"))
                ax.plot([lim * 1.12], [y], marker=">", ms=6,
                        color=colors.get(dim, "gray"))
                y += 1
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("filtration scale")
        ax.set_ylim(-1, max(y, 1))
        ax.set_yticks([])
        ax.grid(alpha=0.25, axis="x")
    axes[0].set_ylabel("bars (H0 blue, H1 orange; dotted = infinite)")
    fig.suptitle(f"RTN rung 1: {n_sites}-site chain, D = {res['bond_dim']}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sweep(sweep, out_path):
    """Wasserstein distance vs log(D), before and after rescaling by log(D)."""
    logs = [r["logD"] for r in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(logs, [r["w_raw"][0] for r in sweep], "o-", label="$W_{H_0}$ raw")
    ax.plot(logs, [r["w_scaled"][0] for r in sweep], "s-",
            label="$W_{H_0}$ after $\\times\\log D$")
    ax.set_xlabel("$\\log D$")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title("H0: boundary vs bulk")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(logs, [r["phi"]["mean"] for r in sweep], "o-", label="measured $\\phi$ (mean)")
    ax.fill_between(logs,
                    [r["phi"]["min"] for r in sweep],
                    [r["phi"]["max"] for r in sweep],
                    alpha=0.2, label="measured $\\phi$ range over pairs")
    ax.plot(logs, logs, "k--", label="prediction $\\phi=\\log D$")
    ax.set_xlabel("$\\log D$")
    ax.set_ylabel("$\\phi$")
    ax.set_title("Rescaling factor: predicted vs measured")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==========================================================================
# 9. REPORT
# ==========================================================================

def print_matrix(D, name, n_sites):
    print(f"  {name}")
    head = "        " + "".join(f"{j:>10d}" for j in range(1, n_sites + 1))
    print(head)
    for i in range(n_sites):
        cells = []
        for j in range(n_sites):
            v = D[i, j]
            cells.append(f"{'inf':>10}" if np.isinf(v) else f"{v:10.6f}")
        print(f"  {i + 1:>5d} " + "".join(cells))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sites", type=int, default=6)
    p.add_argument("--bond-dim", type=int, default=4,
                   help="bond dimension for the detailed rung-1 report")
    p.add_argument("--sweep", type=int, nargs="+", default=DEFAULT_BOND_DIMS)
    p.add_argument("--maxdim", type=int, default=1)
    args = p.parse_args(argv)
    n = args.n_sites

    print("=" * 74)
    print("RTN RUNG 1 -- random tensor network, RT as a theorem")
    print("=" * 74)
    print(f"chain: {n} boundary sites, {n - 1} bonds, open (not periodic)")
    print(f"detailed report at D = {args.bond_dim}; sweep over D = {args.sweep}")
    print("S(A) = min_cut(A) * log(D)  [exact for RTNs, Hayden-Nezami-Qi et al.]")

    # ---- 1/2. the two matrices ----
    res = analyse(n, args.bond_dim, args.maxdim)
    logD = res["logD"]
    print(f"\n[1] DISTANCE MATRICES at D = {args.bond_dim} (log D = {logD:.6f})\n")
    print_matrix(res["D_boundary"], "D_boundary(i,j) = 1 / S({i..j})", n)
    print()
    print_matrix(res["D_bulk"], "D_bulk(i,j) = |pos(b_i) - pos(b_j)|", n)

    print("\n  min_cut / S for each contiguous interval [i..j], i <= j:")
    print("        " + "".join(f"{j:>12d}" for j in range(1, n + 1)))
    for i in range(1, n + 1):
        cells = []
        for j in range(1, n + 1):
            if j < i:
                cells.append(f"{'.':>12}")
            else:
                m = min_cut(i, j, n)
                cells.append(f"{m}c/{m * logD:7.4f}".rjust(12))
        print(f"  {i:>5d} " + "".join(cells))

    print("\n  cut points per interval (relevant to the bulk definition):")
    print("        " + "".join(f"{j:>6d}" for j in range(1, n + 1)))
    for i in range(1, n + 1):
        cells = [f"{'.':>6}" if j < i else f"{n_cut_points(i, j, n):>6d}"
                 for j in range(1, n + 1)]
        print(f"  {i:>5d} " + "".join(cells))

    # ---- metric validation ----
    print("\n[2] METRIC VALIDATION")
    for D, name in ((res["D_boundary"], "D_boundary"), (res["D_bulk"], "D_bulk")):
        print(f"  {name}:")
        for msg in validate_metric(D, name):
            print(f"    - {msg}")

    # ---- 3. barcodes ----
    print(f"\n[3] BARCODES at D = {args.bond_dim} (ripser, maxdim = {args.maxdim})")
    print("  BOUNDARY (EPR) side:")
    for dim, dgm in enumerate(res["dgms_boundary"]):
        describe_barcode(dgm, f"H{dim}")
    print("  BULK (ER) side:")
    for dim, dgm in enumerate(res["dgms_bulk"]):
        describe_barcode(dgm, f"H{dim}")
    print(f"  BOUNDARY rescaled by phi = log D = {logD:.6f}:")
    for dim, dgm in enumerate(res["dgms_boundary_rescaled"]):
        describe_barcode(dgm, f"H{dim}")

    # ---- 4. Wasserstein ----
    print(f"\n[4] WASSERSTEIN DISTANCE (finite bars only; infinite bars dropped)")
    for dim in range(args.maxdim + 1):
        print(f"  H{dim}: raw        W = {res['w_raw'][dim]:.6f}"
              f"   {res['w_raw_notes'][dim]}")
        print(f"      after xlogD  W = {res['w_scaled'][dim]:.6f}"
              f"   {res['w_scaled_notes'][dim]}")

    # ---- 5. identical? ----
    print("\n[5] ARE THE BARCODES IDENTICAL?")
    for tag, dgms in (("raw", res["dgms_boundary"]),
                      ("rescaled", res["dgms_boundary_rescaled"])):
        same = all(
            np.array_equal(np.sort(finite_bars(a), axis=0),
                           np.sort(finite_bars(b), axis=0))
            for a, b in zip(dgms, res["dgms_bulk"]))
        print(f"  boundary ({tag}) vs bulk: {'IDENTICAL' if same else 'NOT identical'}")

    # ---- 6. the map phi ----
    print("\n[6] THE MAP phi: measured pair by pair as d_bulk / d_boundary")
    ph = res["phi"]
    print_matrix(np.nan_to_num(ph["ratios"], nan=0.0),
                 "phi(i,j)  (0 = pair skipped: infinite or zero distance)", n)
    print(f"\n  usable pairs          : {ph['n_usable']} of {n * (n - 1)} ordered")
    print(f"  skipped pairs         : {ph['n_skipped']}")
    print(f"  phi range over pairs  : [{ph['min']:.6f}, {ph['max']:.6f}]")
    print(f"  phi mean +- sd        : {ph['mean']:.6f} +- {ph['std']:.6f}")
    print(f"  max/min spread        : {ph['spread_ratio']:.6f}")
    print(f"  single constant phi?  : {'YES' if ph['consistent'] else 'NO'}")
    print(f"  prediction phi = logD : {logD:.6f}")

    # ---- 7. sweep ----
    print(f"\n[7] BOND-DIMENSION SWEEP over D = {args.sweep}")
    sweep = [analyse(n, D, args.maxdim) for D in args.sweep]
    print(f"  {'D':>4} {'log D':>9} {'W_H0 raw':>11} {'W_H0 scaled':>12} "
          f"{'W_H1 raw':>10} {'W_H1 scaled':>12} {'phi min':>9} {'phi max':>9} "
          f"{'const?':>7}")
    for r in sweep:
        print(f"  {r['bond_dim']:>4d} {r['logD']:>9.6f} "
              f"{r['w_raw'][0]:>11.6f} {r['w_scaled'][0]:>12.6f} "
              f"{r['w_raw'][1]:>10.6f} {r['w_scaled'][1]:>12.6f} "
              f"{r['phi']['min']:>9.6f} {r['phi']['max']:>9.6f} "
              f"{'YES' if r['phi']['consistent'] else 'NO':>7}")

    # ---- plots ----
    bar_png = HERE / "rtn_rung1_barcodes.png"
    sweep_png = HERE / "rtn_sweep.png"
    plot_barcodes(res, bar_png, n)
    plot_sweep(sweep, sweep_png)
    print(f"\n  wrote {bar_png}")
    print(f"  wrote {sweep_png}")

    # ---- 8. verdict ----
    print("\n[8] VERDICT")
    scaled_zero = all(abs(r["w_scaled"][0]) < 1e-9 for r in sweep)
    const_phi = all(r["phi"]["consistent"] for r in sweep)
    print(f"  Does phi = log(D) make the barcodes match?  "
          f"{'YES' if scaled_zero else 'NO'}")
    print(f"  Is a single constant phi even available?    "
          f"{'YES' if const_phi else 'NO'}")
    w_scaled_vals = {round(r["w_scaled"][0], 10) for r in sweep}
    print(f"  W_H0 after rescaling, over the whole sweep: "
          f"{sorted(w_scaled_vals)}")
    print(f"  (one value across all D means the rescaling did remove the "
          f"D-dependence,\n   even where it did not make the barcodes agree)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
