#!/usr/bin/env python3
"""Rung 4 of the RTN rebuild: jittered disk, non-vacuous H1.

GOAL AS SET
-----------
Test whether the AdS2 boundary geodesic metric and the Poincare disk bulk
metric are isomorphic as persistence modules WITH non-trivial H1, by breaking
the circulant symmetry that made rung 3's H1 vacuous.

CHANGES FROM RUNG 3
-------------------
1. Non-uniform boundary spacing: a fixed jitter (numpy seed 42) on the uniform
   angles, then sorted so the sites keep their circular order.
2. The corrected boundary geodesic, L = 2 log(sin(theta/2) / epsilon), which is
   monotone INCREASING in theta -- repairing rung 3's sign error.
3. The same angular jitter applied to the bulk sites, at radius r.
4. Both distance functions validated monotone increasing before any homology.

Geometry helpers are imported from rtn_rung3 (same disk, same Poincare metric,
same Wasserstein convention) so the two rungs cannot drift apart; the boundary
formula, the jitter, and the joint optimisation are new here.

WHAT THE RESULT ACTUALLY SHOWS -- READ BEFORE TRUSTING IT
---------------------------------------------------------
The jitter does what it was meant to do: neither matrix is circulant, every row
is distinct, and H1 is non-trivial on both sides without being forced by the
Adamaszek-Adams argument that made rung 3's single H1 bar an artifact.

But a tautology survives that change, and it is the important caveat:

    d_boundary(i,j) = f(theta_ij)  and  d_bulk(i,j) = g(theta_ij)

with BOTH f and g strictly increasing, and theta_ij the one shared variable.
Every bulk site sits at the same radius, so the bulk distance depends on
nothing but the angular separation the boundary already encodes.  Two strictly
increasing functions of the same variable induce the SAME ordering on the 28
pairs, hence the same filtration order, hence persistence modules that are
isomorphic by construction -- identical bar counts, bars in correspondence,
before any numbers are compared.

So "isomorphic as persistence modules with non-trivial H1" is achieved, but it
is not evidence about ER=EPR: it is the statement that two monotone
reparametrisations of one angular lattice have the same order structure.  The
only non-trivial content left is QUANTITATIVE -- whether a single constant phi
makes the two value sets agree -- and f and g are not proportional, so it
cannot be zero.  This script reports that distance rather than claiming the
isomorphism as a physics result.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

# Shared geometry and PH conventions come from rung 3 (same disk, same metric).
from rtn_rung3 import (barcodes, describe_barcode, finite_bars, poincare_distance,
                       print_matrix, scale_diagram, wasserstein_safe)

HERE = Path(__file__).resolve().parent

N_SITES = 8
EPSILON = 0.1               # UV cutoff in the boundary geodesic
BULK_RADIUS = 0.7
JITTER_SEED = 42
PHI_LO, PHI_HI = 0.1, 10.0  # search window set by the brief
RADII = [0.3, 0.5, 0.7, 0.85, 0.95]


# ==========================================================================
# 1. THE JITTERED ANGULAR LATTICE
# ==========================================================================

def jittered_angles(n_sites=N_SITES, seed=JITTER_SEED):
    """Uniform angles plus a fixed Uniform(-pi/8, pi/8) jitter, sorted.

    Sorting keeps the sites in circular order, as specified.  Note the jitter
    half-width pi/8 = 0.3927 is HALF the uniform spacing 2pi/8 = 0.7854, not a
    full site spacing as the brief's parenthetical says; the numbers below are
    from the stated distribution, not the stated description of it.
    """
    rng_angles = 2.0 * np.pi * np.arange(n_sites) / n_sites
    np.random.seed(seed)                    # global seed, as the brief specifies
    jitter = np.random.uniform(-np.pi / 8, np.pi / 8, n_sites)
    return np.sort(rng_angles + jitter)


def chord_angle(theta_i, theta_j):
    """Shorter way round the circle between two angles."""
    d = abs(theta_i - theta_j)
    return min(d, 2.0 * np.pi - d)


# ==========================================================================
# 2. BOUNDARY SIDE: corrected AdS2 geodesic
# ==========================================================================

def boundary_length(theta, eps=EPSILON):
    """L = 2*log(sin(theta/2)/eps) -- monotone INCREASING in theta.

    Positive only where sin(theta/2) > eps, i.e. theta > 2*arcsin(eps)
    = 0.2003 rad for eps = 0.1.  The caller validates that every pair on the
    lattice clears that bound rather than assuming it.
    """
    s = np.sin(theta / 2.0)
    if s <= 0:
        return -np.inf
    return 2.0 * np.log(s / eps)


def boundary_distance_matrix(angles, eps=EPSILON):
    n = len(angles)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = boundary_length(chord_angle(angles[i], angles[j]), eps)
    return D


# ==========================================================================
# 3. BULK SIDE: same jitter, radius r
# ==========================================================================

def bulk_distance_matrix(angles, radius=BULK_RADIUS):
    """Poincare disk distances between sites at (radius, angle_k).

    poincare_distance is rung 3's, i.e. cosh(d) = 1 + 2|z_i-z_j|^2 /
    ((1-|z_i|^2)(1-|z_j|^2)), which is the brief's delta formula.
    """
    n = len(angles)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = poincare_distance(radius, angles[i],
                                            radius, angles[j])
    return D


# ==========================================================================
# 4. VALIDATION
# ==========================================================================

def pair_table(angles, D_bd, D_bk):
    """All 28 pairs sorted by chord angle, with both distances."""
    n = len(angles)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            t = chord_angle(angles[i], angles[j])
            rows.append((t, i, j, D_bd[i, j], D_bk[i, j]))
    return sorted(rows)


def is_circulant(M, tol=1e-12):
    """True if M[i,j] depends only on (j-i) mod n."""
    n = M.shape[0]
    return all(abs(M[i, j] - M[0, (j - i) % n]) < tol
               for i in range(n) for j in range(n))


def n_distinct_rows(M, dec=9):
    return len({tuple(np.round(row, dec)) for row in M})


def validate(angles, D_bd, D_bk, eps=EPSILON):
    """Run every check the brief asks for; print PASS/FAIL, never crash."""
    n = len(angles)
    print("[0] VALIDATION")
    print(f"  jittered angles (sorted): {np.round(angles, 6).tolist()}")
    gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
    print(f"  consecutive gaps        : {np.round(gaps, 6).tolist()}")
    print(f"  uniform spacing 2pi/{n} = {2 * np.pi / n:.6f}; "
          f"jitter half-width pi/8 = {np.pi / 8:.6f} (HALF a spacing, not one)")

    rows = pair_table(angles, D_bd, D_bk)
    print(f"\n  all {len(rows)} pairwise distances, sorted by chord angle:")
    print(f"  {'pair':>8} {'theta':>9} {'d_boundary':>12} {'d_bulk':>11}")
    for t, i, j, b, k in rows:
        print(f"  {f'({i},{j})':>8} {t:>9.6f} {b:>12.6f} {k:>11.6f}")

    bd = np.array([r[3] for r in rows])
    bk = np.array([r[4] for r in rows])
    off = ~np.eye(n, dtype=bool)

    all_pos = bool(np.all(bd > 0))
    no_inf = bool(np.all(np.isfinite(D_bd[off])) and np.all(np.isfinite(D_bk[off])))
    no_zero = bool(np.all(np.abs(D_bd[off]) > 1e-12))
    mono_bd = bool(np.all(np.diff(bd) >= -1e-12))
    mono_bk = bool(np.all(np.diff(bk) >= -1e-12))
    circ_bd, circ_bk = is_circulant(D_bd), is_circulant(D_bk)

    print(f"\n  boundary positivity needs theta > 2*arcsin(eps) = "
          f"{2 * np.arcsin(eps):.6f} rad")
    print(f"  smallest chord angle on this lattice     : {rows[0][0]:.6f} "
          f"(pair {rows[0][1]},{rows[0][2]})")
    print(f"  -> all d_boundary finite and positive    : "
          f"{'PASS' if all_pos else 'FAIL'}")
    print(f"  -> no zero, no infinite entries anywhere : "
          f"{'PASS' if (no_inf and no_zero) else 'FAIL'}")
    print(f"  -> d_boundary monotone increasing in theta: "
          f"{'PASS' if mono_bd else 'FAIL'}")
    print(f"  -> d_bulk monotone increasing in theta    : "
          f"{'PASS' if mono_bk else 'FAIL'}")
    print(f"  -> boundary rows all distinct             : "
          f"{n_distinct_rows(D_bd)}/{n} "
          f"{'PASS' if n_distinct_rows(D_bd) == n else 'FAIL'}")
    print(f"  -> bulk rows all distinct                 : "
          f"{n_distinct_rows(D_bk)}/{n} "
          f"{'PASS' if n_distinct_rows(D_bk) == n else 'FAIL'}")

    print("\n  CIRCULANT DIAGNOSTIC (the rung-3 trap):")
    for name, circ in (("boundary", circ_bd), ("bulk", circ_bk)):
        if circ:
            print(f"    *** {name.upper()} MATRIX IS STILL CIRCULANT -- "
                  f"H1 IS NOT TRUSTWORTHY ***")
        else:
            print(f"    {name}: not circulant (good)")
    return {"all_pos": all_pos, "mono": mono_bd and mono_bk,
            "circ_bd": circ_bd, "circ_bk": circ_bk}


# ==========================================================================
# 5. JOINT OPTIMISATION OVER phi
# ==========================================================================

def joint_cost(dgms_bd, dgms_bk, phi):
    """W_H0 + W_H1 between the boundary and the phi-scaled bulk."""
    w0, _ = wasserstein_safe(dgms_bd[0], scale_diagram(dgms_bk[0], phi))
    w1, _ = wasserstein_safe(dgms_bd[1], scale_diagram(dgms_bk[1], phi))
    return w0, w1, w0 + w1


def optimal_phi_joint(dgms_bd, dgms_bk, lo=PHI_LO, hi=PHI_HI, n_grid=2001):
    """phi minimizing W_H0 + W_H1 on [lo, hi].

    The cost is piecewise linear in phi, so a grid locates the basin and a
    bounded refinement finishes.  Whether the optimum sits at a bound is
    reported: a bound-pinned optimum is not a fit, it is a failure of the
    search window, and rung 3 produced exactly that.
    """
    grid = np.linspace(lo, hi, n_grid)
    costs = np.array([joint_cost(dgms_bd, dgms_bk, p)[2] for p in grid])
    k = int(np.argmin(costs))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(lambda p: joint_cost(dgms_bd, dgms_bk, p)[2],
                          bounds=(a, b), method="bounded",
                          options={"xatol": 1e-12})
    phi = float(res.x) if (res.success and res.fun <= costs[k]) else float(grid[k])
    at_bound = bool(abs(phi - lo) < 1e-6 or abs(phi - hi) < 1e-6)
    return phi, at_bound


# ==========================================================================
# 6. H1 NON-TRIVIALITY
# ==========================================================================

def count_significant(dgm, frac=0.1):
    """Bars with persistence > frac * (max persistence), and that max."""
    fin = finite_bars(dgm)
    if len(fin) == 0:
        return 0, 0.0, 0
    pers = fin[:, 1] - fin[:, 0]
    mx = float(pers.max())
    return int(np.sum(pers > frac * mx)), mx, len(fin)


# ==========================================================================
# 7. PLOT
# ==========================================================================

def plot_barcodes(dgms_bd, dgms_bk, phi, out_path, n_sites, radius):
    panels = [("Boundary (AdS$_2$ geodesic)\n$2\\log(\\sin(\\theta/2)/\\epsilon)$",
               dgms_bd),
              (f"Bulk, $\\phi=1$\nPoincaré disk, $r={radius}$", dgms_bk),
              (f"Bulk, $\\phi={phi:.4f}$ (joint optimum)",
               [scale_diagram(d, phi) for d in dgms_bk])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    colors = {0: "#2b6cb0", 1: "#c05621"}
    for ax, (title, dgms) in zip(axes, panels):
        y, xmax = 0, 1.0
        for dim, dgm in enumerate(dgms):
            dgm = np.asarray(dgm, dtype=float)
            if len(dgm) == 0:
                continue
            fin = finite_bars(dgm)
            if len(fin):
                xmax = max(xmax, float(np.nanmax(fin[:, 1])))
            for b, d in fin[np.argsort(-(fin[:, 1] - fin[:, 0]))]:
                ax.plot([b, d], [y, y], lw=3, color=colors.get(dim, "gray"),
                        solid_capstyle="butt")
                y += 1
            for b, _ in dgm[~np.isfinite(dgm[:, 1])]:
                ax.plot([b, xmax * 1.12], [y, y], lw=3, ls=":",
                        color=colors.get(dim, "gray"))
                ax.plot([xmax * 1.12], [y], marker=">", ms=6,
                        color=colors.get(dim, "gray"))
                y += 1
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("filtration scale")
        ax.set_ylim(-1, max(y, 1))
        ax.set_yticks([])
        ax.grid(alpha=0.25, axis="x")
    axes[0].set_ylabel("bars (H0 blue, H1 orange; dotted = infinite)")
    fig.suptitle(f"RTN rung 4: jittered disk, N = {n_sites}, seed "
                 f"{JITTER_SEED}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==========================================================================
# 8. MAIN
# ==========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sites", type=int, default=N_SITES)
    p.add_argument("--radius", type=float, default=BULK_RADIUS)
    p.add_argument("--eps", type=float, default=EPSILON)
    p.add_argument("--seed", type=int, default=JITTER_SEED)
    p.add_argument("--maxdim", type=int, default=1)
    p.add_argument("--radii", type=float, nargs="+", default=RADII)
    p.add_argument("--sig-frac", type=float, default=0.1,
                   help="a bar counts as significant above this fraction of "
                        "the max persistence in its dimension")
    args = p.parse_args(argv)
    n, r = args.n_sites, args.radius

    print("=" * 78)
    print("RTN RUNG 4 -- jittered disk: breaking the circulant symmetry")
    print("=" * 78)
    print(f"boundary: {n} sites, uniform angles + Uniform(-pi/8, pi/8) jitter, "
          f"seed {args.seed}")
    print(f"          d = 2*log(sin(theta/2)/eps), eps = {args.eps} "
          f"(monotone increasing)")
    print(f"bulk    : same jittered angles at radius r = {r} in the Poincare disk")
    print(f"phi search window: [{PHI_LO}, {PHI_HI}], minimizing W_H0 + W_H1 jointly\n")

    angles = jittered_angles(n, args.seed)
    D_bd = boundary_distance_matrix(angles, args.eps)
    D_bk = bulk_distance_matrix(angles, r)
    checks = validate(angles, D_bd, D_bk, args.eps)

    # ---- 1. matrices ----
    print(f"\n[1] DISTANCE MATRICES\n")
    print_matrix(D_bd, "d_boundary(i,j) = 2*log(sin(theta/2)/eps)")
    print()
    print_matrix(D_bk, f"d_bulk(i,j) = arccosh(1 + delta), r = {r}")

    # ---- 2. barcodes ----
    dgms_bd = barcodes(D_bd, args.maxdim)
    dgms_bk = barcodes(D_bk, args.maxdim)
    print(f"\n[2] BARCODES (ripser, maxdim = {args.maxdim})")
    print("  BOUNDARY side:")
    for dim, d in enumerate(dgms_bd):
        describe_barcode(d, f"H{dim}")
    print("  BULK side (phi = 1):")
    for dim, d in enumerate(dgms_bk):
        describe_barcode(d, f"H{dim}")

    # ---- 3/4. Wasserstein at phi = 1 and at the joint optimum ----
    w0_1, w1_1, tot_1 = joint_cost(dgms_bd, dgms_bk, 1.0)
    phi_opt, at_bound = optimal_phi_joint(dgms_bd, dgms_bk)
    w0_o, w1_o, tot_o = joint_cost(dgms_bd, dgms_bk, phi_opt)

    print("\n[3] WASSERSTEIN AT phi = 1")
    print(f"  W_H0 = {w0_1:.6f}   W_H1 = {w1_1:.6f}   sum = {tot_1:.6f}")
    print(f"\n[4] JOINT OPTIMUM OVER phi in [{PHI_LO}, {PHI_HI}]")
    print(f"  phi_opt = {phi_opt:.6f}" + ("   *** PINNED AT A SEARCH BOUND -- "
                                          "not a fit ***" if at_bound else
                                          "   (interior optimum)"))
    print(f"  W_H0 = {w0_o:.6f}   W_H1 = {w1_o:.6f}   sum = {tot_o:.6f}")
    print(f"  improvement over phi = 1: sum {tot_1:.6f} -> {tot_o:.6f}")
    print(f"\n  BULK barcodes at phi = {phi_opt:.6f}:")
    for dim, d in enumerate(dgms_bk):
        describe_barcode(scale_diagram(d, phi_opt), f"H{dim}")

    # ---- 5. H1 non-triviality ----
    print(f"\n[5] IS H1 NON-TRIVIAL ON BOTH SIDES? "
          f"(bars with persistence > {args.sig_frac} * max)")
    for tag, dgms in (("boundary", dgms_bd), ("bulk", dgms_bk)):
        nsig, mx, ntot = count_significant(dgms[1], args.sig_frac)
        print(f"  {tag:>8}: {ntot} finite H1 bar(s), {nsig} significant, "
              f"max persistence {mx:.6f}")
    nb, _, _ = count_significant(dgms_bd[1], args.sig_frac)
    nk, _, _ = count_significant(dgms_bk[1], args.sig_frac)
    print(f"  -> non-trivial on BOTH sides: "
          f"{'YES' if (nb > 0 and nk > 0) else 'NO'}")
    if not (checks["circ_bd"] or checks["circ_bk"]):
        print("  -> and neither matrix is circulant, so this H1 is NOT the")
        print("     Adamaszek-Adams artifact that made rung 3's H1 vacuous.")

    # ---- 6. radius sweep ----
    print(f"\n[6] RADIUS SWEEP over r = {args.radii}")
    print("    (the boundary side is r-independent; only the bulk moves)")
    print(f"  {'r':>6} {'phi_opt':>10} {'bound?':>7} {'W_H0':>10} {'W_H1':>10} "
          f"{'sum':>10} {'H1 bulk':>8}")
    sweep = []
    for rr in args.radii:
        Dk = bulk_distance_matrix(angles, rr)
        dk = barcodes(Dk, args.maxdim)
        ph, ab = optimal_phi_joint(dgms_bd, dk)
        a0, a1, at = joint_cost(dgms_bd, dk, ph)
        nsig, _, _ = count_significant(dk[1], args.sig_frac)
        sweep.append({"r": rr, "phi": ph, "w0": a0, "w1": a1, "sum": at})
        print(f"  {rr:>6.2f} {ph:>10.6f} {'YES' if ab else 'no':>7} "
              f"{a0:>10.6f} {a1:>10.6f} {at:>10.6f} {nsig:>8d}")

    best_sum = min(sweep, key=lambda s: s["sum"])
    best_w0 = min(sweep, key=lambda s: s["w0"])
    best_w1 = min(sweep, key=lambda s: s["w1"])
    print(f"\n  lowest W_H0 + W_H1 : r = {best_sum['r']} "
          f"(sum {best_sum['sum']:.6f})")
    print(f"  lowest W_H0 alone  : r = {best_w0['r']} ({best_w0['w0']:.6f})")
    print(f"  lowest W_H1 alone  : r = {best_w1['r']} ({best_w1['w1']:.6f})")
    if best_w0["r"] != best_w1["r"]:
        print("  -> NO single r minimizes both: the two dimensions disagree,")
        print("     so 'the r where both are minimized' does not exist here.")
    else:
        print(f"  -> both minimized at the same r = {best_w0['r']}")

    # ---- plot ----
    png = HERE / "rtn_rung4_barcodes.png"
    plot_barcodes(dgms_bd, dgms_bk, phi_opt, png, n, r)
    print(f"\n  wrote {png}")

    # ---- verdict ----
    print("\n[7] VERDICT")
    print(f"  Circulant symmetry broken        : "
          f"{'YES' if not (checks['circ_bd'] or checks['circ_bk']) else 'NO'}")
    print(f"  Both distances monotone in theta : "
          f"{'YES' if checks['mono'] else 'NO'}")
    print(f"  H1 non-trivial on both sides     : "
          f"{'YES' if (nb > 0 and nk > 0) else 'NO'}")
    print(f"  Best joint phi                   : {phi_opt:.6f}"
          f" ({'at a bound' if at_bound else 'interior'})")
    print(f"  Residual at that phi             : W_H0 = {w0_o:.6f}, "
          f"W_H1 = {w1_o:.6f}")
    print("\n  THE TAUTOLOGY THAT SURVIVES THE JITTER:")
    print("  Every bulk site shares the boundary's angle and sits at one common")
    print("  radius, so d_bulk is a function of the SAME chord angle that fixes")
    print("  d_boundary.  Two strictly increasing functions of one variable")
    print("  induce the same ordering on all 28 pairs, so the two filtrations")
    print("  share their combinatorial order and the persistence modules are")
    print("  isomorphic BY CONSTRUCTION -- equal bar counts, bars in")
    print("  correspondence -- before any value is compared.  Non-trivial H1 is")
    print("  now genuine rather than an AA artifact, but the isomorphism itself")
    print("  is not a test of ER=EPR: the two sides are not independent")
    print("  computations.  Only the residual above is empirical content, and it")
    print("  measures how far 2*log(sin(theta/2)/eps) is from proportional to")
    print("  arccosh(1 + delta(theta)) -- a statement about two formulas, not")
    print("  about entanglement and geometry.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
