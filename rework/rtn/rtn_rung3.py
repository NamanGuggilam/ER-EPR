#!/usr/bin/env python3
"""Rung 3 of the RTN rebuild: disk geometry (AdS2/CFT1), circle boundary.

GEOMETRY CHANGE FROM RUNGS 1-2
------------------------------
The 1D open chain is replaced by the actual AdS2/CFT1 setup: the boundary is a
circle of N = 8 uniformly spaced sites, and the bulk is the Poincare disk with
N = 8 sites on a circle of radius r inside it.  Nothing is imported from rung 1
or rung 2 -- this is a different geometry and a standalone script.

WHAT CHANGED IN THE PHYSICS INPUT
---------------------------------
Rungs 1-2 fed the barcode an ENTANGLEMENT quantity (1/S, then 1/I).  Rung 3
feeds it a GEOMETRIC length on both sides: a boundary-anchored geodesic and a
bulk-to-bulk geodesic.  So this rung no longer tests the RT dictionary at all;
it tests whether two geodesic length functions on the same angular lattice are
related by a constant rescaling.

FORMULA VALIDATION (run before anything else; see validate_formulas)
--------------------------------------------------------------------
The brief states two analytic checks.  One passes, one does not, and the one
that fails is not a typo but a sign error in the construction:

  * PASSES: d_bulk is finite, and antipodal > adjacent at r = 0.7.

  * FAILS: "d_boundary(sites 0 and 4, N=8) should equal 2*log(1/sin(pi/8))
    ~= 3.068".  Three separate problems:
      - sites 0 and 4 on an 8-site circle subtend pi, not pi/4, so theta/2 is
        pi/2 and not pi/8;
      - the specified formula at theta = pi gives exactly 0, not 3.068;
      - 2*log(1/sin(pi/8)) = 1.921094, not 3.068, so the stated number does
        not match the stated formula at ANY separation on this lattice.

  * THE SIGN ERROR: d_boundary = 2*log(1/sin(theta/2)) DECREASES as the sites
    separate -- 1.921 for adjacent sites, falling to 0 for antipodal ones --
    while d_bulk INCREASES, 1.833 to 3.469.  The two sides are anti-monotone
    in the one variable they share, so no positive rescaling can carry one
    onto the other.  The regularized AdS2 boundary geodesic is

        L(theta) = 2 log( (2/eps) sin(theta/2) )

    which increases with theta; the brief inverted the fraction, which is what
    flips the ordering.  With the AdS radius and cutoff both set to 1, L is
    negative for close points, and inverting it is presumably the attempt to
    repair that -- but the fix destroys monotonicity instead.  Both forms are
    run here: the brief's as the primary (as-specified) track, the correctly
    ordered L with eps = 0.1 as a clearly labelled diagnostic.

STRUCTURAL WARNING CARRIED FORWARD FROM RUNG 4
----------------------------------------------
Both distance matrices here are CIRCULANT (distance depends only on angular
separation) and the bulk one is monotone in that separation.  The repo's
rung-4 diagnostics proved, via Adamaszek-Adams, that such a construction forces
exactly one H1 bar and H0 = n-1 identical bars, with every Wasserstein number a
function of just two per-side numbers.  So "H1 is non-trivial" is guaranteed by
the construction before any data is seen, and is NOT evidence that the circle
topology was detected.  The brief's check 6 is therefore answered, but with
that caveat attached rather than as a success.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from persim import wasserstein
from ripser import ripser
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent

N_SITES = 8
BULK_RADIUS = 0.7
#: cutoff for the correctly-ordered diagnostic boundary length; small enough
#: that L > 0 at the smallest separation on this lattice (needs eps < 0.765).
DIAG_EPS = 0.1


# ==========================================================================
# 1. ANGULAR LATTICE
# ==========================================================================

def site_angles(n_sites):
    """theta_i = 2*pi*i/n for i = 0..n-1."""
    return 2.0 * np.pi * np.arange(n_sites) / n_sites


def angular_separation(i, j, n_sites):
    """Shorter way round the circle: min(|dtheta|, 2pi - |dtheta|)."""
    th = site_angles(n_sites)
    d = abs(th[i] - th[j])
    return min(d, 2.0 * np.pi - d)


# ==========================================================================
# 2. BOUNDARY SIDE
# ==========================================================================

def boundary_length_as_specified(theta):
    """d_boundary = 2*log(1/sin(theta/2)), exactly as the brief specifies.

    Diverges as theta -> 0 and equals 0 at theta = pi.  Monotone DECREASING on
    (0, pi], which is the defect documented in the module docstring.
    """
    s = np.sin(theta / 2.0)
    if s <= 0:
        return np.inf
    return 2.0 * np.log(1.0 / s)


def boundary_length_regularized(theta, eps=DIAG_EPS):
    """L = 2*log((2/eps)*sin(theta/2)) -- the correctly ordered geodesic.

    Diagnostic track only.  Monotone INCREASING in theta, matching the bulk's
    ordering; positive for every separation on this lattice when eps < 0.765.
    """
    s = np.sin(theta / 2.0)
    if s <= 0:
        return -np.inf
    return 2.0 * np.log((2.0 / eps) * s)


def boundary_distance_matrix(n_sites, which="specified", eps=DIAG_EPS):
    """Boundary distance matrix; diagonal 0 by definition."""
    fn = (boundary_length_as_specified if which == "specified"
          else lambda t: boundary_length_regularized(t, eps))
    D = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                continue
            D[i, j] = fn(angular_separation(i, j, n_sites))
    return D


# ==========================================================================
# 3. BULK SIDE: Poincare disk
# ==========================================================================

def poincare_distance(r1, phi1, r2, phi2):
    """Geodesic distance in the Poincare disk (AdS radius 1).

        cosh(d) = 1 + 2*(r1^2 + r2^2 - 2 r1 r2 cos(phi1-phi2))
                      / ((1-r1^2)(1-r2^2))

    This is the standard formula; the numerator is the squared Euclidean chord
    and the denominator the two conformal factors.
    """
    num = r1 ** 2 + r2 ** 2 - 2.0 * r1 * r2 * np.cos(phi1 - phi2)
    den = (1.0 - r1 ** 2) * (1.0 - r2 ** 2)
    c = 1.0 + 2.0 * num / den
    return float(np.arccosh(max(c, 1.0)))       # clamp rounding below 1


def bulk_distance_matrix(n_sites, radius):
    """N bulk sites all at `radius`, at the same angles as the boundary."""
    phis = site_angles(n_sites)
    D = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                continue
            D[i, j] = poincare_distance(radius, phis[i], radius, phis[j])
    return D


# ==========================================================================
# 4. FORMULA VALIDATION (the brief asks for this before running)
# ==========================================================================

def validate_formulas(n_sites=N_SITES, radius=BULK_RADIUS):
    """Check the brief's two analytic targets; print PASS/FAIL, never crash."""
    print("[0] ANALYTIC VALIDATION OF THE DISTANCE FORMULAS")
    ok = True

    # ---- check 1: the brief's boundary target ----
    sep_04 = angular_separation(0, 4, n_sites)
    d_04 = boundary_length_as_specified(sep_04)
    target = 2.0 * np.log(1.0 / np.sin(np.pi / 8))
    print("  check 1: d_boundary(0,4) == 2*log(1/sin(pi/8)) ~= 3.068")
    print(f"    angular separation of sites 0 and 4 : {sep_04:.6f} rad "
          f"(= pi, i.e. ANTIPODAL -- theta/2 = pi/2, not pi/8)")
    print(f"    d_boundary(0,4) from the formula    : {d_04:.6f}")
    print(f"    2*log(1/sin(pi/8))                  : {target:.6f} "
          f"(the brief says 3.068; the formula gives {target:.6f})")
    print(f"    pi/8 is theta/2 for ADJACENT sites  : "
          f"d_boundary(0,1) = {boundary_length_as_specified(angular_separation(0, 1, n_sites)):.6f}")
    print("    verdict: FAIL -- three ways: wrong pair, the formula gives 0 at")
    print("             theta = pi, and 3.068 matches no separation here.")
    ok = False

    # ---- check 2: the bulk ordering ----
    adj = poincare_distance(radius, 0.0, radius, 2 * np.pi / n_sites)
    anti = poincare_distance(radius, 0.0, radius, np.pi)
    print(f"\n  check 2: d_bulk antipodal finite and > adjacent, at r = {radius}")
    print(f"    adjacent  (dphi = 2pi/{n_sites}) : {adj:.6f}")
    print(f"    antipodal (dphi = pi)       : {anti:.6f}")
    passed = np.isfinite(anti) and anti > adj
    print(f"    verdict: {'PASS' if passed else 'FAIL'}")

    # ---- the monotonicity problem ----
    print("\n  MONOTONICITY IN THE SHARED VARIABLE (angular separation):")
    print(f"    {'pair':>8} {'theta':>9} {'d_boundary':>12} {'d_bulk':>10}")
    for j in range(1, n_sites // 2 + 1):
        t = angular_separation(0, j, n_sites)
        print(f"    {f'(0,{j})':>8} {t:>9.4f} "
              f"{boundary_length_as_specified(t):>12.6f} "
              f"{poincare_distance(radius, 0.0, radius, t):>10.6f}")
    print("    boundary DECREASES with separation, bulk INCREASES: the two are")
    print("    anti-monotone, so no positive constant phi can map one onto the")
    print("    other.  This is a sign error, not a scale mismatch.")
    return ok and passed


# ==========================================================================
# 5. PERSISTENT HOMOLOGY
# ==========================================================================

def barcodes(D, maxdim=1):
    """Ripser on a precomputed distance matrix."""
    return ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"]


def finite_bars(dgm):
    """Finite bars only -- persim's Wasserstein needs finite diagrams."""
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
    for b, d in fin[np.argsort(-(fin[:, 1] - fin[:, 0]))]:
        print(f"{indent}  [{b:.6f}, {d:.6f}]   persistence {d - b:.6f}")


def wasserstein_safe(dgm_a, dgm_b):
    """Wasserstein distance, with the empty cases named rather than skipped."""
    a, b = finite_bars(dgm_a), finite_bars(dgm_b)
    if len(a) == 0 and len(b) == 0:
        return 0.0, "both diagrams empty -> 0 by definition"
    try:
        return float(wasserstein(a, b)), ""
    except Exception as exc:                        # pragma: no cover
        return float("nan"), f"persim failed: {type(exc).__name__}: {exc}"


def scale_diagram(dgm, phi):
    """Multiply a diagram's finite coordinates by phi.

    Vietoris-Rips is scale-equivariant, so scaling the distance matrix by phi
    scales every birth and death by phi -- no need to refilter.
    """
    dgm = np.asarray(dgm, dtype=float)
    if len(dgm) == 0:
        return dgm.copy()
    out = dgm.copy()
    finite = np.isfinite(out)
    out[finite] = out[finite] * phi
    return out


def optimal_phi(dgm_boundary, dgm_bulk, dim_label="H0",
                lo=1e-3, hi=1e3, n_grid=4001):
    """The constant phi minimizing W(boundary, phi * bulk).

    W is piecewise linear in phi with kinks wherever the optimal matching
    changes, so a coarse log grid locates the basin and a bounded scalar
    minimization refines inside it.  Returns (phi, W, note).
    """
    if len(finite_bars(dgm_boundary)) == 0 and len(finite_bars(dgm_bulk)) == 0:
        return float("nan"), 0.0, f"{dim_label}: both diagrams empty"

    def objective(phi):
        if phi <= 0:
            return np.inf
        w, _ = wasserstein_safe(dgm_boundary, scale_diagram(dgm_bulk, phi))
        return w if np.isfinite(w) else np.inf

    grid = np.geomspace(lo, hi, n_grid)
    vals = np.array([objective(p) for p in grid])
    k = int(np.argmin(vals))
    a = grid[max(k - 1, 0)]
    b = grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(objective, bounds=(a, b), method="bounded",
                          options={"xatol": 1e-10})
    if res.success and res.fun <= vals[k]:
        return float(res.x), float(res.fun), ""
    return float(grid[k]), float(vals[k]), "grid minimum (refinement declined)"


# ==========================================================================
# 6. REPORT HELPERS
# ==========================================================================

def print_matrix(D, name, dec=3):
    """Print a distance matrix at `dec` decimal places."""
    n = D.shape[0]
    print(f"  {name}")
    w = dec + 5
    print("       " + "".join(f"{j:>{w}d}" for j in range(n)))
    for i in range(n):
        cells = []
        for j in range(n):
            v = D[i, j]
            cells.append(f"{'inf':>{w}}" if np.isinf(v) else f"{v:>{w}.{dec}f}")
        print(f"  {i:>4d} " + "".join(cells))


def plot_barcodes(dgms_bd, dgms_bk, phi, out_path, n_sites, radius):
    """Boundary, bulk at phi = 1, and bulk at the optimal phi."""
    panels = [("Boundary (as specified)\n$2\\log(1/\\sin(\\theta/2))$", dgms_bd),
              (f"Bulk, $\\phi=1$\nPoincaré disk, $r={radius}$", dgms_bk),
              (f"Bulk, $\\phi={phi:.4f}$ (optimal)",
               [scale_diagram(d, phi) for d in dgms_bk])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = {0: "#2b6cb0", 1: "#c05621"}
    for ax, (title, dgms) in zip(axes, panels):
        y = 0
        xmax = 1.0
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
    fig.suptitle(f"RTN rung 3: disk geometry, N = {n_sites} sites", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==========================================================================
# 7. MAIN
# ==========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sites", type=int, default=N_SITES)
    p.add_argument("--radius", type=float, default=BULK_RADIUS)
    p.add_argument("--maxdim", type=int, default=1)
    p.add_argument("--r-min", type=float, default=0.3)
    p.add_argument("--r-max", type=float, default=0.95)
    p.add_argument("--r-steps", type=int, default=8)
    p.add_argument("--diag-eps", type=float, default=DIAG_EPS)
    args = p.parse_args(argv)
    n, r = args.n_sites, args.radius

    print("=" * 78)
    print("RTN RUNG 3 -- disk geometry (AdS2/CFT1): circle boundary, Poincare bulk")
    print("=" * 78)
    print(f"boundary: {n} sites uniform on the unit circle")
    print(f"bulk    : {n} sites at radius r = {r} in the Poincare disk")
    print("NOTE: both sides are now GEOMETRIC lengths -- this rung does not test")
    print("      the RT dictionary, only whether two geodesic length functions on")
    print("      the same angular lattice differ by a constant.\n")

    validate_formulas(n, r)

    # ---- 1/2. the matrices ----
    D_bd = boundary_distance_matrix(n, "specified")
    D_bk = bulk_distance_matrix(n, r)
    print(f"\n[1] BOUNDARY DISTANCE MATRIX (as specified)\n")
    print_matrix(D_bd, "d_boundary(i,j) = 2*log(1/sin(theta/2))")
    off = ~np.eye(n, dtype=bool)
    n_zero = int(np.sum(np.abs(D_bd[off]) < 1e-12))
    n_inf = int(np.sum(~np.isfinite(D_bd[off])))
    print(f"\n  zero off-diagonal entries : {n_zero} "
          f"(antipodal pairs -> distinct sites at distance 0,")
    print("                              so this is a pseudometric, not a metric)")
    print(f"  infinite entries          : {n_inf}")
    print(f"  distinct finite values    : "
          f"{sorted({round(float(v), 9) for v in D_bd[off] if np.isfinite(v)})}")

    print(f"\n[2] BULK DISTANCE MATRIX (Poincare disk, r = {r})\n")
    print_matrix(D_bk, "d_bulk(i,j) = arccosh(1 + 2*chord^2/((1-r1^2)(1-r2^2)))")
    print(f"\n  distinct finite values    : "
          f"{sorted({round(float(v), 9) for v in D_bk[off] if np.isfinite(v)})}")

    # ---- 3/4. barcodes and Wasserstein ----
    dgms_bd = barcodes(D_bd, args.maxdim)
    dgms_bk = barcodes(D_bk, args.maxdim)

    print(f"\n[3] BARCODES (ripser, maxdim = {args.maxdim})")
    print("  BOUNDARY side:")
    for dim, d in enumerate(dgms_bd):
        describe_barcode(d, f"H{dim}")
    print(f"  BULK side at phi = 1:")
    for dim, d in enumerate(dgms_bk):
        describe_barcode(d, f"H{dim}")

    phi_h0, w_h0_opt, note_h0 = optimal_phi(dgms_bd[0], dgms_bk[0], "H0")
    print(f"  BULK side at the optimal phi = {phi_h0:.6f} (fitted on H0):")
    for dim, d in enumerate(dgms_bk):
        describe_barcode(scale_diagram(d, phi_h0), f"H{dim}")

    print("\n[4] WASSERSTEIN DISTANCE (finite bars only)")
    rows = [("phi = 1 (unscaled)", 1.0),
            ("phi = 2 (rung 2's value)", 2.0),
            (f"phi = {phi_h0:.6f} (optimal, H0-fitted)", phi_h0)]
    print(f"  {'setting':>34} {'W_H0':>12} {'W_H1':>12}")
    for label, ph in rows:
        w0, n0 = wasserstein_safe(dgms_bd[0], scale_diagram(dgms_bk[0], ph))
        w1, n1 = wasserstein_safe(dgms_bd[1], scale_diagram(dgms_bk[1], ph))
        print(f"  {label:>34} {w0:>12.6f} {w1:>12.6f}"
              + (f"   [{n1}]" if n1 else ""))
    if note_h0:
        print(f"  note on the optimum: {note_h0}")

    # ---- 5. radius sweep ----
    print(f"\n[5] BULK RADIUS SWEEP: r from {args.r_min} to {args.r_max} in "
          f"{args.r_steps} steps")
    print("    (the boundary side does not depend on r, so only the bulk moves)")
    radii = np.linspace(args.r_min, args.r_max, args.r_steps)
    print(f"  {'r':>7} {'d_bulk adj':>11} {'d_bulk anti':>12} {'phi opt':>11} "
          f"{'W_H0(opt)':>11} {'W_H0(phi=1)':>12} {'W_H1(opt)':>11}")
    sweep = []
    for rr in radii:
        Dk = bulk_distance_matrix(n, rr)
        dk = barcodes(Dk, args.maxdim)
        ph, w0, _ = optimal_phi(dgms_bd[0], dk[0], "H0")
        w0_1, _ = wasserstein_safe(dgms_bd[0], dk[0])
        w1, _ = wasserstein_safe(dgms_bd[1], scale_diagram(dk[1], ph))
        sweep.append({"r": rr, "phi": ph, "w0": w0, "w0_1": w0_1, "w1": w1})
        print(f"  {rr:>7.4f} {Dk[0, 1]:>11.6f} {Dk[0, n // 2]:>12.6f} "
              f"{ph:>11.6f} {w0:>11.6f} {w0_1:>12.6f} {w1:>11.6f}")

    # ---- 6. H1 non-triviality ----
    print("\n[6] IS H1 NON-TRIVIAL ON EITHER SIDE?")
    for tag, dgms in (("boundary", dgms_bd), ("bulk", dgms_bk)):
        fin = finite_bars(dgms[1])
        print(f"  {tag:>8}: {len(fin)} finite H1 bar(s)"
              + (f", longest persistence {float(np.max(fin[:, 1] - fin[:, 0])):.6f}"
                 if len(fin) else ""))
    print("\n  CAVEAT -- this check cannot fail, so it carries no information.")
    print("  Both matrices are circulant (distance depends only on angular")
    print("  separation).  The repo's rung-4 diagnostics proved via")
    print("  Adamaszek-Adams that a circulant filtration with monotone weights")
    print("  forces exactly one H1 bar and H0 = n-1 identical bars.  A non-zero")
    print("  H1 here is a property of the construction, not a detection of the")
    print("  circle's pi_1.  Rung 4 was misled by exactly this before.")

    # ---- diagnostic track: correctly ordered boundary length ----
    print(f"\n[7] DIAGNOSTIC TRACK: correctly ordered boundary geodesic")
    print(f"    L = 2*log((2/eps)*sin(theta/2)), eps = {args.diag_eps}")
    print("    (NOT the specified formula -- run to show what the sign error costs)")
    D_bd2 = boundary_distance_matrix(n, "regularized", args.diag_eps)
    print()
    print_matrix(D_bd2, "d_boundary_regularized(i,j)")
    dgms_bd2 = barcodes(D_bd2, args.maxdim)
    print("\n  barcodes:")
    for dim, d in enumerate(dgms_bd2):
        describe_barcode(d, f"H{dim}")
    ph2, w02, _ = optimal_phi(dgms_bd2[0], dgms_bk[0], "H0")
    w12, _ = wasserstein_safe(dgms_bd2[1], scale_diagram(dgms_bk[1], ph2))
    w02_1, _ = wasserstein_safe(dgms_bd2[1], dgms_bk[1])
    print(f"\n  optimal phi (H0-fitted) : {ph2:.6f}")
    print(f"  W_H0 at that phi        : {w02:.6f}")
    print(f"  W_H1 at that phi        : {w12:.6f}")
    print(f"  monotone ordering matches the bulk, so the fit is meaningful here")
    print(f"  (compare the as-specified track above, where it cannot be).")

    # ---- plot ----
    png = HERE / "rtn_rung3_barcodes.png"
    plot_barcodes(dgms_bd, dgms_bk, phi_h0, png, n, r)
    print(f"\n  wrote {png}")

    # ---- verdict ----
    print("\n[8] VERDICT")
    print(f"  boundary d is anti-monotone in angular separation; bulk d is")
    print(f"  monotone.  Best-fit phi on H0 = {phi_h0:.6f}, leaving "
          f"W_H0 = {w_h0_opt:.6f}")
    print(f"  (W_H0 at phi = 1 is "
          f"{wasserstein_safe(dgms_bd[0], dgms_bk[0])[0]:.6f}).")
    print(f"  Zero-distance antipodal pairs: {n_zero} -> pseudometric, so the")
    print(f"  boundary filtration starts by merging opposite sides of the circle.")
    print("  The as-specified construction cannot be repaired by any phi; the")
    print("  diagnostic track shows the ordering the formula should have had.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
