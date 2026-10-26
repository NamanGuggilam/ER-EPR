#!/usr/bin/env python3
"""Rung 2 of the RTN rebuild: mutual-information distance instead of 1/S.

CHANGE FROM RUNG 1
------------------
Rung 1 used d(i,j) = 1 / S({i..j}), the entropy of the whole interval spanned
by i and j.  That failed structurally: on an open chain min_cut([i..j]) depends
only on whether the interval touches an end, never on |i-j|, so the boundary
matrix held just two distinct values against the bulk's five and no rescaling
could match them (see rtn_rung1_results.txt).

Rung 2 replaces it with the mutual information between the two SINGLE sites,

    d(i,j) = 1 / I(i:j),     I(i:j) = S(i) + S(j) - S({i,j})

which is the rungs 1-2 convention of the original ladder.  Every other
ingredient is held fixed: same 6-site open chain, same bulk graph geodesic,
same D sweep, same output format.

MIN CUTS ARE COMPUTED FROM THE GRAPH, NOT ASSUMED
-------------------------------------------------
min_cut(A) = the number of bonds with exactly one endpoint in A.  This is
evaluated directly for every region, which matters because two shortcut rules
stated in the rung-2 brief do not survive the check:

  * "S(i) = log(D) for a single site (1 cut bond)" holds only for the two END
    sites.  An interior site has a bond on each side, so min_cut({i}) = 2 and
    S(i) = 2 log(D).  Measured: {1:1, 2:2, 3:2, 4:2, 5:2, 6:1}.

  * "min_cut({i,j}) = 2 if not adjacent ... = 1 if adjacent or both at the
    same end" is wrong for most pairs.  Measured values run 1, 2, 3, and 4:
    two non-adjacent interior sites cut 4 bonds, not 2.

Both the graph-computed track (primary) and the brief's literal shortcut rules
(secondary, clearly labelled) are run, so the consequence of the difference is
visible rather than argued.

THE RESULT THIS PRODUCES
------------------------
With graph-computed cuts, I(i:j) = 2 log(D) for adjacent sites and EXACTLY
ZERO for every non-adjacent pair -- the min cut factorizes, which is the
standard leading-order RTN statement that single-site mutual information is
short-ranged.  So d = 1/(2 log D) on the nearest-neighbour band and +inf
everywhere else.  That is a real match on H0 after rescaling, but a narrow
one, and the script says so in its own verdict rather than leaving the reader
to infer it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Shared machinery is imported from rung 1 rather than copied, so the bulk
# side, the metric checks, the Wasserstein convention and the plotting cannot
# silently drift between the two rungs.
from rtn_rung1 import (DEFAULT_BOND_DIMS, barcodes, bulk_distance_matrix,
                       describe_barcode, finite_bars, measure_phi, plot_barcodes,
                       plot_sweep, print_matrix, rescale, validate_metric,
                       wasserstein_safe)

HERE = Path(__file__).resolve().parent


# ==========================================================================
# 1. MIN CUTS, COMPUTED FROM THE BOND GRAPH
# ==========================================================================

def min_cut_set(region, n_sites):
    """Bonds with exactly one endpoint in `region` (a set of 1-based sites).

    The bond graph of an open chain is {(k, k+1) : k = 1..n-1}, so this is the
    exact min cut for any region, contiguous or not.
    """
    region = set(region)
    if not region <= set(range(1, n_sites + 1)):
        raise ValueError(f"sites outside 1..{n_sites}: {region}")
    return sum(1 for k in range(1, n_sites)
               if (k in region) != (k + 1 in region))


def rt_entropy_set(region, n_sites, bond_dim):
    """S(A) = min_cut(A) * log(D) -- exact for an RTN (Hayden-Nezami-Qi et al.)."""
    return min_cut_set(region, n_sites) * np.log(bond_dim)


def mutual_information(i, j, n_sites, bond_dim, rule="graph"):
    """I(i:j) = S(i) + S(j) - S({i,j}).

    rule="graph" : every min cut computed from the bond graph (primary track).
    rule="brief" : the shortcut rules as literally stated in the rung-2 brief,
                   S(i) = log D for all i and min_cut({i,j}) = 1 if adjacent
                   or both at an end, else 2.  Kept only for comparison.
    """
    if i == j:
        raise ValueError("mutual information of a site with itself")
    logD = np.log(bond_dim)
    if rule == "graph":
        return (min_cut_set({i}, n_sites) + min_cut_set({j}, n_sites)
                - min_cut_set({i, j}, n_sites)) * logD
    if rule == "brief":
        lo, hi = min(i, j), max(i, j)
        adjacent = (hi == lo + 1)
        both_ends = (lo == 1 and hi == n_sites)
        cut_ij = 1 if (adjacent or both_ends) else 2
        return (2 - cut_ij) * logD          # S(i) = S(j) = log D by that rule
    raise ValueError(f"unknown rule {rule!r}")


# ==========================================================================
# 2. BOUNDARY DISTANCE MATRIX: d = 1 / I(i:j)
# ==========================================================================

def boundary_distance_matrix(n_sites, bond_dim, rule="graph"):
    """d_boundary(i,j) = 1 / I(i:j), d(i,i) = 0.

    Pairs with I = 0 become +inf, exactly as in rung 1: in a Vietoris-Rips
    filtration an infinite distance correctly means the edge is never born,
    whereas clipping to a large finite number would invent a birth scale the
    construction does not contain.  Here the +inf entries are not an edge case
    but the generic one -- all non-adjacent pairs have identically zero mutual
    information at this order.
    """
    D = np.zeros((n_sites, n_sites), dtype=float)
    n_zero_mi = 0
    for i in range(1, n_sites + 1):
        for j in range(1, n_sites + 1):
            if i == j:
                continue
            I = mutual_information(i, j, n_sites, bond_dim, rule=rule)
            if I <= 0.0:
                D[i - 1, j - 1] = np.inf
                n_zero_mi += 1
            else:
                D[i - 1, j - 1] = 1.0 / I
    return D, n_zero_mi


# ==========================================================================
# 3. ONE BOND DIMENSION
# ==========================================================================

def analyse(n_sites, bond_dim, maxdim=1, rule="graph"):
    """Full rung-2 analysis at one bond dimension.

    Returns the same dict shape rung 1 returns, so rung 1's plotting and
    reporting helpers apply unchanged.  Two rescalings are reported: the
    brief's phi = log D, and the phi actually measured from the data.
    """
    logD = float(np.log(bond_dim))
    D_bd, n_zero_mi = boundary_distance_matrix(n_sites, bond_dim, rule=rule)
    D_bk = bulk_distance_matrix(n_sites)

    dgms_bd = barcodes(D_bd, maxdim)
    dgms_bk = barcodes(D_bk, maxdim)

    phi = measure_phi(D_bd, D_bk)
    # The measured factor: if a single constant carries one side onto the
    # other, every usable pair returns it, so phi["min"] == phi["max"].
    phi_measured = phi["mean"]

    D_bd_logD = rescale(D_bd, logD)                 # the brief's prediction
    D_bd_meas = rescale(D_bd, phi_measured)         # what the data says
    dgms_logD = barcodes(D_bd_logD, maxdim)
    dgms_meas = barcodes(D_bd_meas, maxdim)

    w_raw, w_logD, w_meas = {}, {}, {}
    notes_raw, notes_logD, notes_meas = {}, {}, {}
    for dim in range(maxdim + 1):
        w_raw[dim], notes_raw[dim] = wasserstein_safe(dgms_bd[dim], dgms_bk[dim])
        w_logD[dim], notes_logD[dim] = wasserstein_safe(dgms_logD[dim], dgms_bk[dim])
        w_meas[dim], notes_meas[dim] = wasserstein_safe(dgms_meas[dim], dgms_bk[dim])

    return {
        "bond_dim": bond_dim, "logD": logD, "rule": rule,
        "D_boundary": D_bd, "D_bulk": D_bk,
        "D_boundary_rescaled": D_bd_meas,     # key name kept for rung 1's plots
        "D_boundary_logD": D_bd_logD,
        "n_zero_mi": n_zero_mi,
        "dgms_boundary": dgms_bd, "dgms_bulk": dgms_bk,
        "dgms_boundary_rescaled": dgms_meas,
        "dgms_boundary_logD": dgms_logD,
        "phi": phi, "phi_measured": phi_measured,
        "phi_over_logD": phi_measured / logD,
        "w_raw": w_raw, "w_raw_notes": notes_raw,
        "w_logD": w_logD, "w_logD_notes": notes_logD,
        "w_scaled": w_meas, "w_scaled_notes": notes_meas,
    }


# ==========================================================================
# 4. REPORT
# ==========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sites", type=int, default=6)
    p.add_argument("--bond-dim", type=int, default=4,
                   help="bond dimension for the detailed report")
    p.add_argument("--sweep", type=int, nargs="+", default=DEFAULT_BOND_DIMS)
    p.add_argument("--maxdim", type=int, default=1)
    args = p.parse_args(argv)
    n = args.n_sites

    print("=" * 74)
    print("RTN RUNG 2 -- mutual-information distance, d(i,j) = 1 / I(i:j)")
    print("=" * 74)
    print(f"chain: {n} boundary sites, {n - 1} bonds, open (not periodic)")
    print(f"detailed report at D = {args.bond_dim}; sweep over D = {args.sweep}")
    print("I(i:j) = S(i) + S(j) - S({i,j}),  S(A) = min_cut(A) * log(D)")
    print("CHANGE FROM RUNG 1: interval entropy 1/S -> single-site mutual info 1/I")

    # ---- min cuts, measured against the brief's shortcut rules ----
    print("\n[0] MIN CUTS COMPUTED FROM THE BOND GRAPH (vs the brief's rules)")
    singles = {i: min_cut_set({i}, n) for i in range(1, n + 1)}
    print(f"  single-site min_cut : {singles}")
    print(f"    the brief states 1 for every site; it holds only for the two "
          f"end sites,\n    because an interior site has a bond on each side.")
    print("\n  two-site regions:")
    print(f"  {'pair':>8} {'c_i':>4} {'c_j':>4} {'c_ij':>5} {'I/logD':>8} "
          f"{'brief c_ij':>11} {'brief I/logD':>13}")
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            ci, cj = min_cut_set({i}, n), min_cut_set({j}, n)
            cij = min_cut_set({i, j}, n)
            adj, both_ends = (j == i + 1), (i == 1 and j == n)
            bc = 1 if (adj or both_ends) else 2
            print(f"  {f'({i},{j})':>8} {ci:>4} {cj:>4} {cij:>5} "
                  f"{ci + cj - cij:>+8d} {bc:>11} {2 - bc:>+13d}")
    print("\n  Consequence: with graph-computed cuts I(i:j) = 2 log D for "
          "adjacent sites\n  and EXACTLY 0 for every non-adjacent pair -- the "
          "min cut factorizes.\n  The brief's rules instead give log D for "
          "adjacent pairs and a spurious\n  non-zero value for the end pair "
          f"(1,{n}).")

    # ---- 1. matrices ----
    res = analyse(n, args.bond_dim, args.maxdim, rule="graph")
    logD = res["logD"]
    print(f"\n[1] DISTANCE MATRICES at D = {args.bond_dim} (log D = {logD:.6f})\n")
    print_matrix(res["D_boundary"], "D_boundary(i,j) = 1 / I(i:j)", n)
    print(f"\n  pairs with I = 0 -> d = inf : {res['n_zero_mi']} of "
          f"{n * (n - 1)} ordered pairs")
    print()
    print_matrix(res["D_bulk"], "D_bulk(i,j) = |pos(b_i) - pos(b_j)|  (unchanged)", n)

    # ---- 2. metric validation ----
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
    print(f"  BOUNDARY rescaled by the brief's phi = log D = {logD:.6f}:")
    for dim, dgm in enumerate(res["dgms_boundary_logD"]):
        describe_barcode(dgm, f"H{dim}")
    print(f"  BOUNDARY rescaled by the MEASURED phi = {res['phi_measured']:.6f}:")
    for dim, dgm in enumerate(res["dgms_boundary_rescaled"]):
        describe_barcode(dgm, f"H{dim}")

    # ---- 4. Wasserstein ----
    print("\n[4] WASSERSTEIN DISTANCE (finite bars only; infinite bars dropped)")
    for dim in range(args.maxdim + 1):
        print(f"  H{dim}: raw                  W = {res['w_raw'][dim]:.6f}"
              f"   {res['w_raw_notes'][dim]}")
        print(f"      after x log D        W = {res['w_logD'][dim]:.6f}"
              f"   {res['w_logD_notes'][dim]}")
        print(f"      after x measured phi W = {res['w_scaled'][dim]:.6f}"
              f"   {res['w_scaled_notes'][dim]}")

    # ---- 5. identical? ----
    print("\n[5] ARE THE BARCODES IDENTICAL?")
    for tag, dgms in (("raw", res["dgms_boundary"]),
                      ("x log D", res["dgms_boundary_logD"]),
                      ("x measured phi", res["dgms_boundary_rescaled"])):
        same = all(
            np.allclose(np.sort(finite_bars(a), axis=0),
                        np.sort(finite_bars(b), axis=0), rtol=1e-9, atol=1e-12)
            if len(finite_bars(a)) == len(finite_bars(b)) else False
            for a, b in zip(dgms, res["dgms_bulk"]))
        print(f"  boundary ({tag}) vs bulk: {'IDENTICAL' if same else 'NOT identical'}")

    # ---- 6. phi ----
    print("\n[6] THE MAP phi: measured pair by pair as d_bulk / d_boundary")
    ph = res["phi"]
    print_matrix(np.nan_to_num(ph["ratios"], nan=0.0),
                 "phi(i,j)  (0 = pair skipped: infinite or zero distance)", n)
    print(f"\n  usable pairs          : {ph['n_usable']} of {n * (n - 1)} ordered")
    print(f"  skipped pairs (I = 0) : {ph['n_skipped']}")
    print(f"  phi range over pairs  : [{ph['min']:.6f}, {ph['max']:.6f}]")
    print(f"  phi mean +- sd        : {ph['mean']:.6f} +- {ph['std']:.6f}")
    print(f"  single constant phi?  : {'YES' if ph['consistent'] else 'NO'}")
    print(f"  brief's prediction    : phi = log D       = {logD:.6f}")
    print(f"  measured              : phi               = {res['phi_measured']:.6f}")
    print(f"  ratio measured/log D  : {res['phi_over_logD']:.6f}"
          f"   (2 means phi = 2 log D, from I = 2 log D on adjacent sites)")

    # ---- 7. sweep ----
    print(f"\n[7] BOND-DIMENSION SWEEP over D = {args.sweep}")
    sweep = [analyse(n, D, args.maxdim, rule="graph") for D in args.sweep]
    print(f"  {'D':>4} {'log D':>9} {'phi meas':>9} {'phi/logD':>9} "
          f"{'W_H0 raw':>10} {'W_H0 xlogD':>11} {'W_H0 xphi':>10} "
          f"{'W_H1 xphi':>10} {'const?':>7}")
    for r in sweep:
        print(f"  {r['bond_dim']:>4d} {r['logD']:>9.6f} "
              f"{r['phi_measured']:>9.6f} {r['phi_over_logD']:>9.6f} "
              f"{r['w_raw'][0]:>10.6f} {r['w_logD'][0]:>11.6f} "
              f"{r['w_scaled'][0]:>10.6f} {r['w_scaled'][1]:>10.6f} "
              f"{'YES' if r['phi']['consistent'] else 'NO':>7}")

    # ---- the brief's literal rules, for comparison ----
    print("\n[7b] SAME SWEEP UNDER THE BRIEF'S LITERAL SHORTCUT RULES")
    print("     (secondary track: S(i) = log D for all i, min_cut({i,j}) = 1 if")
    print("      adjacent or both at an end, else 2 -- shown only to expose the")
    print("      difference; the graph-computed track above is the primary one)")
    sweep_brief = [analyse(n, D, args.maxdim, rule="brief") for D in args.sweep]
    print(f"  {'D':>4} {'log D':>9} {'phi meas':>9} {'phi/logD':>9} "
          f"{'W_H0 xlogD':>11} {'W_H0 xphi':>10} {'const?':>7}")
    for r in sweep_brief:
        print(f"  {r['bond_dim']:>4d} {r['logD']:>9.6f} "
              f"{r['phi_measured']:>9.6f} {r['phi_over_logD']:>9.6f} "
              f"{r['w_logD'][0]:>11.6f} {r['w_scaled'][0]:>10.6f} "
              f"{'YES' if r['phi']['consistent'] else 'NO':>7}")

    # ---- plots ----
    bar_png = HERE / "rtn_rung2_barcodes.png"
    sweep_png = HERE / "rtn_rung2_sweep.png"
    plot_barcodes(res, bar_png, n)
    plot_sweep(sweep, sweep_png)
    print(f"\n  wrote {bar_png}")
    print(f"  wrote {sweep_png}")

    # ---- 8. verdict ----
    print("\n[8] VERDICT")
    all_const = all(r["phi"]["consistent"] for r in sweep)
    logd_zero = all(abs(r["w_logD"][0]) < 1e-9 for r in sweep)
    meas_zero = all(abs(r["w_scaled"][0]) < 1e-9 for r in sweep)
    meas_h1_zero = all(abs(r["w_scaled"][1]) < 1e-9 for r in sweep)
    ratios = sorted({round(r["phi_over_logD"], 10) for r in sweep})
    print(f"  Is a single constant phi available?            "
          f"{'YES' if all_const else 'NO'}")
    print(f"  Does the brief's phi = log D zero W_H0?        "
          f"{'YES' if logd_zero else 'NO'}")
    print(f"  Does the measured phi zero W_H0?               "
          f"{'YES' if meas_zero else 'NO'}")
    print(f"  Does the measured phi zero W_H1?               "
          f"{'YES' if meas_h1_zero else 'NO'}")
    print(f"  measured phi / log D across the whole sweep:   {ratios}")
    print("\n  READ THIS BEFORE CALLING IT A SUCCESS:")
    print("  Non-adjacent single-site mutual information is identically zero at")
    print("  this order, so the boundary matrix is finite ONLY on the")
    print("  nearest-neighbour band and carries no information about longer")
    print("  separations.  H0 of a Vietoris-Rips filtration depends only on the")
    print("  minimum spanning tree, and both sides' trees are uniform paths, so")
    print("  an H0 match tests exactly one number per side -- the neighbour")
    print("  spacing -- and is blind to the bulk distances 2,3,4,5 entirely.")
    print("  H1 is empty on both sides, so W_H1 = 0 carries no information")
    print("  either (the same trap rung 4 documented).")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
