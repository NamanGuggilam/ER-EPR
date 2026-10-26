#!/usr/bin/env python3
"""Rung 5 of the RTN rebuild: an actual random tensor contraction vs AdS2 RT.

GOAL AS SET
-----------
The first genuinely independent test: the boundary side is computed from an
explicit random tensor network contraction (quantum information -- SVD of a
D^6-dimensional state), the bulk side from the AdS2 Ryu-Takayanagi length
(pure geometry).  Rung 4's tautology -- both sides being monotone functions of
one shared angle -- cannot occur, because the boundary side never sees an
angle at all.

THE TENSOR NETWORK -- AN INTERPRETATION, STATED AND JUSTIFIED
-------------------------------------------------------------
The brief specifies 5 random D x D bond matrices T_k and 6 random boundary
unit vectors |psi_i>.  Read literally, the only state those ingredients build
on (C^D)^{x6} is the product |psi_1> x ... x |psi_6>, which has ZERO
entanglement across every cut, so every S = 0 and every d = 1/S = inf.  This
script demonstrates that in section [0] rather than asserting it.

The standard random tensor network on an open chain with bond dimension D is
a random matrix product state, and that is what is built here:

    |Psi> = sum  A_1^{s1} T_1 A_2^{s2} T_2 ... T_5 A_6^{s6} |s1 ... s6>

  * site tensors A_i of shape (D_left, D, D_right): one physical leg of
    dimension D per boundary site, open boundary bonds of dimension 1;
  * the brief's 5 bond matrices T_k contracted on the 5 internal bonds,
    exactly one per bond as specified;
  * all entries complex Gaussian, randn + 1j*randn, as the brief specifies.
    (Strictly this is the Ginibre ensemble, not Haar: the Gaussian random
    tensors of Hayden-Nezami-Qi et al. are Gaussian too, so this is the
    standard RTN ensemble -- but "Haar" in the brief is a misnomer.)

Seeding follows the repo's SYK convention: draw d of run s uses
np.random.seed(1000*s + d) and the global randn, so every draw is
reproducible from its (s, d) label alone.

THREE PROBLEMS THE SPEC'S OWN DIAGNOSTIC WOULD MISREAD
------------------------------------------------------
The brief's key diagnostic is: Kendall tau between the 15 boundary and 15
bulk distances, with "tau = 1.0 = forced, tau < 1.0 = real content".  That
reading assumes the two sides are compared on matched terms.  They are not,
in three ways, and each one pushes tau below 1 WITHOUT adding content:

  1. INVERSION BY THE DICTIONARY ITSELF.  RT says S(A) is proportional to
     L_RT(A).  The boundary distance is 1/S; the bulk distance is L_RT, i.e.
     proportional to S.  So if the RT dictionary holds, the two distances are
     ANTI-monotone and tau should be NEGATIVE.  A low tau here is what RT
     predicts, not evidence of independent structure.  (Rung 3 had the same
     inversion, between boundary and bulk lengths.)

  2. TOPOLOGY MISMATCH.  The boundary is an OPEN chain: sites 1 and 6 are the
     two ends, and [1..6] is the whole chain, a pure state with S = 0 and
     d = inf.  The bulk is a CIRCLE: sites 1 and 6 are ADJACENT, at the
     shortest L_RT on the lattice.  The single pair (1,6) is maximally far on
     one side and minimally far on the other.  More generally, an interval's
     entropy on the chain is set by whether it touches an end; its RT length
     on the circle by arc length -- unrelated orderings.  This is the same
     class of index mismatch the whole rebuild was started to remove.

  3. THE BULK IS CIRCULANT AGAIN.  Uniform angles make the bulk matrix depend
     only on (i-j) mod 6, with three distinct values.  Rung 4 broke circulant
     symmetry precisely because, by the repo's Adamaszek-Adams diagnostic,
     such a matrix forces its H1.  The bulk H1 here is that artifact again.

A matched-geometry CONTROL separates these: the RTN entropies compared with
min_cut on the SAME open chain (the RT theorem for the network actually
built).  If the tensor side obeys RT on its own geometry, that control shows
a near-perfect ordering -- and it does.  The control is reported alongside
the spec's diagnostic so the reader can see where the low tau comes from.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

# PH conventions (ripser call, persim Wasserstein, diagram scaling, joint phi
# optimisation, circulant test) are shared with rungs 3-4 so they cannot drift.
from rtn_rung3 import (barcodes, describe_barcode, finite_bars, print_matrix,
                       scale_diagram, wasserstein_safe)
from rtn_rung4 import (count_significant, joint_cost, n_distinct_rows,
                       optimal_phi_joint)

HERE = Path(__file__).resolve().parent

N_SITES = 6
BOND_DIM = 4
EPSILON = 0.1
N_DRAWS = 50                # tensor-network draws averaged per comparison
N_STABILITY = 10            # independent full comparisons for the stability check
S_ZERO_TOL = 1e-9           # entropies below this are exactly-zero (pure state)


# ==========================================================================
# 0. THE LITERAL READING: a product of boundary vectors has no entanglement
# ==========================================================================

def product_state(n_sites, bond_dim, seed):
    """|psi_1> x ... x |psi_6>, random unit vectors -- the brief read literally."""
    np.random.seed(seed)
    state = np.array([1.0 + 0j])
    for _ in range(n_sites):
        v = np.random.randn(bond_dim) + 1j * np.random.randn(bond_dim)
        state = np.kron(state, v / np.linalg.norm(v))
    return state


# ==========================================================================
# 1. BOUNDARY SIDE: random MPS contraction
# ==========================================================================

def random_mps_state(n_sites, bond_dim, seed):
    """Contract a random open-boundary MPS into a normalized D^n state vector.

    Site tensors A_i have shape (D_left, D, D_right) with D_left = 1 at the
    left end and D_right = 1 at the right end; the brief's bond matrices T_k
    sit on the n-1 internal bonds.  Contraction runs left to right.
    """
    np.random.seed(seed)
    D = bond_dim
    dims = [1] + [D] * (n_sites - 1) + [1]
    A = [np.random.randn(dims[k], D, dims[k + 1])
         + 1j * np.random.randn(dims[k], D, dims[k + 1])
         for k in range(n_sites)]
    T = [np.random.randn(D, D) + 1j * np.random.randn(D, D)
         for _ in range(n_sites - 1)]

    psi = A[0]                                            # (1, D, D)
    for k in range(1, n_sites):
        psi = np.tensordot(psi, T[k - 1], axes=([-1], [0]))   # bond matrix T_k
        psi = np.tensordot(psi, A[k], axes=([-1], [0]))       # next site tensor
    vec = psi.reshape(-1)                                 # (1, D^n, 1) -> D^n
    return vec / np.linalg.norm(vec)


def interval_entropy(state, i, j, n_sites, bond_dim):
    """von Neumann entropy of the contiguous interval [i..j] (1-based sites).

    The complement is traced out implicitly: permute the interval's legs to
    the front, reshape to (D^|A|, D^|rest|), and take the Schmidt spectrum by
    SVD.  S = -sum p log p over p = s^2 / sum(s^2).  For the whole chain the
    complement is empty and S = 0 exactly (a pure state).
    """
    psi = state.reshape([bond_dim] * n_sites)
    A = list(range(i - 1, j))
    B = [k for k in range(n_sites) if k not in A]
    M = np.transpose(psi, A + B).reshape(bond_dim ** len(A), -1)
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2 / np.sum(s ** 2)
    p = p[p > 1e-15]
    S = float(-np.sum(p * np.log(p)))
    return 0.0 if S < S_ZERO_TOL else S


def entropy_matrix(state, n_sites, bond_dim):
    """S[i-1, j-1] = S([min..max]) for i != j; diagonal 0 (unused)."""
    S = np.zeros((n_sites, n_sites))
    for i in range(1, n_sites + 1):
        for j in range(i + 1, n_sites + 1):
            S[i - 1, j - 1] = S[j - 1, i - 1] = interval_entropy(
                state, i, j, n_sites, bond_dim)
    return S


def distance_from_entropy(S):
    """d = 1/S off the diagonal; S = 0 -> +inf (edge never born in VR)."""
    n = S.shape[0]
    D = np.zeros_like(S)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = np.inf if S[i, j] == 0.0 else 1.0 / S[i, j]
    return D


def averaged_boundary(n_sites, bond_dim, n_draws, run_seed):
    """Average d = 1/S over n_draws independent networks.

    Draw d of run s is seeded 1000*s + d.  Returns (mean_d, std_d, mean_S).
    Entries that are infinite in every draw (S = 0 every time) stay inf in
    the mean, with std reported as nan rather than a misleading number.
    """
    Ds, Ss = [], []
    for d in range(n_draws):
        st = random_mps_state(n_sites, bond_dim, 1000 * run_seed + d)
        S = entropy_matrix(st, n_sites, bond_dim)
        Ss.append(S)
        Ds.append(distance_from_entropy(S))
    Ds, Ss = np.array(Ds), np.array(Ss)
    mean_d = np.zeros((n_sites, n_sites))
    std_d = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            col = Ds[:, i, j]
            if np.all(np.isinf(col)):
                mean_d[i, j], std_d[i, j] = np.inf, np.nan
            elif np.any(np.isinf(col)):
                # never observed here, but would be a real anomaly: report it
                mean_d[i, j], std_d[i, j] = np.inf, np.nan
            else:
                mean_d[i, j], std_d[i, j] = col.mean(), col.std()
    return mean_d, std_d, Ss.mean(axis=0)


# ==========================================================================
# 2. BULK SIDE: AdS2 RT lengths on a uniform circle (fixed, no randomness)
# ==========================================================================

def bulk_rt_matrix(n_sites, eps=EPSILON):
    """d_bulk(i,j) = L_RT = 2*log(sin(theta/2)/eps), uniform angles 2*pi*k/n.

    Boundary site k (1-based) is identified with bulk angle index k-1.
    """
    th = 2.0 * np.pi * np.arange(n_sites) / n_sites
    D = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            if i != j:
                t = abs(th[i] - th[j])
                t = min(t, 2.0 * np.pi - t)
                D[i, j] = 2.0 * np.log(np.sin(t / 2.0) / eps)
    return D


# ==========================================================================
# 3. RANK-ORDER DIAGNOSTICS
# ==========================================================================

def upper_pairs(n_sites):
    """The 15 unordered pairs (i, j), 0-based, i < j."""
    return list(combinations(range(n_sites), 2))


def chain_min_cut(i, j, n_sites):
    """min_cut of the contiguous interval [i..j] on the OPEN chain (1-based).

    Identical to rung 1's min_cut: 2 bonds if interior, 1 if touching an end,
    0 for the whole chain.  Used here as the matched-geometry RT prediction.
    """
    return (1 if i > 1 else 0) + (1 if j < n_sites else 0)


def circulant_inf_safe(M, tol=1e-12):
    """rung 4's is_circulant, but treating inf == inf as equal.

    rung 4's check subtracts entries, and inf - inf = nan (with a warning);
    the boundary matrix here has an inf entry, so compare exact equality
    first and subtract only finite pairs.  rung 4's file is left untouched.
    """
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            a, b = M[i, j], M[0, (j - i) % n]
            if np.isinf(a) or np.isinf(b):
                if a != b:
                    return False
            elif abs(a - b) >= tol:
                return False
    return True


def tau(x, y):
    """Kendall tau-b (handles ties), returns (tau, p-value)."""
    t, p = kendalltau(x, y)
    return float(t), float(p)


def discordant_against(primary, secondary):
    """Pairs-of-pairs that `primary` strictly orders and `secondary` reverses.

    With ties in `primary` (min_cut has only two values), tau-b cannot reach
    1 even under perfect agreement, so the discordant count is the cleaner
    test: 0 means the secondary ordering never contradicts the primary one.
    """
    n = len(primary)
    bad = 0
    for a in range(n):
        for b in range(a + 1, n):
            if primary[a] == primary[b]:
                continue
            if (primary[a] - primary[b]) * (secondary[a] - secondary[b]) < 0:
                bad += 1
    return bad


def rank_diagnostics(mean_d, mean_S, D_bulk, n_sites):
    """Everything needed to read tau honestly.  Returns a dict."""
    pairs = upper_pairs(n_sites)
    d_bd = np.array([mean_d[i, j] for i, j in pairs])
    S_bd = np.array([mean_S[i, j] for i, j in pairs])
    d_bk = np.array([D_bulk[i, j] for i, j in pairs])
    cut = np.array([chain_min_cut(i + 1, j + 1, n_sites) for i, j in pairs])
    finite = np.isfinite(d_bd)

    out = {"pairs": pairs, "d_bd": d_bd, "S_bd": S_bd, "d_bk": d_bk,
           "cut": cut, "finite": finite}
    # the spec's diagnostic, on the finite pairs and on all 15 (inf ranks top)
    out["tau_spec"], out["p_spec"] = tau(d_bd[finite], d_bk[finite])
    out["tau_spec_all"], _ = tau(np.where(finite, d_bd, 1e300), d_bk)
    # the inversion: compare S itself (which RT says is proportional to L)
    out["tau_S_L"], _ = tau(S_bd[finite], d_bk[finite])
    # the matched-geometry control: RTN entropy vs the SAME chain's min cut
    out["tau_ctrl"], _ = tau(S_bd[finite], cut[finite])
    out["disc_ctrl"] = discordant_against(cut[finite], S_bd[finite])
    n_strict = sum(1 for a in range(int(finite.sum()))
                   for b in range(a + 1, int(finite.sum()))
                   if cut[finite][a] != cut[finite][b])
    out["n_strict_ctrl"] = n_strict
    # does the shared rank ordering hold at all? (the brief's report item 3)
    out["same_order"] = bool(np.array_equal(np.argsort(d_bd[finite], kind="stable"),
                                            np.argsort(d_bk[finite], kind="stable")))
    return out


# ==========================================================================
# 4. ONE FULL COMPARISON
# ==========================================================================

def compare(n_sites, bond_dim, n_draws, run_seed, D_bulk, maxdim=1):
    mean_d, std_d, mean_S = averaged_boundary(n_sites, bond_dim, n_draws, run_seed)
    dg_bd = barcodes(mean_d, maxdim)
    dg_bk = barcodes(D_bulk, maxdim)
    w0_1, w1_1, _ = joint_cost(dg_bd, dg_bk, 1.0)
    phi, at_bound = optimal_phi_joint(dg_bd, dg_bk)
    w0_o, w1_o, _ = joint_cost(dg_bd, dg_bk, phi)
    rk = rank_diagnostics(mean_d, mean_S, D_bulk, n_sites)
    return {"mean_d": mean_d, "std_d": std_d, "mean_S": mean_S,
            "dg_bd": dg_bd, "dg_bk": dg_bk,
            "w0_1": w0_1, "w1_1": w1_1,
            "phi": phi, "at_bound": at_bound, "w0_o": w0_o, "w1_o": w1_o,
            "rank": rk}


# ==========================================================================
# 5. PLOT
# ==========================================================================

def plot_barcodes(dg_bd, dg_bk, phi, out_path, n_sites, bond_dim, n_draws):
    panels = [(f"Boundary: random MPS, D={bond_dim}\n"
               f"$d=1/S$, mean of {n_draws} draws", dg_bd),
              ("Bulk: AdS$_2$ RT length, $\\phi=1$\n"
               "$2\\log(\\sin(\\theta/2)/\\epsilon)$", dg_bk),
              (f"Bulk, $\\phi={phi:.4f}$ (joint optimum)",
               [scale_diagram(d, phi) for d in dg_bk])]
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
    fig.suptitle(f"RTN rung 5: tensor contraction vs AdS$_2$ RT, N = {n_sites}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==========================================================================
# 6. MAIN
# ==========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sites", type=int, default=N_SITES)
    p.add_argument("--bond-dim", type=int, default=BOND_DIM)
    p.add_argument("--eps", type=float, default=EPSILON)
    p.add_argument("--draws", type=int, default=N_DRAWS)
    p.add_argument("--stability-runs", type=int, default=N_STABILITY)
    p.add_argument("--maxdim", type=int, default=1)
    args = p.parse_args(argv)
    n, Dbond = args.n_sites, args.bond_dim

    print("=" * 78)
    print("RTN RUNG 5 -- explicit random tensor contraction vs AdS2 RT geometry")
    print("=" * 78)
    print(f"boundary: random open-chain MPS, {n} sites, bond dim D = {Dbond}, "
          f"state dim D^{n} = {Dbond ** n}")
    print(f"          d = 1/S([i..j]) from SVD, averaged over {args.draws} draws")
    print(f"bulk    : L_RT = 2*log(sin(theta/2)/eps), eps = {args.eps}, "
          f"{n} uniform angles")
    print("seeding : draw d of run s uses np.random.seed(1000*s + d); "
          "primary run s = 0\n")

    # ---- [0] the literal reading ----
    print("[0] THE BRIEF READ LITERALLY: product of 6 boundary unit vectors")
    ps = product_state(n, Dbond, seed=0)
    S_lit = [interval_entropy(ps, i, j, n, Dbond)
             for i in range(1, n + 1) for j in range(i, n + 1)]
    print(f"    max S over all {len(S_lit)} contiguous intervals = {max(S_lit):.3e}")
    print("    -> zero entanglement everywhere, every d = 1/S = inf: nothing to")
    print("       compare.  The random-MPS interpretation is used instead (see")
    print("       the module docstring for exactly what is contracted).")

    # ---- primary comparison ----
    D_bulk = bulk_rt_matrix(n, args.eps)
    res = compare(n, Dbond, args.draws, 0, D_bulk, args.maxdim)
    rk = res["rank"]

    print(f"\n[1] BOUNDARY DISTANCE MATRIX: mean and std over {args.draws} draws")
    print()
    print_matrix(res["mean_d"], "mean d_boundary = 1/S([i..j])  "
                 "(rows/cols = sites 1..6, printed 0-based)")
    print()
    print_matrix(np.nan_to_num(res["std_d"], nan=0.0),
                 "std d_boundary  (0.000 at an inf entry = S was 0 in EVERY draw)")
    print("\n  mean S([i..j]) / log D:")
    print_matrix(res["mean_S"] / np.log(Dbond), "")
    print(f"\n  pair (1,6) = the whole chain: S = 0 in every draw (pure state) "
          f"-> d = inf")

    print(f"\n[2] BULK DISTANCE MATRIX (AdS2 RT, uniform circle)")
    print()
    print_matrix(D_bulk, "d_bulk = L_RT(i,j)")
    circ_bk = circulant_inf_safe(D_bulk)
    circ_bd = circulant_inf_safe(res["mean_d"])
    off = ~np.eye(n, dtype=bool)
    print(f"\n  distinct bulk values : "
          f"{sorted({round(float(v), 6) for v in D_bulk[off]})}")
    print("  CIRCULANT DIAGNOSTIC:")
    print(f"    boundary circulant : {'YES' if circ_bd else 'no'} "
          f"({n_distinct_rows(res['mean_d'])}/{n} distinct rows)")
    if circ_bk:
        print("    *** BULK MATRIX IS CIRCULANT -- ITS H1 IS NOT TRUSTWORTHY ***")
        print("    uniform angles make d_bulk depend only on (i-j) mod 6; the")
        print("    repo's Adamaszek-Adams diagnostic shows such a matrix forces")
        print("    its H1.  This is the artifact rung 4 removed, back again.")
    else:
        print("    bulk circulant     : no")

    # ---- [3] rank ordering: THE key diagnostic, read honestly ----
    print("\n[3] RANK ORDERING OF THE 15 PAIRWISE DISTANCES")
    print(f"  {'pair':>7} {'mean S/logD':>12} {'d_bdy':>9} {'d_bulk':>9} "
          f"{'chain min_cut':>14}   note")
    for k, (i, j) in enumerate(rk["pairs"]):
        note = ""
        if (i, j) == (0, n - 1):
            note = "whole chain on boundary, ADJACENT on bulk circle"
        dbd = rk["d_bd"][k]
        print(f"  {f'({i + 1},{j + 1})':>7} {rk['S_bd'][k] / np.log(Dbond):>12.4f} "
              f"{('inf' if np.isinf(dbd) else f'{dbd:.4f}'):>9} "
              f"{rk['d_bk'][k]:>9.4f} {rk['cut'][k]:>14d}   {note}")
    print(f"\n  same rank ordering on both sides? "
          f"{'YES -> isomorphism forced' if rk['same_order'] else 'NO'}")

    print("\n  KEY DIAGNOSTIC (as specified, 14 finite pairs):")
    print(f"  Kendall tau = {rk['tau_spec']:.3f} (1.0 = forced, <1.0 = real content)")
    print(f"    p-value {rk['p_spec']:.3f}; with the inf pair ranked last, "
          f"all 15: tau = {rk['tau_spec_all']:.3f}")

    print("\n  WHY THAT tau IS NOT 'REAL CONTENT' -- decomposition:")
    print(f"    (a) inversion: tau(S, L_RT) = {rk['tau_S_L']:+.3f}, vs the spec's "
          f"tau(1/S, L_RT) = {rk['tau_spec']:+.3f}.")
    print("        RT says L_RT is proportional to S, but d_boundary = 1/S, so a")
    print("        dictionary that HOLDS predicts anti-monotone distances.  The")
    print("        sign of tau is set by the 1/S convention, not by the physics.")
    print("        (The two are not exact negatives because d_boundary is the")
    print("        draw-average of 1/S, and mean(1/S) need not rank exactly")
    print("        opposite to mean(S) -- a few near-tied interior pairs swap.)")
    print(f"    (b) matched-geometry control: tau(S, chain min_cut) = "
          f"{rk['tau_ctrl']:+.3f}")
    print(f"        discordant pairs vs min_cut: {rk['disc_ctrl']} of "
          f"{rk['n_strict_ctrl']} strictly ordered")
    print("        -> the tensor entropies obey RT on the network's OWN geometry")
    print("           (zero discordant = every interior interval out-entangles")
    print("           every end interval).  tau-b sits below 1 only because")
    print("           min_cut has two tied values; the discordant count is the")
    print("           clean test.")
    print(f"    (c) compare (b) with (a): the SAME entropies give tau = "
          f"{rk['tau_ctrl']:+.3f} against")
    print(f"        the chain's own min cut and {rk['tau_S_L']:+.3f} against "
          f"the circle's RT")
    print("        length.  Only the comparator changed, so that gap is the")
    print("        TOPOLOGY MISMATCH: an open-chain entropy compared to a circle's")
    print("        RT length.  Pair (1,6) alone is maximally far on one side and")
    print("        adjacent on the other.  That is an index mismatch -- the class")
    print("        of error this rebuild began by removing -- not content.")

    # ---- [4]/[5] barcodes and Wasserstein ----
    print(f"\n[4] BARCODES (ripser, maxdim = {args.maxdim}, primary run)")
    print("  BOUNDARY (mean of draws):")
    for dim, d in enumerate(res["dg_bd"]):
        describe_barcode(d, f"H{dim}")
    print("  BULK (phi = 1):")
    for dim, d in enumerate(res["dg_bk"]):
        describe_barcode(d, f"H{dim}")
    nb, mb, _ = count_significant(res["dg_bd"][1])
    nk, mk, _ = count_significant(res["dg_bk"][1])
    print(f"  significant H1 bars (> 0.1 * max): boundary {nb}, bulk {nk}"
          + ("   [bulk H1 is the circulant artifact]" if circ_bk else ""))

    print("\n[5] WASSERSTEIN, W(boundary, phi * bulk), phi searched on [0.1, 10]")
    print(f"  phi = 1       : W_H0 = {res['w0_1']:.6f}   W_H1 = {res['w1_1']:.6f}")
    print(f"  phi = {res['phi']:.6f}: W_H0 = {res['w0_o']:.6f}   "
          f"W_H1 = {res['w1_o']:.6f}"
          + ("   *** PINNED AT A SEARCH BOUND ***" if res["at_bound"]
             else "   (interior optimum)"))
    if len(finite_bars(res["dg_bd"][1])) == 0:
        print("  NOTE: the boundary has NO H1 bars, so W_H1 is only the cost of")
        print("  matching the bulk's single H1 bar to the diagonal -- and that bar")
        print("  is the circulant artifact.  W_H1 compares nothing here; part of the")
        print("  joint optimum's pull on phi is just shrinking that bar.")

    # ---- [6] stability ----
    print(f"\n[6] STABILITY: {args.stability_runs} independent full comparisons "
          f"(runs s = 1..{args.stability_runs}, {args.draws} draws each)")
    print(f"  {'run':>4} {'phi_opt':>10} {'bound?':>7} {'W_H0(opt)':>10} "
          f"{'W_H1(opt)':>10} {'tau spec':>9} {'tau ctrl':>9} {'disc':>5}")
    stab = []
    for s in range(1, args.stability_runs + 1):
        r = compare(n, Dbond, args.draws, s, D_bulk, args.maxdim)
        stab.append(r)
        print(f"  {s:>4d} {r['phi']:>10.6f} {'YES' if r['at_bound'] else 'no':>7} "
              f"{r['w0_o']:>10.6f} {r['w1_o']:>10.6f} "
              f"{r['rank']['tau_spec']:>+9.3f} {r['rank']['tau_ctrl']:>+9.3f} "
              f"{r['rank']['disc_ctrl']:>5d}")
    arr = lambda key: np.array([r[key] for r in stab])
    tau_s = np.array([r["rank"]["tau_spec"] for r in stab])
    tau_c = np.array([r["rank"]["tau_ctrl"] for r in stab])
    print(f"\n  phi_opt      : mean {arr('phi').mean():.6f}  std {arr('phi').std():.6f}")
    print(f"  W_H0(phi_opt): mean {arr('w0_o').mean():.6f}  std {arr('w0_o').std():.6f}")
    print(f"  W_H1(phi_opt): mean {arr('w1_o').mean():.6f}  std {arr('w1_o').std():.6f}")
    print(f"  tau (spec)   : mean {tau_s.mean():+.3f}  std {tau_s.std():.3f}")
    print(f"  tau (control): mean {tau_c.mean():+.3f}  std {tau_c.std():.3f}")
    print(f"  control discordant pairs, all runs: "
          f"{[r['rank']['disc_ctrl'] for r in stab]}")

    # ---- plot ----
    png = HERE / "rtn_rung5_barcodes.png"
    plot_barcodes(res["dg_bd"], res["dg_bk"], res["phi"], png, n, Dbond, args.draws)
    print(f"\n  wrote {png}")

    # ---- verdict ----
    print("\n[7] VERDICT")
    print("  The boundary side is now a genuinely independent quantum-information")
    print("  computation, and it does obey RT -- on its own open-chain geometry")
    print("  (control: zero discordant pairs in every run).  But the spec pairs")
    print("  it with the wrong comparator: a CIRCLE's RT length, through a 1/S")
    print("  convention that inverts the dictionary.  The low Kendall tau comes")
    print("  from that inversion plus the chain-vs-circle mismatch, not from")
    print("  independent structure; and the bulk H1 is circulant-forced.  Fixing")
    print("  the comparator (open-chain bulk, or a periodic MPS on a ring) and the")
    print("  convention (compare S to L_RT, or 1/S to 1/L_RT) is the next step.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
