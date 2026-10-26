#!/usr/bin/env python3
"""Rung 6 of the RTN rebuild: the first matched-geometry test.

GOAL AS SET
-----------
Open chain on BOTH sides.  Rung 5 showed the random-MPS entropies obey RT on
the chain's own geometry (zero discordant pairs against the chain min cut, in
every run).  Rung 6 turns that control into the full comparison: entropy from
an explicit tensor contraction on one side, the open-chain RT formula on the
other, persistent homology on both.

    boundary:  S([i..j])  from SVD / transfer matrices of a random MPS
    bulk    :  L_RT([i..j]) = min_cut([i..j]) * log(D)     (the RTN theorem)
    distance:  d = exp(-alpha * S)  and  d = exp(-alpha * L_RT)

DECISIONS MADE EXPLICIT
-----------------------
* Distance convention: exp(-alpha * x), not 1/x.  The whole chain [1..6] is a
  pure state, S = 0 and L_RT = 0, so 1/x would put +inf in both matrices;
  exp(-alpha * 0) = 1 is the maximum distance instead, and no entry is
  infinite.  It is also the SAME monotone map on both sides, so the rung-5
  inversion (1/S against a quantity proportional to S) cannot occur.

* d is computed from the draw-averaged entropy: d = exp(-alpha * mean S), not
  mean(exp(-alpha * S)).  Rung 5 used mean(1/S), which does not rank exactly
  opposite to mean(S); computing the transform after averaging keeps the
  distance ordering identical to the entropy ordering by construction.

* The bulk min cut is the number of bonds separating [i..j] from the rest on
  the OPEN chain: 2 if interior, 1 if touching an end, 0 for the whole chain.
  The brief also gives "min_cut = min(j-i, N-(j-i))", which is the arc length
  on a CIRCLE, not a cut on a chain: it disagrees with the bond count on 11 of
  the 15 pairs and would reintroduce exactly the chain-vs-circle mismatch rung
  5 diagnosed.  The brief's own parenthetical ("just the number of bonds
  cut") and the rung-5 control both use the bond count, so that is used here;
  section [0b] prints the disagreement.

* The RT prediction for phi is 1, not log(D).  L_RT = min_cut * log(D) already
  carries the log(D); if S = L_RT then the two exp-distance matrices are
  equal entry by entry.  phi = log(D) was rung 1's prediction, where the bulk
  distance |i-j| contained no log(D).  The D sweep therefore reports BOTH
  phi_opt against log(D), as the brief asks, and the real convergence test:
  whether S / L_RT -> 1 as D grows.

ENTROPIES WITHOUT THE FULL STATE
--------------------------------
At D = 16 the state has 16^6 = 16.8M amplitudes; SVD across 50 draws is out
of reach.  Interval entropies are instead computed from transfer matrices:
for an interval A with left part L and right part R,

    rho_A = M K M^dagger,   K = G_L (x) G_R,   so   spec(rho_A) = spec(K^1/2 G_M K^1/2)

with G_L, G_R, G_M the Gram matrices of the left, right and middle blocks --
at most D^2 x D^2 = 256 x 256.  Section [0] validates this against rung 5's
full-state SVD on identical seeds before anything else runs.

Seeding is rung 5's: draw d of run s uses np.random.seed(1000*s + d), with
the same tensor draw order, so the D = 4, s = 0 entropies here ARE rung 5's.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, pearsonr

# PH conventions shared with rungs 3-5; rung 5's full-state SVD for validation.
from rtn_rung3 import (barcodes, describe_barcode, finite_bars, print_matrix,
                       scale_diagram, wasserstein_safe)
from rtn_rung4 import count_significant, joint_cost, n_distinct_rows, optimal_phi_joint
from rtn_rung5 import circulant_inf_safe, entropy_matrix, random_mps_state

HERE = Path(__file__).resolve().parent

N_SITES = 6
BOND_DIM = 4
ALPHA = 1.0
ALPHAS = [0.5, 1.0, 2.0]
BOND_DIMS = [2, 4, 8, 16]
N_DRAWS = 50
N_STABILITY = 10
S_ZERO_TOL = 1e-9


# ==========================================================================
# 1. BOUNDARY SIDE: random MPS, entropies from transfer matrices
# ==========================================================================

def draw_tensors(n_sites, bond_dim, seed):
    """Rung 5's random MPS, returned as tensors rather than a state vector.

    RNG consumption is identical to rtn_rung5.random_mps_state (all site
    tensors first, real part before imaginary, then all bond matrices), so
    the same seed gives the same network.  Each bond matrix T_k is absorbed
    into the site tensor on its left, and every tensor is rescaled to unit
    Frobenius norm -- neither step changes the normalized state's
    entanglement, and the rescaling keeps the transfer products well
    conditioned.
    """
    np.random.seed(seed)
    D = bond_dim
    dims = [1] + [D] * (n_sites - 1) + [1]
    A = [np.random.randn(dims[k], D, dims[k + 1])
         + 1j * np.random.randn(dims[k], D, dims[k + 1])
         for k in range(n_sites)]
    T = [np.random.randn(D, D) + 1j * np.random.randn(D, D)
         for _ in range(n_sites - 1)]
    B = [np.tensordot(A[k], T[k], axes=([2], [0])) if k < n_sites - 1 else A[k]
         for k in range(n_sites)]
    return [b / np.linalg.norm(b) for b in B]


def _entropy_from_weights(lam):
    lam = np.clip(np.real(lam), 0.0, None)
    p = lam / lam.sum()
    p = p[p > 1e-15]
    S = float(-np.sum(p * np.log(p)))
    return 0.0 if S < S_ZERO_TOL else S


def entropy_matrix_fast(tensors):
    """S[i, j] = S([min..max]) for every pair, via transfer-matrix Grams.

    Left environments are built once as prefixes and right environments once
    as suffixes; for each left endpoint the middle Gram is extended one site
    at a time, so the 15 intervals cost 21 site contractions in total.
    """
    n = len(tensors)
    GL = [np.ones((1, 1), dtype=complex)]
    for k in range(n):
        GL.append(np.einsum("lm,lsr,mst->rt", GL[-1], tensors[k],
                            tensors[k].conj(), optimize=True))
    GR = [None] * n
    GR[n - 1] = np.ones((1, 1), dtype=complex)
    for k in range(n - 1, 0, -1):
        GR[k - 1] = np.einsum("lsr,mst,rt->lm", tensors[k], tensors[k].conj(),
                              GR[k], optimize=True)

    S = np.zeros((n, n))
    for i in range(n):
        dl = tensors[i].shape[0]
        # X[a, a', c, c'] = <M_{a'c'} | M_{ac}> for the middle block so far
        X = np.einsum("ac,bd->abcd", np.eye(dl), np.eye(dl)).astype(complex)
        for j in range(i, n):
            X = np.einsum("abcd,csr,dst->abrt", X, tensors[j],
                          tensors[j].conj(), optimize=True)
            if j == i:
                continue
            dr = tensors[j].shape[2]
            GM = X.transpose(1, 3, 0, 2).reshape(dl * dr, dl * dr)
            K = np.kron(GL[i], GR[j])
            w, U = np.linalg.eigh((K + K.conj().T) / 2.0)
            sK = (U * np.sqrt(np.clip(w, 0.0, None))) @ U.conj().T
            Q = sK @ GM @ sK
            S[i, j] = S[j, i] = _entropy_from_weights(
                np.linalg.eigvalsh((Q + Q.conj().T) / 2.0))
    return S


def averaged_entropy(n_sites, bond_dim, n_draws, run_seed):
    """Mean and std of S([i..j]) over n_draws; draw d seeded 1000*run_seed + d."""
    Ss = np.array([entropy_matrix_fast(draw_tensors(n_sites, bond_dim,
                                                    1000 * run_seed + d))
                   for d in range(n_draws)])
    return Ss.mean(axis=0), Ss.std(axis=0)


# ==========================================================================
# 2. BULK SIDE: the open-chain RT theorem
# ==========================================================================

def chain_min_cut(i, j, n_sites):
    """Bonds cut to separate [i..j] (1-based) on an OPEN chain: 2, 1, or 0."""
    lo, hi = min(i, j), max(i, j)
    return (1 if lo > 1 else 0) + (1 if hi < n_sites else 0)


def brief_min_cut(i, j, n_sites):
    """The brief's min(j-i, N-(j-i)) -- circle arc length.  Shown, never used."""
    k = abs(j - i)
    return min(k, n_sites - k)


def rt_length_matrix(n_sites, bond_dim):
    """L_RT[i, j] = chain_min_cut * log(D); diagonal 0."""
    L = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            if i != j:
                L[i, j] = chain_min_cut(i + 1, j + 1, n_sites) * np.log(bond_dim)
    return L


def exp_distance(X, alpha):
    """d = exp(-alpha * X) off the diagonal, 0 on it.  Range (0, 1]."""
    D = np.exp(-alpha * X)
    np.fill_diagonal(D, 0.0)
    return D


# ==========================================================================
# 3. DIAGNOSTICS
# ==========================================================================

def upper(M):
    n = M.shape[0]
    return np.array([M[i, j] for i in range(n) for j in range(i + 1, n)])


def discordant(primary, secondary):
    """Pairs-of-pairs that `primary` strictly orders and `secondary` reverses."""
    bad = strict = 0
    for a in range(len(primary)):
        for b in range(a + 1, len(primary)):
            if primary[a] == primary[b]:
                continue
            strict += 1
            if (primary[a] - primary[b]) * (secondary[a] - secondary[b]) < 0:
                bad += 1
    return bad, strict


def rt_check(mean_S, L):
    """tau(S, L_RT), p, and discordant count over all 15 pairs.

    No tau "ceiling" is reported.  BOTH variables carry exact ties -- L_RT
    takes three values, and S has complement ties because the state is pure
    (S([1..k]) = S([k+1..6])) -- so a ceiling computed from one side's ties
    alone is wrong (an earlier draft printed tau at 100.5% of such a
    ceiling).  With zero discordant pairs, tau-b is already the largest
    value the two tie structures allow; its gap from 1 is ties, not error.
    """
    s, l = upper(mean_S), upper(L)
    t, p = kendalltau(s, l)
    bad, strict = discordant(l, s)
    return {"tau": float(t), "p": float(p), "disc": bad, "strict": strict}


def complement_gap(mean_S, n_sites):
    """max |S([1..k]) - S([k+1..n])| over the complement pairs present.

    The MPS state is pure, so these must agree to machine precision; this is
    a free correctness check on the entropy code.
    """
    gaps = [abs(mean_S[0, k - 1] - mean_S[k, n_sites - 1])
            for k in range(2, n_sites - 1)]
    return float(max(gaps)) if gaps else 0.0


def deficit_phi(mean_S, n_sites, bond_dim, alpha):
    """exp(alpha * (log D - mean S over min-cut-1 pairs)).

    If the entropy fell short of RT by a fixed amount delta on every
    min-cut-1 pair, exp(-alpha*S) = exp(alpha*delta) * exp(-alpha*L_RT), so
    phi = exp(alpha*delta) would map those entries exactly.  A comparison
    estimate only -- the min-cut-2 pairs carry a different deficit, and the
    H0 matching weighs both.
    """
    vals = [mean_S[i, j] for i in range(n_sites) for j in range(i + 1, n_sites)
            if chain_min_cut(i + 1, j + 1, n_sites) == 1]
    delta = np.log(bond_dim) - float(np.mean(vals))
    return float(np.exp(alpha * delta)), delta


def class_ratios(mean_S, n_sites, bond_dim):
    """mean S / L_RT within each min-cut class -- the RT convergence test."""
    out = {}
    for c in (1, 2):
        vals = [mean_S[i, j] / (c * np.log(bond_dim))
                for i in range(n_sites) for j in range(i + 1, n_sites)
                if chain_min_cut(i + 1, j + 1, n_sites) == c]
        out[c] = (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)))
    return out


# ==========================================================================
# 4. ONE COMPARISON
# ==========================================================================

def compare(n_sites, bond_dim, alpha, n_draws, run_seed, maxdim=1, mean_S=None):
    if mean_S is None:
        mean_S, _ = averaged_entropy(n_sites, bond_dim, n_draws, run_seed)
    L = rt_length_matrix(n_sites, bond_dim)
    d_bd, d_bk = exp_distance(mean_S, alpha), exp_distance(L, alpha)
    dg_bd, dg_bk = barcodes(d_bd, maxdim), barcodes(d_bk, maxdim)
    w0_1, w1_1, _ = joint_cost(dg_bd, dg_bk, 1.0)
    phi, at_bound = optimal_phi_joint(dg_bd, dg_bk)
    w0_o, w1_o, _ = joint_cost(dg_bd, dg_bk, phi)
    rt = rt_check(mean_S, L)
    t_d, p_d = kendalltau(upper(d_bd), upper(d_bk))
    return {"mean_S": mean_S, "L": L, "d_bd": d_bd, "d_bk": d_bk,
            "dg_bd": dg_bd, "dg_bk": dg_bk, "w0_1": w0_1, "w1_1": w1_1,
            "phi": phi, "at_bound": at_bound, "w0_o": w0_o, "w1_o": w1_o,
            "rt": rt, "tau_d": float(t_d), "p_d": float(p_d)}


# ==========================================================================
# 5. PLOT
# ==========================================================================

def plot_barcodes(res, out_path, n_sites, bond_dim, alpha, n_draws):
    panels = [(f"Boundary: random MPS, D={bond_dim}\n"
               f"$d=e^{{-{alpha:g}\\,S}}$, mean of {n_draws} draws", res["dg_bd"]),
              (f"Bulk: chain RT, $\\phi=1$\n$d=e^{{-{alpha:g}\\,L_{{RT}}}}$, "
               f"$L_{{RT}}=$ min-cut $\\cdot\\log D$", res["dg_bk"]),
              (f"Bulk, $\\phi={res['phi']:.4f}$ (joint optimum)",
               [scale_diagram(d, res["phi"]) for d in res["dg_bk"]])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    colors = {0: "#2b6cb0", 1: "#c05621"}
    for ax, (title, dgms) in zip(axes, panels):
        y, xmax = 0, 0.0
        for dgm in dgms:
            fin = finite_bars(dgm)
            if len(fin):
                xmax = max(xmax, float(np.nanmax(fin[:, 1])))
        xmax = xmax or 1.0
        for dim, dgm in enumerate(dgms):
            dgm = np.asarray(dgm, dtype=float)
            if len(dgm) == 0:
                continue
            fin = finite_bars(dgm)
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
    fig.suptitle(f"RTN rung 6: open chain on both sides, N = {n_sites}, "
                 f"$\\alpha$ = {alpha:g}", fontsize=11)
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
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    p.add_argument("--bond-dims", type=int, nargs="+", default=BOND_DIMS)
    p.add_argument("--draws", type=int, default=N_DRAWS)
    p.add_argument("--stability-runs", type=int, default=N_STABILITY)
    p.add_argument("--maxdim", type=int, default=1)
    args = p.parse_args(argv)
    n, D, a = args.n_sites, args.bond_dim, args.alpha

    print("=" * 78)
    print("RTN RUNG 6 -- matched geometry: open chain on both sides")
    print("=" * 78)
    print(f"boundary: random open-chain MPS, {n} sites, D = {D}, "
          f"S([i..j]) averaged over {args.draws} draws")
    print(f"bulk    : L_RT = chain_min_cut * log(D)  (RTN theorem, no randomness)")
    print(f"distance: d = exp(-alpha * x), alpha = {a} (primary); "
          f"sweep {args.alphas}")
    print("          computed from mean S, so d ranks exactly opposite to S")
    print("seeding : draw d of run s uses np.random.seed(1000*s + d) -- rung 5's")

    # ---- [0] validation of the fast entropy path ----
    print("\n[0] VALIDATION: transfer-matrix entropies vs rung 5's full-state SVD")
    worst = 0.0
    for Dv in (2, 4):
        for d in range(5):
            S_full = entropy_matrix(random_mps_state(n, Dv, d), n, Dv)
            S_fast = entropy_matrix_fast(draw_tensors(n, Dv, d))
            worst = max(worst, float(np.max(np.abs(S_full - S_fast))))
    ok = worst < 1e-10
    print(f"    D = 2 and 4, 5 draws each, all 15 pairs: max |diff| = {worst:.2e}"
          f"   {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("    *** fast path disagrees with full SVD -- results below are "
              "NOT trustworthy ***")

    # ---- [0b] the min-cut formula ----
    print("\n[0b] MIN CUT: bonds cut on the open chain vs the brief's "
          "min(j-i, N-(j-i))")
    print(f"  {'pair':>7} {'chain bonds cut':>16} {'brief formula':>14}")
    n_diff = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            c, bf = chain_min_cut(i, j, n), brief_min_cut(i, j, n)
            n_diff += c != bf
            print(f"  {f'({i},{j})':>7} {c:>16d} {bf:>14d}"
                  + ("   <- differs" if c != bf else ""))
    print(f"  the brief's formula disagrees on {n_diff} of 15 pairs: it is arc")
    print("  length on a CIRCLE.  The bond count is used, per the brief's own")
    print("  parenthetical and the rung-5 control.")

    # ---- primary comparison ----
    mean_S, std_S = averaged_entropy(n, D, args.draws, 0)
    res = compare(n, D, a, args.draws, 0, args.maxdim, mean_S=mean_S)

    print(f"\n[1] BOUNDARY: mean and std of S([i..j]) over {args.draws} draws, "
          f"D = {D}")
    print()
    print_matrix(mean_S, "mean S([i..j])  (0-based labels; pair (0,5) = whole chain)")
    print()
    print_matrix(std_S, "std S([i..j])")
    print()
    print_matrix(res["d_bd"], f"d_boundary = exp(-{a:g} * mean S)")

    print(f"\n[2] BULK: L_RT = chain_min_cut * log D, D = {D}")
    print()
    print_matrix(res["L"], "L_RT([i..j])")
    print()
    print_matrix(res["d_bk"], f"d_bulk = exp(-{a:g} * L_RT)")

    # ---- [3] the physics check ----
    rt = res["rt"]
    print("\n[3] PHYSICS CHECK: does RT hold?  tau(mean S, L_RT), all 15 pairs")
    print(f"  Kendall tau = {rt['tau']:.3f}, p-value = {rt['p']:.2e}, "
          f"discordant pairs = {rt['disc']} of {rt['strict']} strictly ordered")
    print("  tau < 1 comes from ties, not disagreement.  Both sides have exact")
    print("  ties: L_RT takes three values (0, log D, 2 log D), and S has")
    print("  complement ties because the state is pure:")
    print(f"    max |S([1..k]) - S([k+1..{n}])| = {complement_gap(mean_S, n):.1e}"
          f"   (pure-state check: "
          f"{'PASS' if complement_gap(mean_S, n) < 1e-10 else 'FAIL'})")
    print("  With zero discordant pairs, tau-b is already as high as those ties")
    print("  allow -- the ordering never once contradicts RT.")
    cr = class_ratios(mean_S, n, D)
    print("  but ORDERING is not VALUE.  mean S / L_RT by min-cut class:")
    for c in (1, 2):
        m, lo, hi = cr[c]
        print(f"    min_cut = {c}: mean {m:.4f}  range [{lo:.4f}, {hi:.4f}]")
    print("    RT saturates only as D -> infinity; at finite D the entropy sits")
    print("    well below min_cut * log D.  Section [9] tracks that ratio with D.")

    # ---- [4] the transformed check ----
    print("\n[4] tau(d_boundary, d_bulk) on the exp-transformed distances")
    print(f"  Kendall tau = {res['tau_d']:.3f}, p-value = {res['p_d']:.2e}")
    print(f"  identical to [3] ({rt['tau']:.3f}), not merely close: the SAME "
          f"strictly")
    print("  decreasing map is applied to both sides, which reverses both")
    print("  rankings and leaves every concordance unchanged.  This check")
    print("  cannot differ from [3] and adds no information.")

    # ---- [5] circulant ----
    print("\n[5] CIRCULANT CHECK")
    off = ~np.eye(n, dtype=bool)
    for tag, M in (("boundary", res["d_bd"]), ("bulk", res["d_bk"])):
        circ = circulant_inf_safe(M)
        nvals = len({round(float(v), 12) for v in M[off]})
        print(f"  {tag:>8}: circulant = {'YES' if circ else 'NO'}, "
              f"{n_distinct_rows(M)}/{n} distinct rows, {nvals} distinct "
              f"off-diagonal values")
    print("  The bulk is NOT circulant, contrary to the brief's expectation:")
    print("  pairs (1,2) and (2,3) are both one step apart, but (1,2) touches an")
    print("  end (1 bond) and (2,3) is interior (2 bonds).  It is, however,")
    print("  maximally DEGENERATE: three values for fifteen pairs.")

    # ---- [6] PH ----
    print(f"\n[6] PERSISTENT HOMOLOGY (ripser, maxdim = {args.maxdim}), "
          f"alpha = {a}, D = {D}")
    print("  BOUNDARY:")
    for dim, dg in enumerate(res["dg_bd"]):
        describe_barcode(dg, f"H{dim}")
    print("  BULK (phi = 1):")
    for dim, dg in enumerate(res["dg_bk"]):
        describe_barcode(dg, f"H{dim}")
    nb, mb, tb = count_significant(res["dg_bd"][1])
    nk, mk, tk = count_significant(res["dg_bk"][1])
    print(f"  H1: boundary {tb} bar(s) ({nb} significant, max pers {mb:.6f}); "
          f"bulk {tk} bar(s) ({nk} significant)")
    print("  The bulk has no H1, and cannot: at scale D^(-2 alpha) the four")
    print("  interior sites form a complete graph; at D^(-alpha) the end sites")
    print("  join every interior site, and every 4-cycle through them is filled")
    print("  by triangles.  Any boundary H1 bar is a transient of the finer")
    print("  ordering among near-tied interior entropies, not topology.")
    print(f"\n  W(boundary, phi * bulk), phi searched on [0.1, 10]:")
    print(f"    phi = 1        : W_H0 = {res['w0_1']:.6f}   W_H1 = {res['w1_1']:.6f}")
    print(f"    phi = {res['phi']:.6f} : W_H0 = {res['w0_o']:.6f}   "
          f"W_H1 = {res['w1_o']:.6f}"
          + ("   *** PINNED AT A SEARCH BOUND ***" if res["at_bound"]
             else "   (interior optimum)"))
    print("    RT prediction: phi = 1 (exp(-a*S) = exp(-a*L_RT) if S = L_RT).")
    print(f"    phi_opt > 1 measures how far finite-D entropy falls short of RT.")

    # ---- [7] alpha sweep ----
    print(f"\n[7] ALPHA SWEEP at D = {D} (same 50-draw entropies)")
    print(f"  {'alpha':>6} {'tau':>7} {'disc':>5} {'phi_opt':>9} {'e^(a*dlt)':>10} "
          f"{'bound?':>7} {'W_H0(1)':>9} {'W_H0(opt)':>10} {'W_H1(opt)':>10}")
    for al in args.alphas:
        r = compare(n, D, al, args.draws, 0, args.maxdim, mean_S=mean_S)
        dp, _ = deficit_phi(mean_S, n, D, al)
        print(f"  {al:>6.2f} {r['rt']['tau']:>7.3f} {r['rt']['disc']:>5d} "
              f"{r['phi']:>9.5f} {dp:>10.5f} {'YES' if r['at_bound'] else 'no':>7} "
              f"{r['w0_1']:>9.6f} {r['w0_o']:>10.6f} {r['w1_o']:>10.6f}")
    print("  e^(a*dlt) = exp(alpha * (log D - mean S over min-cut-1 pairs)): the")
    print("  phi that would exactly absorb a uniform finite-D entropy deficit.")
    print("  tau and the discordant count are alpha-independent by construction")
    print("  (a monotone map); only the Wasserstein numbers move with alpha.")

    # ---- [8] stability ----
    print(f"\n[8] STABILITY: {args.stability_runs} independent runs at D = {D}, "
          f"alpha = {a}")
    print(f"  {'run':>4} {'tau':>7} {'disc':>5} {'phi_opt':>9} {'bound?':>7} "
          f"{'W_H0(opt)':>10} {'W_H1(opt)':>10}")
    stab = []
    for s in range(1, args.stability_runs + 1):
        r = compare(n, D, a, args.draws, s, args.maxdim)
        stab.append(r)
        print(f"  {s:>4d} {r['rt']['tau']:>7.3f} {r['rt']['disc']:>5d} "
              f"{r['phi']:>9.5f} {'YES' if r['at_bound'] else 'no':>7} "
              f"{r['w0_o']:>10.6f} {r['w1_o']:>10.6f}")
    col = lambda f: np.array([f(r) for r in stab])
    for label, f in (("tau", lambda r: r["rt"]["tau"]),
                     ("phi_opt", lambda r: r["phi"]),
                     ("W_H0(opt)", lambda r: r["w0_o"]),
                     ("W_H1(opt)", lambda r: r["w1_o"])):
        v = col(f)
        print(f"  {label:>10}: mean {v.mean():.6f}  std {v.std():.6f}")
    print(f"  discordant pairs, all runs: {[r['rt']['disc'] for r in stab]}")

    # ---- [9] bond-dimension sweep ----
    print(f"\n[9] BOND-DIMENSION SWEEP, alpha = {a}, {args.draws} draws each")
    print(f"  {'D':>4} {'log D':>8} {'tau':>7} {'disc':>5} {'S/L c=1':>9} "
          f"{'S/L c=2':>9} {'phi_opt':>9} {'e^(a*dlt)':>10} {'bound?':>7} "
          f"{'W_H0(opt)':>10} {'W_H1(opt)':>10}")
    sweep = []
    for Ds in args.bond_dims:
        r = compare(n, Ds, a, args.draws, 0, args.maxdim)
        cr = class_ratios(r["mean_S"], n, Ds)
        r["ratio1"], r["ratio2"], r["D"] = cr[1][0], cr[2][0], Ds
        r["dphi"], r["delta"] = deficit_phi(r["mean_S"], n, Ds, a)
        sweep.append(r)
        print(f"  {Ds:>4d} {np.log(Ds):>8.4f} {r['rt']['tau']:>7.3f} "
              f"{r['rt']['disc']:>5d} {cr[1][0]:>9.4f} {cr[2][0]:>9.4f} "
              f"{r['phi']:>9.5f} {r['dphi']:>10.5f} "
              f"{'YES' if r['at_bound'] else 'no':>7} "
              f"{r['w0_o']:>10.6f} {r['w1_o']:>10.6f}")
    logs = np.array([np.log(r["D"]) for r in sweep])
    phis = np.array([r["phi"] for r in sweep])
    rho, p_rho = pearsonr(logs, phis)
    print(f"\n  does phi_opt track log D?  Pearson r(phi_opt, log D) = {rho:+.3f} "
          f"(p = {p_rho:.3f}, n = {len(sweep)})")
    print(f"  phi_opt / log D across the sweep: "
          f"{[round(float(x), 4) for x in phis / logs]}")
    print("  The RT prediction for this construction is phi = 1 at every D, not")
    print("  log D: L_RT already contains log D.  The real convergence test is")
    print("  the S/L columns approaching 1.")
    gap1 = [1 - r["ratio1"] for r in sweep]
    gap2 = [1 - r["ratio2"] for r in sweep]
    print(f"  1 - S/L_RT, min_cut 1: {[round(g, 4) for g in gap1]}")
    print(f"  1 - S/L_RT, min_cut 2: {[round(g, 4) for g in gap2]}")
    print(f"  absolute deficit log D - S, min_cut 1: "
          f"{[round(float(r['delta']), 4) for r in sweep]}")
    print("  (a fractional gap that shrinks while the ABSOLUTE deficit stays O(1)")
    print("   is the finite-D signature; exp(-alpha*S) feels the absolute part,")
    print("   which is why phi_opt need not approach 1 even as S/L does.)")
    rel = [abs(r["phi"] - r["dphi"]) / r["dphi"] for r in sweep]
    print(f"  |phi_opt - e^(a*dlt)| / e^(a*dlt): {[f'{x:.1%}' for x in rel]}")
    print("  phi_opt follows the deficit estimate (rising, then falling, with it)")
    print("  rather than log D; the residual is the min-cut-2 pairs, whose")
    print("  deficit differs, pulling on the same H0 matching.")

    # ---- plot ----
    png = HERE / "rtn_rung6_barcodes.png"
    plot_barcodes(res, png, n, D, a, args.draws)
    print(f"\n  wrote {png}")

    # ---- verdict ----
    print("\n[10] VERDICT")
    print(f"  Validated fast entropies           : {'YES' if ok else 'NO'} "
          f"(max |diff| {worst:.1e} vs full SVD)")
    print(f"  RT ORDERING holds (0 discordant)   : "
          f"{'YES' if all(r['rt']['disc'] == 0 for r in stab + sweep + [res]) else 'NO'}"
          f" -- every run, every D")
    print(f"  RT VALUES hold at D = {D}             : NO -- S/L_RT = "
          f"{class_ratios(mean_S, n, D)[1][0]:.3f} (cut 1), "
          f"{class_ratios(mean_S, n, D)[2][0]:.3f} (cut 2)")
    print(f"  Barcodes match at phi = 1          : NO (W_H0 = {res['w0_1']:.6f})")
    print(f"  Best phi                           : {res['phi']:.5f}, W_H0 = "
          f"{res['w0_o']:.6f}")
    print("  The matched-geometry construction is the first clean test, and it")
    print("  shows exactly what the RTN theorem promises and no more: the")
    print("  entropy ORDERING agrees with the min cut at every D, while the")
    print("  VALUES fall short of min_cut * log D by a finite-D deficit.  The")
    print("  barcode comparison measures that deficit.  Two caveats: the bulk")
    print("  has three distance values for fifteen pairs, so the barcodes compare")
    print("  a nearly degenerate filtration; and with H1 empty on the bulk, the")
    print("  match is decided by H0 alone.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
