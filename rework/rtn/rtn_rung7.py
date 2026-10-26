#!/usr/bin/env python3
"""Rung 7 of the RTN rebuild: does the open chain converge to RT as D grows?

GOAL AS SET
-----------
Confirm W_H0 -> 0 as D -> infinity on the open chain and fit the convergence
curve.  Everything else is rung 6: 6 sites, open chain, complex Gaussian MPS
tensors, 50 draws per D, alpha = 1, d = exp(-alpha * x) on both sides.

WHAT WAS CHANGED, AND WHY (agreed before this was built)
--------------------------------------------------------
1. The brief's sweep D = [16, 32, 64, 128, 256] is not computable for ALL 15
   intervals.  An interior interval (min cut 2) has up to D^2 Schmidt values,
   so each draw needs a dense D^2 x D^2 eigenproblem: cost ~ D^6, and at
   D = 256 a single Gram matrix is 68.7 GB against 17 GB of RAM.  So rung 7
   runs TWO routes:

     route A, one-cut intervals, D = 16..256:  intervals touching a chain end
       cut one bond, so their spectrum is a D x D problem.  This gives the
       finite-D entropy deficit delta(D) = log D - S to the full range.
     route B, all 15 intervals, D = 16, 24, 32, 40:  the full persistent-
       homology comparison, exactly as in rung 6, at the D values that fit.
     spot check, all 15 intervals at D = 64, 3 draws:  RT ordering only.

2. The brief's success criterion (W_H0 ~ D^-beta with beta > 0, R^2 > 0.95)
   CANNOT FAIL, so it cannot confirm convergence.  Every exp(-S) and
   exp(-L_RT) entry shrinks like a power of 1/D, so the ABSOLUTE W_H0 goes to
   zero whether or not RT holds.  Section [0b] reruns the counterexample: fake
   entropies with RT's ordering REVERSED pass the criterion with R^2 ~ 0.997.
   The power law is still fitted and reported, labelled for what it is.

   The real convergence test: rung 6 found phi_opt ~= exp(alpha * delta),
   with delta the absolute deficit on one-cut pairs.  So "phi_opt -> 1" and
   "delta -> 0" are the same question -- and that one CAN come out either
   way.  If delta tends to a constant, phi_opt stays above 1 at every D, even
   though S / L_RT = 1 - delta / log D still creeps toward 1.  S / L_RT is
   therefore fitted against 1 / log D, its natural variable, not against D.

ENTROPY METHODS (both validated in section [0] before anything runs)
--------------------------------------------------------------------
Environments are Gram matrices, G_L[a,a'] = <L_a'|L_a>, built with two BLAS
calls per site.

* one-cut (bond k): |Psi> = sum_a |L_a>|R_a>, so rho_L = L K L^dagger with
  K = G_R and L^dagger L = G_L^T -- the TRANSPOSE.  For complex tensors
  G_L^T = conj(G_L) != G_L; an early draft of this rung used G_L and was off
  by up to 0.29 nats, caught by this validation.  spec(rho_L) =
  spec(sqrt(G_R) G_L^T sqrt(G_R)).

* all intervals: rung 6's construction, but K^1/2 = sqrt(G_L) (x) sqrt(G_R)
  is applied as two D x D factors instead of one D^2 x D^2 eigendecomposition.
  That roughly halves the cost and removes the D^2 x D^2 `eigh` that crashed
  inside LAPACK (ZHEEVD) at D = 64.

Seeding is rungs 5-6's: draw d of run s uses np.random.seed(1000*s + d).
Runs s = 0 (primary) and s = 1..5 (the stability seeds) are all used, so every
D has 6 x 50 = 300 draws.  Per-run results are cached in rtn_rung7_cache/ so a
crash loses at most one run; a cached run is reported as loaded, never
silently.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

from rtn_rung3 import barcodes
from rtn_rung4 import joint_cost, optimal_phi_joint
from rtn_rung5 import entropy_matrix, random_mps_state
from rtn_rung6 import (chain_min_cut, class_ratios, complement_gap, draw_tensors,
                       entropy_matrix_fast, exp_distance, rt_check,
                       rt_length_matrix)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "rtn_rung7_cache"

N_SITES = 6
ALPHA = 1.0
N_DRAWS = 50
RUN_SEEDS = [0, 1, 2, 3, 4, 5]          # primary + the brief's 5 stability seeds
ONECUT_DS = [16, 32, 64, 128, 256]
FULL_DS = [16, 24, 32, 40]
SPOT_D, SPOT_DRAWS = 64, 3
S_ZERO_TOL = 1e-9


# ==========================================================================
# 1. ENTROPIES
# ==========================================================================

def hsqrt(G):
    """Hermitian PSD square root via eigh (clips rounding negatives)."""
    w, U = np.linalg.eigh((G + G.conj().T) / 2.0)
    return (U * np.sqrt(np.clip(w, 0.0, None))) @ U.conj().T


def entropy_from_weights(lam):
    lam = np.clip(np.real(lam), 0.0, None)
    p = lam / lam.sum()
    p = p[p > 1e-15]
    S = float(-np.sum(p * np.log(p)))
    return 0.0 if S < S_ZERO_TOL else S


def environments(tensors):
    """Left and right Gram environments, two BLAS matmuls per site.

    GL[k][a,a'] = <L_a'|L_a> for the block of sites 1..k (k = 0: trivial).
    GR[k][a,a'] = <R_a'|R_a> for the block of sites k+2..n (0-based k+1..n-1).
    """
    n = len(tensors)
    GL = [np.ones((1, 1), dtype=complex)]
    for k in range(n):
        A = tensors[k]
        l, s, r = A.shape
        Z = (GL[-1].T @ A.reshape(l, s * r)).reshape(l * s, r)
        GL.append(Z.T @ A.conj().reshape(l * s, r))
    GR = [None] * n
    GR[n - 1] = np.ones((1, 1), dtype=complex)
    for k in range(n - 1, 0, -1):
        A = tensors[k]
        l, s, r = A.shape
        Z = (A.reshape(l * s, r) @ GR[k]).reshape(l, s * r)
        GR[k - 1] = Z @ A.conj().reshape(l, s * r).T
    return GL, GR


def onecut_entropies(tensors):
    """Entropy across each internal bond k = 1..n-1 (left block = sites 1..k).

    spec(rho_L) = spec(sqrt(G_R) G_L^T sqrt(G_R)) -- note the transpose.
    """
    GL, GR = environments(tensors)
    out = []
    for k in range(1, len(tensors)):
        sR = hsqrt(GR[k - 1])
        Q = sR @ GL[k].T @ sR
        out.append(entropy_from_weights(np.linalg.eigvalsh((Q + Q.conj().T) / 2.0)))
    return np.array(out)


def full_entropies(tensors):
    """S[i, j] for all 15 intervals, K^1/2 applied as sqrt(GL) (x) sqrt(GR)."""
    n = len(tensors)
    GL, GR = environments(tensors)
    S = np.zeros((n, n))
    for i in range(n):
        dl = tensors[i].shape[0]
        X = np.einsum("ac,bd->abcd", np.eye(dl), np.eye(dl)).astype(complex)
        sL = hsqrt(GL[i])
        for j in range(i, n):
            X = np.einsum("abcd,csr,dst->abrt", X, tensors[j], tensors[j].conj(),
                          optimize=True)
            if j == i:
                continue
            dr = tensors[j].shape[2]
            sR = hsqrt(GR[j])
            G4 = X.transpose(1, 3, 0, 2)              # [a',b',a,b] = <M_a'b'|M_ab>
            G4 = np.einsum("pa,qb,abcd,cr,ds->pqrs", sL, sR, G4, sL, sR,
                           optimize=True)
            Q = G4.reshape(dl * dr, dl * dr)
            S[i, j] = S[j, i] = entropy_from_weights(
                np.linalg.eigvalsh((Q + Q.conj().T) / 2.0))
    return S


# ==========================================================================
# 2. CACHED RUNS
# ==========================================================================

def _cached(name, compute):
    """Load rtn_rung7_cache/<name>.npz if present, else compute and save."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.npz"
    if path.exists():
        return np.load(path)["data"], True
    data = compute()
    np.savez(path, data=data)
    return data, False


def onecut_run(D, s, n_draws, n_sites=N_SITES):
    """(n_draws, n_sites-1) array of bond entropies for run s at bond dim D."""
    return _cached(f"onecut_D{D}_s{s}_n{n_draws}", lambda: np.array(
        [onecut_entropies(draw_tensors(n_sites, D, 1000 * s + d))
         for d in range(n_draws)]))


def full_run(D, s, n_draws, n_sites=N_SITES):
    """(n_draws, n, n) array of interval entropies for run s at bond dim D."""
    return _cached(f"full_D{D}_s{s}_n{n_draws}", lambda: np.array(
        [full_entropies(draw_tensors(n_sites, D, 1000 * s + d))
         for d in range(n_draws)]))


# ==========================================================================
# 3. FITS
# ==========================================================================

def linfit(x, y):
    """y = c0 + c1 * x by least squares; returns (c0, c1, se0, se1, R^2, SSE)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    sse = float(res @ res)
    dof = max(len(x) - 2, 1)
    cov = (sse / dof) * np.linalg.inv(A.T @ A)
    r2 = 1.0 - sse / float(np.sum((y - y.mean()) ** 2))
    return float(coef[0]), float(coef[1]), float(np.sqrt(cov[0, 0])), \
        float(np.sqrt(cov[1, 1])), float(r2), sse


def power_law(Ds, W):
    """W = a * D^-beta, fitted in log-log space.  Returns (beta, se, R^2, a)."""
    c0, c1, _, se1, r2, _ = linfit(np.log(Ds), np.log(W))
    return -c1, se1, r2, float(np.exp(c0))


# ==========================================================================
# 4. COUNTERFACTUAL: can the brief's criterion fail?
# ==========================================================================

def fake_W_curve(Ds, kind, alpha=ALPHA, n_sites=N_SITES):
    """W_H0(phi_opt) vs D for synthetic entropies that VIOLATE RT.

    Returns (W, pinned): pinned[k] is True where phi_opt hit the search bound.
    """
    W, pinned = [], []
    for D in Ds:
        L = rt_length_matrix(n_sites, D)
        S = np.zeros_like(L)
        for i in range(n_sites):
            for j in range(n_sites):
                if i != j:
                    c = chain_min_cut(i + 1, j + 1, n_sites)
                    S[i, j] = {"reversed": 3 - c, "half": 0.5 * c}[kind] * np.log(D)
        bd = barcodes(exp_distance(S, alpha), 1)
        bk = barcodes(exp_distance(L, alpha), 1)
        phi, at_bound = optimal_phi_joint(bd, bk)
        W.append(joint_cost(bd, bk, phi)[0])
        pinned.append(bool(at_bound))
    return np.array(W), pinned


# ==========================================================================
# 5. MAIN
# ==========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draws", type=int, default=N_DRAWS)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--onecut-ds", type=int, nargs="+", default=ONECUT_DS)
    p.add_argument("--full-ds", type=int, nargs="+", default=FULL_DS)
    p.add_argument("--seeds", type=int, nargs="+", default=RUN_SEEDS)
    p.add_argument("--spot-d", type=int, default=SPOT_D)
    p.add_argument("--spot-draws", type=int, default=SPOT_DRAWS)
    args = p.parse_args(argv)
    n, a = N_SITES, args.alpha
    t_start = time.time()

    print("=" * 78)
    print("RTN RUNG 7 -- convergence of the open chain toward RT as D grows")
    print("=" * 78)
    print(f"6-site open-chain random MPS, complex Gaussian tensors, alpha = {a}")
    print(f"route A (one-cut intervals) : D = {args.onecut_ds}")
    print(f"route B (all 15 intervals)  : D = {args.full_ds}")
    print(f"spot check (all 15)         : D = {args.spot_d}, {args.spot_draws} draws")
    print(f"runs per D: seeds s = {args.seeds} (s = 0 primary, s = 1..5 the "
          f"stability seeds), {args.draws} draws each")
    print("draw d of run s seeded np.random.seed(1000*s + d), as in rungs 5-6\n")

    # ---- [0] validation ----
    print("[0] VALIDATION of both entropy methods against earlier rungs")
    w_full = 0.0
    for Dv in (2, 4):
        for d in range(3):
            w_full = max(w_full, float(np.max(np.abs(
                full_entropies(draw_tensors(n, Dv, d))
                - entropy_matrix(random_mps_state(n, Dv, d), n, Dv)))))
    w_full6 = 0.0
    for Dv in (8, 16):
        for d in range(2):
            w_full6 = max(w_full6, float(np.max(np.abs(
                full_entropies(draw_tensors(n, Dv, d))
                - entropy_matrix_fast(draw_tensors(n, Dv, d))))))
    w_one = 0.0
    for Dv in (4, 8, 16):
        for d in range(3):
            ref = entropy_matrix_fast(draw_tensors(n, Dv, d))
            ref_bonds = [ref[1, n - 1]] + [ref[0, k - 1] for k in range(2, n)]
            w_one = max(w_one, float(np.max(np.abs(
                onecut_entropies(draw_tensors(n, Dv, d)) - np.array(ref_bonds)))))
    checks = [("all-interval route vs rung-5 full SVD (D = 2, 4)", w_full),
              ("all-interval route vs rung-6 transfer (D = 8, 16)", w_full6),
              ("one-cut route vs rung-6 transfer (D = 4, 8, 16)", w_one)]
    ok = True
    for label, w in checks:
        good = w < 1e-10
        ok &= good
        print(f"    {label:<50} max |diff| = {w:.1e}  {'PASS' if good else 'FAIL'}")
    print("    (bond 1 is the single site [1..1], checked via its complement")
    print("     [2..6]; bonds 2..5 are the pairs (1,k) directly)")
    if not ok:
        print("\n*** VALIDATION FAILED -- stopping before any sweep ***")
        return 1

    # ---- [0b] counterfactual ----
    print("\n[0b] CAN THE BRIEF'S SUCCESS CRITERION FAIL?")
    print("     Synthetic entropies that VIOLATE RT, scored by the brief's test")
    print("     (power law W_H0 ~ D^-beta, success = beta > 0 and R^2 > 0.95):")
    any_pass = False
    for kind, desc in (("reversed", "S = (3 - cut) log D: RT ordering REVERSED"),
                       ("half", "S = 0.5 * L_RT: RT broken by a factor of 2")):
        Wf, pinned = fake_W_curve(args.onecut_ds, kind, a)
        beta, se, r2, _ = power_law(args.onecut_ds, Wf)
        passes = beta > 0 and r2 > 0.95
        any_pass |= passes
        print(f"     {desc}")
        print(f"       beta = {beta:.3f}, R^2 = {r2:.4f} -> criterion "
              f"{'PASSES' if passes else 'fails'}"
              + (f"; phi_opt pinned at the [0.1, 10] bound at D = "
                 f"{[d for d, pn in zip(args.onecut_ds, pinned) if pn]}"
                 if any(pinned) else ""))
        if not passes and any(pinned):
            print("       (it fails because the phi it needs runs off the search")
            print("        window, not because the test detected the violation)")
    if any_pass:
        print("     At least one RT-violating construction passes, so passing the")
        print("     criterion cannot confirm convergence.")
    else:
        print("     Neither violation passes at these D -- but see the pinning")
        print("     notes: a failure here can be a search-window artifact.")

    # ---- [A] one-cut route ----
    print(f"\n[A] ROUTE A: one-cut intervals, entropy across each bond, "
          f"D = {args.onecut_ds}")
    print("    delta = log D - S (absolute deficit), S/L_RT = S / log D")
    rows_A = []
    for D in args.onecut_ds:
        t0 = time.time()
        runs, loaded = [], []
        for s in args.seeds:
            data, was = onecut_run(D, s, args.draws)
            runs.append(data)
            loaded.append(was)
        runs = np.array(runs)                          # (runs, draws, bonds)
        logD = np.log(D)
        per_run_mean = runs.mean(axis=(1, 2))          # mean S over draws, bonds
        per_run_delta = logD - per_run_mean
        bond_mean = runs.mean(axis=(0, 1))             # per bond, all runs
        row = {"D": D, "logD": logD,
               "S": float(runs.mean()), "S_sd": float(per_run_mean.std()),
               "delta": float(logD - runs.mean()),
               "delta_sd": float(per_run_delta.std()),
               "ratio": float(runs.mean() / logD),
               "bond_delta": logD - bond_mean}
        rows_A.append(row)
        tag = (" (all runs loaded from cache)" if all(loaded) else
               " (some runs loaded from cache)" if any(loaded) else "")
        print(f"\n  D = {D:>3}  [{time.time() - t0:7.1f} s{tag}]")
        print(f"    mean S         = {row['S']:.6f}   (sd over runs {row['S_sd']:.6f})")
        print(f"    S / L_RT       = {row['ratio']:.6f}")
        print(f"    delta          = {row['delta']:.6f}   (sd over runs "
              f"{row['delta_sd']:.6f})")
        print(f"    delta by bond  = {np.round(row['bond_delta'], 4).tolist()}"
              f"   (bonds 1..5)")
        print(f"    e^(alpha*delta)= {np.exp(a * row['delta']):.6f}"
              f"   (phi that would absorb this deficit)")

    DsA = np.array([r["D"] for r in rows_A], float)
    dA = np.array([r["delta"] for r in rows_A])
    rA = np.array([r["ratio"] for r in rows_A])

    print("\n  SUMMARY, route A")
    print(f"  {'D':>4} {'log D':>8} {'mean S':>10} {'S/L_RT':>9} {'delta':>9} "
          f"{'sd':>8} {'e^(a*dlt)':>10}")
    for r in rows_A:
        print(f"  {r['D']:>4d} {r['logD']:>8.4f} {r['S']:>10.6f} {r['ratio']:>9.6f} "
              f"{r['delta']:>9.6f} {r['delta_sd']:>8.6f} "
              f"{np.exp(a * r['delta']):>10.6f}")

    print("\n  FIT 1: S/L_RT = A - B / log D   (does S/L_RT -> 1?)")
    A0, A1, Ase, Bse, r2_ratio, _ = linfit(1.0 / np.log(DsA), rA)
    print(f"    A = {A0:.6f} +- {Ase:.6f},  B = {-A1:.6f} +- {Bse:.6f},  "
          f"R^2 = {r2_ratio:.6f}")
    print(f"    extrapolated S/L_RT at D -> infinity: {A0:.6f}  "
          f"({abs(A0 - 1) / Ase:.1f} sigma from 1)")
    print("    By definition S/L_RT = 1 - delta/log D on one-cut pairs, so ANY")
    print("    bounded delta sends this ratio to 1.  It is a weak test: it")
    print("    cannot tell delta -> 0 from delta -> constant.")

    print("\n  FIT 2: delta(D) = delta_inf + c / D   (does the deficit vanish?)")
    d0, d1, d0se, d1se, r2_d, sse_c = linfit(1.0 / DsA, dA)
    print(f"    delta_inf = {d0:.6f} +- {d0se:.6f},  c = {d1:.4f} +- {d1se:.4f},  "
          f"R^2 = {r2_d:.4f}")
    print(f"    delta_inf is {d0 / d0se:.1f} sigma from zero")
    print("\n  FIT 3: delta(D) = a * D^-gamma   (the delta_inf = 0 hypothesis)")
    gam, gse, r2_g, ag = power_law(DsA, dA)
    sse_p = float(np.sum((dA - ag * DsA ** (-gam)) ** 2))
    print(f"    gamma = {gam:.4f} +- {gse:.4f},  R^2 (log space) = {r2_g:.4f}")
    print(f"    SSE in delta: constant + c/D = {sse_c:.3e},  pure power law = "
          f"{sse_p:.3e}")
    better = "constant + c/D" if sse_c < sse_p else "pure power law"
    print(f"    better fit: {better} (both are 2-parameter models, "
          f"{len(DsA)} points)")
    print(f"    phi implied at D -> infinity by fit 2: e^(alpha*delta_inf) = "
          f"{np.exp(a * d0):.4f}")
    print("    (5 points and 2-parameter fits: treat these as a direction, not")
    print("     a law.  The run-to-run sd above is the honest error scale.)")

    # ---- [B] full route ----
    print(f"\n[B] ROUTE B: all 15 intervals, full PH comparison, D = {args.full_ds}")
    rows_B = []
    for D in args.full_ds:
        t0 = time.time()
        per_run = []
        any_loaded = False
        for s in args.seeds:
            data, was = full_run(D, s, args.draws)
            any_loaded |= was
            mean_S = data.mean(axis=0)
            L = rt_length_matrix(n, D)
            bd = barcodes(exp_distance(mean_S, a), 1)
            bk = barcodes(exp_distance(L, a), 1)
            phi, at_bound = optimal_phi_joint(bd, bk)
            w0, w1, _ = joint_cost(bd, bk, phi)
            rt = rt_check(mean_S, L)
            cr = class_ratios(mean_S, n, D)
            vals1 = [mean_S[i, j] for i in range(n) for j in range(i + 1, n)
                     if chain_min_cut(i + 1, j + 1, n) == 1]
            per_run.append({"s": s, "phi": phi, "bound": at_bound, "w0": w0,
                            "w1": w1, "tau": rt["tau"], "disc": rt["disc"],
                            "r1": cr[1][0], "r2": cr[2][0],
                            "delta1": float(np.log(D) - np.mean(vals1)),
                            "meanS": float(np.mean(mean_S[np.triu_indices(n, 1)])),
                            "cgap": complement_gap(mean_S, n)})
        col = lambda k: np.array([r[k] for r in per_run])
        row = {"D": D, "logD": np.log(D), "runs": per_run,
               **{k: float(col(k).mean()) for k in
                  ("phi", "w0", "w1", "tau", "r1", "r2", "delta1", "meanS")},
               **{k + "_sd": float(col(k).std()) for k in ("phi", "w0", "delta1")},
               "disc_all": [int(x) for x in col("disc")],
               "bound_any": bool(col("bound").any())}
        rows_B.append(row)
        print(f"\n  D = {D:>3}  [{time.time() - t0:7.1f} s"
              f"{' (cache used)' if any_loaded else ''}]")
        print(f"    {'run':>4} {'phi_opt':>9} {'bound?':>7} {'W_H0':>10} {'W_H1':>10} "
              f"{'tau':>6} {'disc':>5} {'S/L c1':>8} {'S/L c2':>8}")
        for r in per_run:
            print(f"    {r['s']:>4d} {r['phi']:>9.5f} "
                  f"{'YES' if r['bound'] else 'no':>7} {r['w0']:>10.3e} "
                  f"{r['w1']:>10.3e} {r['tau']:>6.3f} {r['disc']:>5d} "
                  f"{r['r1']:>8.4f} {r['r2']:>8.4f}")
        print(f"    pure-state check, max |S(A) - S(A^c)| over runs: "
              f"{max(r['cgap'] for r in per_run):.1e}")

    print("\n  SUMMARY, route B (mean over the 6 runs; sd in parentheses)")
    print(f"  {'D':>4} {'mean S':>9} {'S/L c1':>8} {'S/L c2':>8} {'phi_opt':>18} "
          f"{'e^(a*d1)':>9} {'W_H0(opt)':>22} {'disc (all runs)':>18}")
    for r in rows_B:
        print(f"  {r['D']:>4d} {r['meanS']:>9.5f} {r['r1']:>8.4f} {r['r2']:>8.4f} "
              f"{r['phi']:>9.5f} ({r['phi_sd']:.5f}) "
              f"{np.exp(a * r['delta1']):>9.5f} "
              f"{r['w0']:>10.3e} ({r['w0_sd']:.2e}) "
              f"{str(r['disc_all']):>18}")
    if any(r["bound_any"] for r in rows_B):
        print("  *** at least one run's phi_opt was PINNED at a search bound ***")

    DsB = np.array([r["D"] for r in rows_B], float)
    WB = np.array([r["w0"] for r in rows_B])
    beta, bse, r2_w, aw = power_law(DsB, WB)
    print("\n  THE BRIEF'S CRITERION: W_H0(phi_opt) ~ D^-beta over route B")
    print(f"    beta = {beta:.4f} +- {bse:.4f},  R^2 = {r2_w:.4f}  -> "
          f"{'meets' if (beta > 0 and r2_w > 0.95) else 'does not meet'} "
          f"beta > 0, R^2 > 0.95")
    print("    UNINFORMATIVE: section [0b] shows RT-reversed data passing the")
    print("    same test.  This number confirms that both distance matrices")
    print("    shrink, not that their entropies converge.")

    phis = np.array([r["phi"] for r in rows_B])
    print("\n  phi_opt vs D: does it approach 1?")
    print(f"    phi_opt = {np.round(phis, 4).tolist()} at D = {[int(x) for x in DsB]}")
    print(f"    distance from 1: {np.round(phis - 1, 4).tolist()}")
    print(f"    route A's e^(alpha*delta) at the same D range is "
          f"{np.round(np.exp(a * dA[DsA <= max(DsB)]), 4).tolist()} "
          f"(D = {[int(x) for x in DsA[DsA <= max(DsB)]]})")
    print(f"    and at D = {int(DsA[-1])}: {np.exp(a * dA[-1]):.4f}")

    # ---- [C] spot check at D = 64 ----
    print(f"\n[C] SPOT CHECK: RT ordering at D = {args.spot_d}, all 15 intervals, "
          f"{args.spot_draws} draws (run s = 0)")
    t0 = time.time()
    spot, was = _cached(f"full_D{args.spot_d}_s0_spot{args.spot_draws}", lambda: np.array(
        [full_entropies(draw_tensors(n, args.spot_d, d)) for d in range(args.spot_draws)]))
    mean_spot = spot.mean(axis=0)
    rt64 = rt_check(mean_spot, rt_length_matrix(n, args.spot_d))
    per_draw_disc = [rt_check(S1, rt_length_matrix(n, args.spot_d))["disc"] for S1 in spot]
    cr64 = class_ratios(mean_spot, n, args.spot_d)
    print(f"    [{time.time() - t0:.1f} s{' (cache)' if was else ''}]")
    print(f"    Kendall tau = {rt64['tau']:.3f}, p = {rt64['p']:.2e}, "
          f"discordant = {rt64['disc']} of {rt64['strict']}")
    print(f"    discordant per individual draw: {per_draw_disc}")
    print(f"    S/L_RT: cut 1 = {cr64[1][0]:.4f}, cut 2 = {cr64[2][0]:.4f}")
    print(f"    pure-state check: max |S(A) - S(A^c)| = "
          f"{complement_gap(mean_spot, n):.1e}")
    print(f"    ({args.spot_draws} draws only: an ordering check, not a precision "
          f"measurement)")

    # ---- [D] plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.loglog(DsB, WB, "o", ms=7, label="route B: $W_{H_0}(\\phi_{opt})$")
    xs = np.geomspace(DsB.min(), DsB.max(), 50)
    ax.loglog(xs, aw * xs ** (-beta), "-", lw=1.5,
              label=f"fit $D^{{-{beta:.2f}}}$, $R^2$={r2_w:.3f}")
    ax.set_xlabel("bond dimension $D$")
    ax.set_ylabel("$W_{H_0}$ at $\\phi_{opt}$")
    ax.set_title("Brief's criterion (cannot fail, see [0b])", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(DsA, dA, yerr=[r["delta_sd"] for r in rows_A], fmt="o", ms=6,
                capsize=3, label="route A: $\\delta = \\log D - S$")
    xs = np.geomspace(DsA.min(), DsA.max(), 100)
    ax.plot(xs, d0 + d1 / xs, "-", lw=1.5,
            label=f"$\\delta_\\infty + c/D$, $\\delta_\\infty$={d0:.3f}")
    ax.plot(xs, ag * xs ** (-gam), "--", lw=1.5,
            label=f"$a D^{{-\\gamma}}$, $\\gamma$={gam:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("bond dimension $D$")
    ax.set_ylabel("one-cut entropy deficit $\\delta$ (nats)")
    ax.set_title("Real test: does the deficit vanish?", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.errorbar(DsB, phis, yerr=[r["phi_sd"] for r in rows_B], fmt="o", ms=6,
                capsize=3, label="route B: $\\phi_{opt}$")
    ax.plot(DsA, np.exp(a * dA), "s--", ms=5, label="route A: $e^{\\alpha\\delta}$")
    ax.axhline(1.0, color="k", lw=1, ls=":", label="RT prediction $\\phi=1$")
    ax.set_xscale("log")
    ax.set_xlabel("bond dimension $D$")
    ax.set_ylabel("$\\phi$")
    ax.set_title("$\\phi_{opt}$ vs $D$", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.suptitle("RTN rung 7: open-chain convergence toward RT", fontsize=11)
    fig.tight_layout()
    png = HERE / "rtn_rung7_convergence.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"\n  wrote {png}")

    # ---- [E] verdict ----
    print("\n[E] VERDICT")
    print(f"  RT ORDERING at large D        : discordant = {rt64['disc']} at "
          f"D = {args.spot_d}; route B runs {[r['disc_all'] for r in rows_B]}")
    print(f"  S/L_RT -> 1 (fit 1)           : A = {A0:.4f} +- {Ase:.4f}")
    print(f"  deficit delta(D), D = {[int(x) for x in DsA]} : "
          f"{np.round(dA, 4).tolist()}")
    print(f"  delta_inf (fit 2)             : {d0:.4f} +- {d0se:.4f} "
          f"({d0 / d0se:.1f} sigma from 0)")
    print(f"  phi_opt, D = {[int(x) for x in DsB]} : "
          f"{np.round(phis, 4).tolist()}")
    print(f"  brief's power law (route B)   : beta = {beta:.3f}, R^2 = {r2_w:.3f}"
          f" -- uninformative by [0b]")
    print(f"\n  total runtime {time.time() - t_start:.0f} s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
