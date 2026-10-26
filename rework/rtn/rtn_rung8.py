#!/usr/bin/env python3
"""Rung 8 of the RTN rebuild: is rung 7's residual deficit the ensemble's doing?

QUESTION
--------
Rung 7 found that the one-cut entropy deficit delta = log D - S does not vanish
as D grows: delta_inf = 0.634.  Is that a property of RT, or of the tensor
ensemble?  The brief proposed a global Haar state; that design was replaced,
by agreement, because it measures a different quantity:

  * a single-site cut of a global Haar state is D against D^5, so Page gives
    delta ~ 1/(2 D^4) -> 0, not 0.5 -- Page's 1/2 needs EQUAL dimensions;
  * a Haar state has no chain geometry (its RT graph is one 6-leg tensor), so
    against the open-chain RT it shows 8 discordant pairs at D = 64;
  * an explicit D^6 state needs 17 GB at D = 32 and 1.1 TB at D = 64.

THE TEST INSTEAD: BOND BY BOND, SAME CHAIN, TWO ENSEMBLES
---------------------------------------------------------
Each bond's entanglement spectrum is set by how many independent D x D random
matrix factors cross it.  One factor between two large flat blocks is Page's
equal-dimension case (delta -> 1/2); two factors in a row give a larger
deficit; a bond whose smaller side is far bigger than D gives delta -> 0.

  ensemble "T"  : rung 7's MPS -- site tensors x bond matrices T_k
      interior bonds 2-4: one factor (T_k)                  -> delta = 1/2
      end bonds 1, 5   : two factors (site tensor, T_k)     -> delta = delta_2
  ensemble "noT": the SAME site tensors with T_k removed
      end bonds 1, 5   : one site (dim D) against a bond (dim D) -> delta = 1/2
      interior bonds   : smaller side >> D                  -> delta -> 0

  predicted mean over the 5 bonds:  T   -> (2 delta_2 + 3 * 0.5) / 5
                                    noT -> (2 * 0.5 + 3 * 0) / 5 = 0.200

delta_2 is the deficit of a product of two independent D x D Ginibre matrices.
It is MEASURED in section [1] (at D = 1024) rather than taken from memory;
no published closed form was checked, so none is asserted.  The k = 1 case is
measured alongside it as a control on Page's 1/2.

PAIRED DRAWS
------------
rtn_rung6.draw_tensors consumes the RNG as: all site tensors, then all bond
matrices.  So for the same seed, the noT ensemble here uses EXACTLY rung 7's
site tensors and simply omits T_k -- a paired comparison, not two unrelated
samples.  Rung 7's T-ensemble entropies are reused from rtn_rung7_cache/.

PRE-REGISTRATION, HONESTLY STATED
---------------------------------
The predictions and the pass tolerance are printed in section [2] BEFORE any
measurement in this run.  They were set after a 10-draw probe up to D = 128
had been seen (in the conversation that led to this rung), so this is a
prediction from Page and random-matrix theory checked at full statistics, not
a blind pre-registration.

Seeding as rungs 5-7: draw d of run s uses np.random.seed(1000*s + d); runs
s = 0..5, 50 draws each.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rtn_rung3 import barcodes
from rtn_rung4 import joint_cost, optimal_phi_joint
from rtn_rung5 import interval_entropy
from rtn_rung6 import (chain_min_cut, class_ratios, complement_gap, exp_distance,
                       rt_check, rt_length_matrix)
from rtn_rung7 import (full_entropies, full_run, linfit, onecut_entropies,
                       onecut_run)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "rtn_rung8_cache"

N_SITES = 6
ALPHA = 1.0
N_DRAWS = 50
RUN_SEEDS = [0, 1, 2, 3, 4, 5]
ONECUT_DS = [16, 32, 64, 128, 256]
FULL_DS = [16, 24, 32, 40]
SPOT_D, SPOT_DRAWS = 64, 3
TOLERANCE = 0.02            # pre-registered: |fitted delta_inf - prediction|
GINIBRE_D, GINIBRE_REPS = 1024, 8
END_BONDS, INTERIOR_BONDS = [0, 4], [1, 2, 3]   # 0-based bond indices 1..5


# ==========================================================================
# 1. THE noT ENSEMBLE
# ==========================================================================

def draw_tensors_noT(n_sites, bond_dim, seed):
    """Rung 7's site tensors for this seed, with the bond matrices T_k omitted.

    Draws the site tensors exactly as rtn_rung6.draw_tensors does (same order,
    real part before imaginary) and stops before drawing any T_k, so the site
    tensors are identical to rung 7's for the same seed.
    """
    np.random.seed(seed)
    D = bond_dim
    dims = [1] + [D] * (n_sites - 1) + [1]
    A = [np.random.randn(dims[k], D, dims[k + 1])
         + 1j * np.random.randn(dims[k], D, dims[k + 1])
         for k in range(n_sites)]
    return [a / np.linalg.norm(a) for a in A]


def contract_state(tensors):
    """Full state vector from an MPS (validation only; small D)."""
    psi = tensors[0]
    for t in tensors[1:]:
        psi = np.tensordot(psi, t, axes=([-1], [0]))
    v = psi.reshape(-1)
    return v / np.linalg.norm(v)


def _cached8(name, compute):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.npz"
    if path.exists():
        return np.load(path)["data"], True
    data = compute()
    np.savez(path, data=data)
    return data, False


def onecut_run_noT(D, s, n_draws, n_sites=N_SITES):
    return _cached8(f"onecut_noT_D{D}_s{s}_n{n_draws}", lambda: np.array(
        [onecut_entropies(draw_tensors_noT(n_sites, D, 1000 * s + d))
         for d in range(n_draws)]))


def full_run_noT(D, s, n_draws, n_sites=N_SITES):
    return _cached8(f"full_noT_D{D}_s{s}_n{n_draws}", lambda: np.array(
        [full_entropies(draw_tensors_noT(n_sites, D, 1000 * s + d))
         for d in range(n_draws)]))


# ==========================================================================
# 2. RANDOM-MATRIX REFERENCE: products of k Ginibre matrices
# ==========================================================================

def ginibre_product_deficit(D, k, reps, seed=20261026):
    """Mean and standard error of log D - S for psi = G_1 ... G_k (D x D).

    Independent RNG (numpy Generator, fixed seed) -- it never touches the
    tensor draws.  k = 1 is Page's equal-dimension case.
    """
    rng = np.random.default_rng(seed + k)
    vals = []
    for _ in range(reps):
        M = np.eye(D, dtype=complex)
        for _ in range(k):
            M = M @ (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D)))
        s = np.linalg.svd(M, compute_uv=False)
        p = s ** 2 / np.sum(s ** 2)
        p = p[p > 1e-300]
        vals.append(np.log(D) + float(np.sum(p * np.log(p))))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(reps))


# ==========================================================================
# 3. ROUTE A: per-bond deficits for one ensemble
# ==========================================================================

def route_A(ensemble, Ds, seeds, n_draws):
    """Per-bond and mean delta at each D.  Returns a list of row dicts."""
    rows = []
    for D in Ds:
        t0 = time.time()
        runs, loaded = [], []
        for s in seeds:
            if ensemble == "T":
                data, was = onecut_run(D, s, n_draws)      # rung 7's cache
            else:
                data, was = onecut_run_noT(D, s, n_draws)
            runs.append(data)
            loaded.append(was)
        runs = np.array(runs)                             # (runs, draws, bonds)
        logD = np.log(D)
        bond_delta = logD - runs.mean(axis=(0, 1))        # (bonds,)
        run_bond_delta = logD - runs.mean(axis=1)         # (runs, bonds)
        rows.append({
            "D": D, "secs": time.time() - t0, "all_loaded": all(loaded),
            "bond": bond_delta,
            "bond_sd": run_bond_delta.std(axis=0),
            "end": float(bond_delta[END_BONDS].mean()),
            "interior": float(bond_delta[INTERIOR_BONDS].mean()),
            "mean": float(bond_delta.mean()),
            "mean_sd": float(run_bond_delta.mean(axis=1).std()),
        })
    return rows


def fit_inf(Ds, y):
    """delta = delta_inf + c / D; returns (delta_inf, se, c, R^2)."""
    d0, d1, d0se, _, r2, _ = linfit(1.0 / np.asarray(Ds, float), y)
    return d0, d0se, d1, r2


# ==========================================================================
# 4. ROUTE B: full barcode comparison for one ensemble
# ==========================================================================

def route_B(ensemble, Ds, seeds, n_draws, alpha, n_sites=N_SITES):
    rows = []
    for D in Ds:
        t0 = time.time()
        per_run, any_loaded = [], False
        for s in seeds:
            if ensemble == "T":
                data, was = full_run(D, s, n_draws)       # rung 7's cache
            else:
                data, was = full_run_noT(D, s, n_draws)
            any_loaded |= was
            mean_S = data.mean(axis=0)
            L = rt_length_matrix(n_sites, D)
            bd = barcodes(exp_distance(mean_S, alpha), 1)
            bk = barcodes(exp_distance(L, alpha), 1)
            phi, at_bound = optimal_phi_joint(bd, bk)
            w0, w1, _ = joint_cost(bd, bk, phi)
            rt = rt_check(mean_S, L)
            cr = class_ratios(mean_S, n_sites, D)
            v1 = [mean_S[i, j] for i in range(n_sites) for j in range(i + 1, n_sites)
                  if chain_min_cut(i + 1, j + 1, n_sites) == 1]
            per_run.append({"phi": phi, "bound": bool(at_bound), "w0": w0, "w1": w1,
                            "disc": rt["disc"], "tau": rt["tau"],
                            "r1": cr[1][0], "r2": cr[2][0],
                            "d1": float(np.log(D) - np.mean(v1)),
                            "cgap": complement_gap(mean_S, n_sites)})
        col = lambda k: np.array([r[k] for r in per_run])
        rows.append({"D": D, "secs": time.time() - t0, "loaded": any_loaded,
                     "phi": float(col("phi").mean()), "phi_sd": float(col("phi").std()),
                     "w0": float(col("w0").mean()), "w1": float(col("w1").mean()),
                     "r1": float(col("r1").mean()), "r2": float(col("r2").mean()),
                     "d1": float(col("d1").mean()), "tau": float(col("tau").mean()),
                     "disc": [int(x) for x in col("disc")],
                     "bound": bool(col("bound").any()),
                     "cgap": float(col("cgap").max())})
    return rows


# ==========================================================================
# 5. MAIN
# ==========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draws", type=int, default=N_DRAWS)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--onecut-ds", type=int, nargs="+", default=ONECUT_DS)
    p.add_argument("--full-ds", type=int, nargs="+", default=FULL_DS)
    p.add_argument("--seeds", type=int, nargs="+", default=RUN_SEEDS)
    p.add_argument("--spot-d", type=int, default=SPOT_D)
    p.add_argument("--spot-draws", type=int, default=SPOT_DRAWS)
    p.add_argument("--ginibre-d", type=int, default=GINIBRE_D)
    p.add_argument("--ginibre-reps", type=int, default=GINIBRE_REPS)
    args = p.parse_args(argv)
    n, a = N_SITES, args.alpha
    t_start = time.time()

    print("=" * 78)
    print("RTN RUNG 8 -- is rung 7's residual deficit set by the tensor ensemble?")
    print("=" * 78)
    print("same 6-site open chain, two ensembles on PAIRED draws:")
    print("  T   = rung 7's MPS (site tensors x bond matrices T_k), from its cache")
    print("  noT = the same site tensors with T_k omitted")
    print(f"route A (per-bond deficits)  : D = {args.onecut_ds}")
    print(f"route B (full barcodes)      : D = {args.full_ds}")
    print(f"spot check (RT ordering, noT): D = {args.spot_d}, {args.spot_draws} draws")
    print(f"runs s = {args.seeds}, {args.draws} draws each; alpha = {a}\n")

    # ---- [0] validation ----
    print("[0] VALIDATION of the noT ensemble's entropies")
    w_full = 0.0
    for Dv in (2, 3, 4):
        for d in range(3):
            T = draw_tensors_noT(n, Dv, d)
            st = contract_state(T)
            ref = np.zeros((n, n))
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    ref[i - 1, j - 1] = ref[j - 1, i - 1] = interval_entropy(st, i, j, n, Dv)
            w_full = max(w_full, float(np.max(np.abs(full_entropies(T) - ref))))
    w_one = 0.0
    for Dv in (4, 8, 16):
        for d in range(3):
            T = draw_tensors_noT(n, Dv, d)
            Sf = full_entropies(T)
            ref = np.array([Sf[1, n - 1]] + [Sf[0, k - 1] for k in range(2, n)])
            w_one = max(w_one, float(np.max(np.abs(onecut_entropies(T) - ref))))
    # paired-draw check: noT site tensors must equal rung 7's for the same seed
    np.random.seed(1000 * 0 + 0)
    D0 = 4
    dims = [1] + [D0] * (n - 1) + [1]
    A_ref = [np.random.randn(dims[k], D0, dims[k + 1])
             + 1j * np.random.randn(dims[k], D0, dims[k + 1]) for k in range(n)]
    A_ref = [x / np.linalg.norm(x) for x in A_ref]
    paired = all(np.array_equal(x, y) for x, y in zip(A_ref, draw_tensors_noT(n, D0, 0)))
    ok = (w_full < 1e-10) and (w_one < 1e-10) and paired
    print(f"    all intervals vs explicit state SVD (D = 2,3,4): max |diff| = "
          f"{w_full:.1e}  {'PASS' if w_full < 1e-10 else 'FAIL'}")
    print(f"    one-cut vs all-interval route (D = 4,8,16)    : max |diff| = "
          f"{w_one:.1e}  {'PASS' if w_one < 1e-10 else 'FAIL'}")
    print(f"    noT site tensors identical to rung 7's draw   : "
          f"{'PASS' if paired else 'FAIL'} (paired comparison)")
    if not ok:
        print("\n*** VALIDATION FAILED -- stopping ***")
        return 1

    # ---- [1] random-matrix reference ----
    print(f"\n[1] RANDOM-MATRIX REFERENCE: deficit of psi = G_1 ... G_k, "
          f"D = {args.ginibre_d}, {args.ginibre_reps} reps")
    t0 = time.time()
    d1_ref, d1_se = ginibre_product_deficit(args.ginibre_d, 1, args.ginibre_reps)
    d2_ref, d2_se = ginibre_product_deficit(args.ginibre_d, 2, args.ginibre_reps)
    print(f"    k = 1 (Page, equal dimensions): delta = {d1_ref:.4f} +- {d1_se:.4f}"
          f"   (Page's analytic value: 0.5)")
    print(f"    k = 2 (two factors in a row)  : delta = {d2_ref:.4f} +- {d2_se:.4f}"
          f"   (measured; no closed form asserted)")
    print(f"    [{time.time() - t0:.1f} s]")

    # ---- [2] predictions, printed before any tensor measurement ----
    pred = {"T": {"end": d2_ref, "interior": 0.5},
            "noT": {"end": 0.5, "interior": 0.0}}
    for e in pred:
        pred[e]["mean"] = (2 * pred[e]["end"] + 3 * pred[e]["interior"]) / 5
    print("\n[2] PREDICTIONS (fixed here, before this run measures any tensors)")
    print(f"    {'ensemble':>9} {'end bonds 1,5':>14} {'interior 2-4':>13} {'mean of 5':>10}")
    for e in ("T", "noT"):
        print(f"    {e:>9} {pred[e]['end']:>14.4f} {pred[e]['interior']:>13.4f} "
              f"{pred[e]['mean']:>10.4f}")
    print(f"    pass: every fitted delta_inf within {TOLERANCE} of its prediction,")
    print("          and delta_inf(noT) < delta_inf(T)")
    print("    (set after a 10-draw probe to D = 128 had been seen: a theory")
    print("     prediction checked at full statistics, not a blind registration)")

    # ---- [3] route A ----
    results_A = {}
    for e in ("T", "noT"):
        print(f"\n[3{'a' if e == 'T' else 'b'}] ROUTE A, ensemble {e}: "
              f"per-bond delta = log D - S")
        rows = route_A(e, args.onecut_ds, args.seeds, args.draws)
        results_A[e] = rows
        print(f"    {'D':>4} {'bond 1':>8} {'bond 2':>8} {'bond 3':>8} {'bond 4':>8} "
              f"{'bond 5':>8} {'end':>8} {'interior':>9} {'mean':>8} {'sd(mean)':>9}")
        for r in rows:
            tag = "  (cache)" if r["all_loaded"] else f"  [{r['secs']:.0f} s]"
            print(f"    {r['D']:>4d} " + " ".join(f"{x:>8.4f}" for x in r["bond"])
                  + f" {r['end']:>8.4f} {r['interior']:>9.4f} {r['mean']:>8.4f} "
                    f"{r['mean_sd']:>9.5f}{tag}")

    print("\n[4] FITS: delta = delta_inf + c / D, per class, both ensembles")
    print(f"    {'ensemble':>9} {'class':>9} {'delta_inf':>10} {'+-':>8} {'R^2':>8} "
          f"{'predicted':>10} {'|diff|':>8} {'verdict':>8}")
    fits, all_pass = {}, True
    for e in ("T", "noT"):
        rows = results_A[e]
        Ds = [r["D"] for r in rows]
        fits[e] = {}
        for cls in ("end", "interior", "mean"):
            d0, se, c, r2 = fit_inf(Ds, [r[cls] for r in rows])
            diff = abs(d0 - pred[e][cls])
            good = diff <= TOLERANCE
            all_pass &= good
            fits[e][cls] = (d0, se, c, r2)
            print(f"    {e:>9} {cls:>9} {d0:>10.4f} {se:>8.4f} {r2:>8.4f} "
                  f"{pred[e][cls]:>10.4f} {diff:>8.4f} {'PASS' if good else 'FAIL':>8}")
    direction = fits["noT"]["mean"][0] < fits["T"]["mean"][0]
    all_pass &= direction
    print(f"    delta_inf(noT) < delta_inf(T): "
          f"{fits['noT']['mean'][0]:.4f} < {fits['T']['mean'][0]:.4f} -> "
          f"{'PASS' if direction else 'FAIL'}")
    print("    (fit standard errors understate the true uncertainty: 5 points,")
    print("     2 parameters; the tolerance is the operative criterion)")

    print("\n[5] SIDE BY SIDE: delta_inf, rung 7 ensemble vs rung 8 ensemble")
    print(f"    {'':>22} {'T (rung 7)':>12} {'noT (rung 8)':>14}")
    for cls, label in (("end", "end bonds 1, 5"), ("interior", "interior bonds 2-4"),
                       ("mean", "mean of 5 bonds")):
        print(f"    {label:>22} {fits['T'][cls][0]:>12.4f} {fits['noT'][cls][0]:>14.4f}")

    # ---- [6] route B ----
    print(f"\n[6] ROUTE B: full barcode comparison, D = {args.full_ds}")
    results_B = {}
    for e in ("T", "noT"):
        rows = route_B(e, args.full_ds, args.seeds, args.draws, a)
        results_B[e] = rows
        print(f"\n    ensemble {e}" + (" (rung 7's cached entropies, re-scored)"
                                      if e == "T" else ""))
        print(f"    {'D':>4} {'phi_opt':>18} {'e^(a*d1)':>9} {'W_H0':>10} {'W_H1':>10} "
              f"{'S/L c1':>7} {'S/L c2':>7} {'disc (runs)':>20}")
        for r in rows:
            tag = "" if r["loaded"] else f"  [{r['secs']:.0f} s]"
            print(f"    {r['D']:>4d} {r['phi']:>9.5f} ({r['phi_sd']:.5f}) "
                  f"{np.exp(a * r['d1']):>9.5f} {r['w0']:>10.3e} {r['w1']:>10.3e} "
                  f"{r['r1']:>7.4f} {r['r2']:>7.4f} {str(r['disc']):>20}{tag}")
        if any(r["bound"] for r in rows):
            print("    *** a phi_opt was PINNED at a search bound ***")
        print(f"    pure-state check, max |S(A) - S(A^c)|: "
              f"{max(r['cgap'] for r in rows):.1e}")

    print("\n    phi_opt, T vs noT:")
    print(f"    {'D':>4} {'T':>9} {'noT':>9} {'noT - 1':>9}")
    for rt_, rn in zip(results_B["T"], results_B["noT"]):
        print(f"    {rt_['D']:>4d} {rt_['phi']:>9.5f} {rn['phi']:>9.5f} {rn['phi'] - 1:>9.5f}")
    lower = all(rn["phi"] < rt_["phi"] for rt_, rn in zip(results_B["T"], results_B["noT"]))
    print(f"    noT phi_opt below rung 7's at every D: {'YES' if lower else 'NO'}")

    # ---- [7] spot check ----
    print(f"\n[7] SPOT CHECK: RT ordering, noT ensemble, D = {args.spot_d}, "
          f"{args.spot_draws} draws (run s = 0)")
    t0 = time.time()
    spot, was = _cached8(f"full_noT_D{args.spot_d}_s0_spot{args.spot_draws}",
                         lambda: np.array([full_entropies(draw_tensors_noT(n, args.spot_d, d))
                                           for d in range(args.spot_draws)]))
    L64 = rt_length_matrix(n, args.spot_d)
    rt64 = rt_check(spot.mean(axis=0), L64)
    per = [rt_check(S1, L64)["disc"] for S1 in spot]
    cr64 = class_ratios(spot.mean(axis=0), n, args.spot_d)
    print(f"    [{time.time() - t0:.1f} s{' (cache)' if was else ''}]")
    print(f"    Kendall tau = {rt64['tau']:.3f}, p = {rt64['p']:.2e}, "
          f"discordant = {rt64['disc']} of {rt64['strict']}; per draw {per}")
    print(f"    S/L_RT: cut 1 = {cr64[1][0]:.4f}, cut 2 = {cr64[2][0]:.4f}")

    # ---- [8] plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"T": "#c05621", "noT": "#2b6cb0"}
    ax = axes[0]
    for e in ("T", "noT"):
        Ds = [r["D"] for r in results_A[e]]
        ax.plot(Ds, [r["end"] for r in results_A[e]], "o-", color=colors[e],
                label=f"{e}: end bonds")
        ax.plot(Ds, [r["interior"] for r in results_A[e]], "s--", color=colors[e],
                label=f"{e}: interior bonds")
    for yv, lab in ((d2_ref, f"two factors {d2_ref:.3f}"), (0.5, "Page 1/2"), (0.0, "0")):
        ax.axhline(yv, color="gray", lw=0.8, ls=":")
        ax.text(args.onecut_ds[-1], yv, f" {lab}", va="center", fontsize=7, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("bond dimension $D$")
    ax.set_ylabel("deficit $\\delta = \\log D - S$ (nats)")
    ax.set_title("Per-bond deficit vs prediction", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[1]
    for e in ("T", "noT"):
        Ds = np.array([r["D"] for r in results_A[e]], float)
        ax.errorbar(Ds, [r["mean"] for r in results_A[e]],
                    yerr=[r["mean_sd"] for r in results_A[e]], fmt="o", ms=6,
                    capsize=3, color=colors[e], label=f"{e}: mean of 5 bonds")
        d0, _, c, _ = fits[e]["mean"]
        xs = np.geomspace(Ds.min(), Ds.max(), 100)
        ax.plot(xs, d0 + c / xs, "-", color=colors[e], lw=1.2,
                label=f"fit $\\delta_\\infty$={d0:.3f}")
        ax.axhline(pred[e]["mean"], color=colors[e], lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("bond dimension $D$")
    ax.set_ylabel("mean deficit (nats)")
    ax.set_title("Mean deficit; dotted = prediction", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[2]
    for e in ("T", "noT"):
        Ds = [r["D"] for r in results_B[e]]
        ax.errorbar(Ds, [r["phi"] for r in results_B[e]],
                    yerr=[r["phi_sd"] for r in results_B[e]], fmt="o-", ms=6,
                    capsize=3, color=colors[e], label=f"{e}: $\\phi_{{opt}}$")
    ax.axhline(1.0, color="k", lw=1, ls=":", label="RT prediction $\\phi=1$")
    ax.set_xscale("log")
    ax.set_xlabel("bond dimension $D$")
    ax.set_ylabel("$\\phi_{opt}$")
    ax.set_title("$\\phi_{opt}$: rung 7 ensemble vs rung 8", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.suptitle("RTN rung 8: the residual deficit, bond by bond", fontsize=11)
    fig.tight_layout()
    png = HERE / "rtn_rung8_convergence.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"\n  wrote {png}")

    # ---- [9] verdict ----
    print("\n[9] VERDICT")
    print(f"  predictions met within {TOLERANCE} (all classes, both ensembles) "
          f"and noT < T: {'YES' if all_pass else 'NO'}")
    print(f"  delta_inf, mean of 5 bonds: T = {fits['T']['mean'][0]:.4f} "
          f"(predicted {pred['T']['mean']:.4f}), noT = {fits['noT']['mean'][0]:.4f} "
          f"(predicted {pred['noT']['mean']:.4f})")
    print(f"  phi_opt at D = {args.full_ds[-1]}: T = {results_B['T'][-1]['phi']:.4f}, "
          f"noT = {results_B['noT'][-1]['phi']:.4f}")
    print(f"  RT ordering, noT, D = {args.spot_d}: discordant = {rt64['disc']}")
    print(f"\n  total runtime {time.time() - t_start:.0f} s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
