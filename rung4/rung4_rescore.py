"""Convention-robustness rescore of the rung-4 compare stage.

Recomputes the exact compare-run G profiles (same seeds, same quadratures),
then scores the pre-declared criterion under BOTH distance conventions:

  A. d = 1/Ghat, Ghat = G/G(beta/2)     — the compare run's convention.
     Reproducing the committed table (d4a4c5a, rung4_results.txt) is the
     regression check that this rescore is faithful.
  B. d_ij = log(gmax / G_ij)            — Check C's anchored log convention
     (rung4_checks.py distance_matrix). Shift-invariant in G, so no gauge
     choice is needed on either side.

Pre-declared criterion, unchanged: favored at (N, beta) iff
mean_r[W_H0 + W_H1](declared) < that of x5 AND of x0.2.
"""
import sys
from pathlib import Path

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import rung4
from rung4 import (circle_distance_matrix, schwarzian_G_profile,
                   conformal_G_profile, declared_C, _one_corr_realization,
                   BETA_SWEEP, N_TAU, R_CORR, C_CONTROL_FACTORS, MAX_WORKERS)

from ripser import ripser
from persim import wasserstein

N_VALUES = [12, 14, 16, 18]
TAGS = ["declared"] + [f"x{f:g}" for f in C_CONTROL_FACTORS] + ["conformal"]

# committed 1/Ghat table (rung4_results.txt, d4a4c5a) for the regression check:
# (N, beta) -> (W_declared, W_x5, W_x0.2, W_conf, favored)
COMMITTED = {
    (12, 5): (11.2241, 10.2582, 13.3563, 9.9296, False),
    (12, 10): (7.0658, 5.5892, 9.4458, 4.9758, False),
    (12, 20): (4.1478, 2.1545, 6.4368, 1.0686, False),
    (12, 40): (3.1152, 1.0104, 5.0270, 1.2540, False),
    (14, 5): (10.8774, 10.0111, 12.9166, 9.7265, False),
    (14, 10): (6.8214, 5.4644, 9.1742, 4.9287, False),
    (14, 20): (3.3265, 1.4440, 5.6639, 0.6148, False),
    (14, 40): (0.9412, 1.6668, 2.6472, 3.2919, True),
    (16, 5): (10.8084, 10.0229, 12.7589, 9.7718, False),
    (16, 10): (6.7863, 5.5311, 9.1017, 5.0555, False),
    (16, 20): (3.1543, 1.3701, 5.5215, 0.6308, False),
    (16, 40): (0.8569, 1.4872, 2.8405, 2.9667, True),
    (18, 5): (10.6893, 9.9707, 12.5566, 9.7461, False),
    (18, 10): (6.8092, 5.6416, 9.0818, 5.2139, False),
    (18, 20): (3.5143, 1.8154, 5.8981, 1.0343, False),
    (18, 40): (1.6177, 0.5786, 3.7768, 1.9117, False),
}


def log_distance_matrix(G_prof, n_tau):
    """Check C's convention, on the full tau circle: d = log(gmax/G(sep)),
    gmax = largest off-diagonal correlator. Shift-invariant under G -> cG."""
    gmax = float(np.max(G_prof))
    i = np.arange(n_tau)
    m = np.abs(i[:, None] - i[None, :])
    m = np.minimum(m, n_tau - m)
    D = np.zeros((n_tau, n_tau))
    nz = m > 0
    D[nz] = np.log(gmax / G_prof[m[nz] - 1])
    return D


def run_ph(D):
    return ripser(D, distance_matrix=True, maxdim=1)["dgms"]


def finite_part(dgm):
    return dgm[np.isfinite(dgm).all(axis=1)] if dgm.size else dgm


def w_tot(dg1, dg2):
    return (float(wasserstein(finite_part(dg1[0]), finite_part(dg2[0])))
            + float(wasserstein(finite_part(dg1[1]), finite_part(dg2[1]))))


def main():
    betas = list(BETA_SWEEP)
    print(f"Rescore: N={N_VALUES}, betas={betas}, n_tau={N_TAU}, "
          f"R={R_CORR}, tags={TAGS}", flush=True)

    print("\n[1/3] SYK G(tau) profiles (seeded rebuild, identical to compare run)",
          flush=True)
    syk_G = {}
    for N in N_VALUES:
        results = [None] * R_CORR
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(_one_corr_realization, (N, r, betas, N_TAU)): r
                    for r in range(R_CORR)}
            for fut in as_completed(futs):
                r, E, G, rowsum_err = fut.result()
                assert rowsum_err < 1e-10
                results[r] = G
        syk_G[N] = np.array(results)
        print(f"  N={N} done", flush=True)

    print("\n[2/3] Schwarzian G(tau) profiles at declared C + controls", flush=True)
    er_G = {}
    for N in N_VALUES:
        C0 = declared_C(N)
        for beta in betas:
            for tag, C in ([("declared", C0)]
                           + [(f"x{f:g}", C0 * f) for f in C_CONTROL_FACTORS]):
                prof, conv = schwarzian_G_profile(beta, C, N_TAU)
                assert conv < 1e-6, f"conv_err {conv:.1e} N={N} beta={beta} {tag}"
                er_G[(N, beta, tag)] = prof
            er_G[(N, beta, "conformal")] = conformal_G_profile(beta, N_TAU)
            print(f"  N={N} beta={beta:g} done", flush=True)

    print("\n[3/3] Scoring both conventions", flush=True)
    conventions = {"invGhat": circle_distance_matrix, "log": log_distance_matrix}
    means = {c: {} for c in conventions}   # conv -> (N,beta) -> tag -> mean W
    stds = {c: {} for c in conventions}
    fav = {c: {} for c in conventions}
    for N in N_VALUES:
        for bi, beta in enumerate(betas):
            for cname, dmat in conventions.items():
                dgms_er = {t: run_ph(dmat(er_G[(N, beta, t)], N_TAU))
                           for t in TAGS}
                W = {t: [] for t in TAGS}
                for r in range(R_CORR):
                    dg_syk = run_ph(dmat(syk_G[N][r, bi], N_TAU))
                    for t in TAGS:
                        W[t].append(w_tot(dg_syk, dgms_er[t]))
                m = {t: float(np.mean(W[t])) for t in TAGS}
                means[cname][(N, beta)] = m
                stds[cname][(N, beta)] = {t: float(np.std(W[t])) for t in TAGS}
                fav[cname][(N, beta)] = (m["declared"] < m["x5"]
                                         and m["declared"] < m["x0.2"])
            print(f"  N={N} beta={beta:g} scored", flush=True)

    print("\n=== REGRESSION CHECK: 1/Ghat rescore vs committed table ===",
          flush=True)
    worst = 0.0
    flags_ok = True
    for key, (wd, w5, w02, wc, f) in COMMITTED.items():
        m = means["invGhat"][key]
        worst = max(worst,
                    abs(m["declared"] - wd), abs(m["x5"] - w5),
                    abs(m["x0.2"] - w02), abs(m["conformal"] - wc))
        flags_ok &= (fav["invGhat"][key] == f)
    print(f"max |mean W - committed| over all cells/tags: {worst:.4e} "
          f"(table printed to 4 dp, so <= 5e-5 = exact)")
    print(f"favored flags identical to committed run: {flags_ok}")

    print("\n=== PER-CELL COMPARISON (criterion: declared < x5 AND < /5) ===")
    print(f"{'N':>3} {'beta':>5} | {'1/Ghat: Wd':>10} {'Wx5':>8} {'W/5':>8} "
          f"{'fav':>4} | {'log: Wd':>8} {'Wx5':>8} {'W/5':>8} {'fav':>4} | flip?")
    print("-" * 100)
    flips = []
    for N in N_VALUES:
        for beta in betas:
            a, b = means["invGhat"][(N, beta)], means["log"][(N, beta)]
            fa, fb = fav["invGhat"][(N, beta)], fav["log"][(N, beta)]
            flip = "FLIP" if fa != fb else ""
            if fa != fb:
                flips.append((N, beta, fa, fb))
            print(f"{N:>3} {beta:>5g} | {a['declared']:10.4f} {a['x5']:8.4f} "
                  f"{a['x0.2']:8.4f} {'YES' if fa else 'no':>4} | "
                  f"{b['declared']:8.4f} {b['x5']:8.4f} {b['x0.2']:8.4f} "
                  f"{'YES' if fb else 'no':>4} | {flip}")

    na = sum(fav["invGhat"].values())
    nb = sum(fav["log"].values())
    print(f"\nfavored count: 1/Ghat {na}/16   log {nb}/16")
    if flips:
        for N, beta, fa, fb in flips:
            print(f"  FLIP at (N={N}, beta={beta:g}): "
                  f"1/Ghat={'YES' if fa else 'no'} -> log={'YES' if fb else 'no'}")
    else:
        print("  no cells flip: identical favored set under both conventions")

    # spread context for the log-convention favored cells
    print("\nlog-convention declared W mean +- std (per-realization spread):")
    for N in N_VALUES:
        for beta in betas:
            m, s = means["log"][(N, beta)], stds["log"][(N, beta)]
            print(f"  N={N} beta={beta:g}: {m['declared']:.4f} "
                  f"+- {s['declared']:.4f}"
                  + ("   [FAVORED]" if fav["log"][(N, beta)] else ""))


if __name__ == "__main__":
    main()
