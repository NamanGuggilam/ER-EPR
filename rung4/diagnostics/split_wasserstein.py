"""Retroactive H0/H1 split of the committed compare table, per cell.

The committed run and the 2026-08-29 rescore recorded only W_H0 + W_H1
totals; no diagrams or profiles were cached. This recomputes the profiles
through the identical code path (proven to reproduce the committed totals
to 4.9e-5), then reports per cell:
  - beta/C for the declared dictionary
  - W_H0 and W_H1 SEPARATELY (mean over realizations) for every tag
  - H1 bar counts on both sides; empty-H1 cells marked
  - regression: recombined W_H0+W_H1 vs the committed table
  - monotonicity of every Ghat profile in separation m (min forward step)
  - test of the circulant prediction: every H1 diagram should be exactly
    one bar [1/Ghat(1), 1/Ghat(8)] (n_tau=24, death at k = n_tau/3)
Profiles and diagrams are cached to npz/pkl for future rescoring.
Numbers only; no interpretation.
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "/Users/naman/projects/ER-EPR")
sys.path.insert(0, "/Users/naman/projects/ER-EPR/rung4")

from concurrent.futures import ProcessPoolExecutor, as_completed
from rung4 import (_one_corr_realization, schwarzian_G_profile,
                   conformal_G_profile, circle_distance_matrix, declared_C,
                   BETA_SWEEP, N_TAU, R_CORR, C_CONTROL_FACTORS, MAX_WORKERS)
from ripser import ripser
from persim import wasserstein

N_VALUES = [12, 14, 16, 18]
TAGS = ["declared", "x5", "x0.2", "conformal"]
CACHE = "/private/tmp/claude-501/-Users-naman/d2153a54-68ef-4744-a61e-58afea02c44d/scratchpad/profiles_cache.pkl"

COMMITTED_TOTAL = {  # (N, beta) -> tag -> committed mean W_H0+W_H1
    (12, 5): {"declared": 11.2241, "x5": 10.2582, "x0.2": 13.3563, "conformal": 9.9296},
    (12, 10): {"declared": 7.0658, "x5": 5.5892, "x0.2": 9.4458, "conformal": 4.9758},
    (12, 20): {"declared": 4.1478, "x5": 2.1545, "x0.2": 6.4368, "conformal": 1.0686},
    (12, 40): {"declared": 3.1152, "x5": 1.0104, "x0.2": 5.0270, "conformal": 1.2540},
    (14, 5): {"declared": 10.8774, "x5": 10.0111, "x0.2": 12.9166, "conformal": 9.7265},
    (14, 10): {"declared": 6.8214, "x5": 5.4644, "x0.2": 9.1742, "conformal": 4.9287},
    (14, 20): {"declared": 3.3265, "x5": 1.4440, "x0.2": 5.6639, "conformal": 0.6148},
    (14, 40): {"declared": 0.9412, "x5": 1.6668, "x0.2": 2.6472, "conformal": 3.2919},
    (16, 5): {"declared": 10.8084, "x5": 10.0229, "x0.2": 12.7589, "conformal": 9.7718},
    (16, 10): {"declared": 6.7863, "x5": 5.5311, "x0.2": 9.1017, "conformal": 5.0555},
    (16, 20): {"declared": 3.1543, "x5": 1.3701, "x0.2": 5.5215, "conformal": 0.6308},
    (16, 40): {"declared": 0.8569, "x5": 1.4872, "x0.2": 2.8405, "conformal": 2.9667},
    (18, 5): {"declared": 10.6893, "x5": 9.9707, "x0.2": 12.5566, "conformal": 9.7461},
    (18, 10): {"declared": 6.8092, "x5": 5.6416, "x0.2": 9.0818, "conformal": 5.2139},
    (18, 20): {"declared": 3.5143, "x5": 1.8154, "x0.2": 5.8981, "conformal": 1.0343},
    (18, 40): {"declared": 1.6177, "x5": 0.5786, "x0.2": 3.7768, "conformal": 1.9117},
}


def run_ph(D):
    return ripser(D, distance_matrix=True, maxdim=1)["dgms"]


def finite_part(dgm):
    return dgm[np.isfinite(dgm).all(axis=1)] if dgm.size else dgm


def main():
    betas = list(BETA_SWEEP)

    print("[1/3] SYK G(tau) profiles", flush=True)
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

    print("[2/3] Schwarzian profiles", flush=True)
    er_G = {}
    conv_errs = {}
    for N in N_VALUES:
        C0 = declared_C(N)
        for beta in betas:
            for tag, C in ([("declared", C0)]
                           + [(f"x{f:g}", C0 * f) for f in C_CONTROL_FACTORS]):
                prof, conv = schwarzian_G_profile(beta, C, N_TAU)
                assert conv < 1e-6
                er_G[(N, beta, tag)] = prof
                conv_errs[(N, beta, tag)] = conv
            er_G[(N, beta, "conformal")] = conformal_G_profile(beta, N_TAU)
            conv_errs[(N, beta, "conformal")] = 0.0
            print(f"  N={N} beta={beta:g} done", flush=True)

    with open(CACHE, "wb") as fh:
        pickle.dump({"syk_G": syk_G, "er_G": er_G, "conv_errs": conv_errs,
                     "n_tau": N_TAU, "betas": betas}, fh)
    print(f"profiles cached to {CACHE}", flush=True)

    print("[3/3] split scoring + structure checks", flush=True)
    # monotonicity of every profile (Ghat decreasing in m <=> G decreasing)
    min_step_syk, min_step_er = np.inf, np.inf
    violations = 0
    for N in N_VALUES:
        for bi in range(len(betas)):
            for r in range(R_CORR):
                steps = syk_G[N][r, bi][:-1] - syk_G[N][r, bi][1:]
                min_step_syk = min(min_step_syk, float(steps.min()))
                violations += int((steps <= 0).any())
    for key, prof in er_G.items():
        steps = prof[:-1] - prof[1:]
        min_step_er = min(min_step_er, float(steps.min()))
        violations += int((steps <= 0).any())
    print(f"\nMONOTONICITY (G strictly decreasing in separation m):")
    print(f"  min forward step, SYK profiles (160): {min_step_syk:.3e}")
    print(f"  min forward step, ER profiles (64):   {min_step_er:.3e}")
    print(f"  violations: {violations}")

    # circulant one-bar prediction: H1 = [1/Ghat(1), 1/Ghat(8)] exactly
    def h1_pred(prof):
        ghat = prof / prof[-1]
        return 1.0 / ghat[0], 1.0 / ghat[7]      # m=1 birth, m=8 death (k/n=1/3)

    max_pred_dev, extra_bars, empty_h1 = 0.0, 0, 0
    dgm_cache = {}

    def get_dgm(prof, key):
        if key not in dgm_cache:
            dgm_cache[key] = run_ph(circle_distance_matrix(prof, N_TAU))
        return dgm_cache[key]

    header = (f"{'N':>3} {'beta':>5} {'b/C':>6} | "
              f"{'W0 dec':>8} {'W1 dec':>8} | {'W0 x5':>8} {'W1 x5':>8} | "
              f"{'W0 /5':>8} {'W1 /5':>8} | {'W0 conf':>8} {'W1 conf':>8} | "
              f"H1 syk | H1 er | reg")
    print("\nSPLIT TABLE (means over realizations; H1 counts: syk = "
          "counts over 10 realizations, er = declared/x5/x0.2/conf)")
    print(header)
    print("-" * len(header))
    worst_reg = 0.0
    for N in N_VALUES:
        boc = {beta: beta / declared_C(N) for beta in betas}
        for bi, beta in enumerate(betas):
            dgms_er = {t: get_dgm(er_G[(N, beta, t)], ("er", N, beta, t))
                       for t in TAGS}
            W0 = {t: [] for t in TAGS}
            W1 = {t: [] for t in TAGS}
            syk_counts = []
            for r in range(R_CORR):
                dg = get_dgm(syk_G[N][r, bi], ("syk", N, beta, r))
                syk_counts.append(len(dg[1]))
                for t in TAGS:
                    W0[t].append(float(wasserstein(finite_part(dg[0]),
                                                   finite_part(dgms_er[t][0]))))
                    W1[t].append(float(wasserstein(finite_part(dg[1]),
                                                   finite_part(dgms_er[t][1]))))
            # prediction + emptiness checks on every diagram of this cell
            for r in range(R_CORR):
                dg = dgm_cache[("syk", N, beta, r)]
                if len(dg[1]) == 0:
                    empty_h1 += 1
                elif len(dg[1]) > 1:
                    extra_bars += 1
                else:
                    pb, pd = h1_pred(syk_G[N][r, bi])
                    max_pred_dev = max(max_pred_dev,
                                       abs(dg[1][0][0] - pb),
                                       abs(dg[1][0][1] - pd))
            for t in TAGS:
                dg = dgms_er[t]
                if len(dg[1]) == 0:
                    empty_h1 += 1
                elif len(dg[1]) > 1:
                    extra_bars += 1
                else:
                    pb, pd = h1_pred(er_G[(N, beta, t)])
                    max_pred_dev = max(max_pred_dev,
                                       abs(dg[1][0][0] - pb),
                                       abs(dg[1][0][1] - pd))
            m0 = {t: float(np.mean(W0[t])) for t in TAGS}
            m1 = {t: float(np.mean(W1[t])) for t in TAGS}
            reg = max(abs(m0[t] + m1[t] - COMMITTED_TOTAL[(N, beta)][t])
                      for t in TAGS)
            worst_reg = max(worst_reg, reg)
            er_counts = "/".join(str(len(dgms_er[t][1])) for t in TAGS)
            sc = (str(syk_counts[0]) if len(set(syk_counts)) == 1
                  else str(sorted(set(syk_counts))))
            print(f"{N:>3} {beta:>5g} {boc[beta]:>6.1f} | "
                  f"{m0['declared']:8.4f} {m1['declared']:8.4f} | "
                  f"{m0['x5']:8.4f} {m1['x5']:8.4f} | "
                  f"{m0['x0.2']:8.4f} {m1['x0.2']:8.4f} | "
                  f"{m0['conformal']:8.4f} {m1['conformal']:8.4f} | "
                  f"{sc:>6} | {er_counts:>7} | {reg:.1e}")

    print(f"\nregression: worst |(W0+W1) - committed| over all cells/tags: "
          f"{worst_reg:.2e}  (committed table rounded to 4 dp)")
    print(f"H1 diagrams that are EMPTY: {empty_h1}")
    print(f"H1 diagrams with MORE than one bar: {extra_bars}")
    print(f"one-bar prediction [1/Ghat(1), 1/Ghat(8)]: "
          f"max |actual - predicted| coordinate = {max_pred_dev:.3e}")
    print(f"worst Schwarzian conv_err across all profiles: "
          f"{max(conv_errs.values()):.3e}")

    with open(CACHE.replace("profiles_cache", "diagram_cache"), "wb") as fh:
        pickle.dump(dgm_cache, fh)
    print("diagrams cached.", flush=True)


if __name__ == "__main__":
    main()
