"""Diagnostics on the min-beta/C cell (N=18, beta=5, declared C=0.1805).

1. H0 prediction check: 23 finite bars all [0, 1/Ghat(1)] per side; persim
   (p=1, L2 ground metric, sum of matched costs) should give
   W_H0 = 23*|1/Ghat_syk(1) - 1/Ghat_er(1)| if direct matching is optimal.
   Verified against actual persim output per realization, plus an explicit
   direct-vs-diagonal cost comparison.
2. Reachability: Ghat_Schwarzian(1) as a function of beta/C over ~4 orders
   of magnitude; is it bounded below by the conformal value and monotone?
3. UV dominance: (a) n_tau = 12/24/48 sweep; (b) filtration restricted to
   m >= 2 and m >= 3 (short separations clamped to enter at m_min).
Diagnostic only — nothing here changes any convention. Numbers only.
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/naman/projects/ER-EPR")
sys.path.insert(0, "/Users/naman/projects/ER-EPR/rung4")

import syk_model
from rung4 import circle_distance_matrix, declared_C
from schwarzian import compute_G
from ripser import ripser
from persim import wasserstein

N, BETA = 18, 5.0
C_DECL = declared_C(N)
R = 10
NTAUS = [12, 24, 48]
DELTA = 0.25

print(f"Cell: N={N}, beta={BETA}, declared C={C_DECL:.6f}, "
      f"beta/C={BETA/C_DECL:.2f}")


def run_ph(D):
    return ripser(D, distance_matrix=True, maxdim=1)["dgms"]


def finite_part(dgm):
    return dgm[np.isfinite(dgm).all(axis=1)] if dgm.size else dgm


def w01(dg1, dg2):
    return (float(wasserstein(finite_part(dg1[0]), finite_part(dg2[0]))),
            float(wasserstein(finite_part(dg1[1]), finite_part(dg2[1]))))


def conformal_profile(n_tau):
    m = np.arange(1, n_tau // 2 + 1)
    return np.sin(np.pi * m / n_tau) ** (-2.0 * DELTA)


def syk_profiles():
    """One H build per realization; G evaluated on all three tau grids."""
    out = {n: np.zeros((R, n // 2)) for n in NTAUS}
    for r in range(R):
        np.random.seed(1000 * N + r)
        H = syk_model.build_syk_hamiltonian(N, J=1.0)
        E, V = np.linalg.eigh(H)
        W = np.zeros((E.size, E.size))
        for chi in syk_model.generate_majoranas(N):
            M = V.conj().T @ chi @ V
            W += M.real ** 2 + M.imag ** 2
        W /= N
        x = E - E[0]
        Z = float(np.exp(-BETA * x).sum())
        for n_tau in NTAUS:
            for m in range(1, n_tau // 2 + 1):
                tau = m * BETA / n_tau
                a = np.exp(-(BETA - tau) * x)
                b = np.exp(-tau * x)
                out[n_tau][r, m - 1] = float(a @ W @ b) / Z
        print(f"  syk r={r} done", flush=True)
    return out


def er_profile(n_tau, C):
    prof = np.zeros(n_tau // 2)
    worst = 0.0
    for m in range(1, n_tau // 2 + 1):
        res = compute_G(m * BETA / n_tau, BETA, C)
        prof[m - 1] = res["G"]
        worst = max(worst, res["conv_err"])
    return prof, worst


print("\n[SYK profiles: one Hamiltonian per realization, three tau grids]",
      flush=True)
syk = syk_profiles()

# ---------------------------------------------------------------- item 1
print("\n=== 1. H0 PREDICTION CHECK (n_tau=24) ===", flush=True)
er24, conv24 = er_profile(24, C_DECL)
print(f"declared-C profile conv_err = {conv24:.1e}")
dgm_er24 = run_ph(circle_distance_matrix(er24, 24))
h0_er = finite_part(dgm_er24[0])
ghat_er1 = er24[0] / er24[-1]
print(f"ER H0: {len(h0_er)} finite bars; births all 0: "
      f"{bool(np.all(h0_er[:, 0] == 0))}; "
      f"deaths all equal: max spread {np.ptp(h0_er[:, 1]):.2e}; "
      f"death = {h0_er[0, 1]:.6f}, 1/Ghat_er(1) = {1/ghat_er1:.6f}")
max_dev, max_diag_margin = 0.0, -np.inf
for r in range(R):
    dg = run_ph(circle_distance_matrix(syk[24][r], 24))
    h0_s = finite_part(dg[0])
    ghat_s1 = syk[24][r][0] / syk[24][r][-1]
    pred = 23.0 * abs(1 / ghat_s1 - 1 / ghat_er1)
    actual, _ = w01(dg, dgm_er24)
    max_dev = max(max_dev, abs(actual - pred))
    # direct cost per pair vs diagonal-deletion cost per pair
    direct = abs(1 / ghat_s1 - 1 / ghat_er1)
    diag = (1 / ghat_s1 + 1 / ghat_er1) / np.sqrt(2.0)
    max_diag_margin = max(max_diag_margin, direct - diag)
print(f"W_H0 actual vs 23*|1/Ghat_s(1) - 1/Ghat_er(1)|: "
      f"max |dev| over 10 realizations = {max_dev:.3e}")
print(f"direct-vs-diagonal: max over realizations of "
      f"(direct - diagonal) cost = {max_diag_margin:.4f} "
      f"(negative = direct matching always cheaper)")

# ---------------------------------------------------------------- item 2
print("\n=== 2. REACHABILITY: Ghat_Schwarzian(1) vs beta/C ===", flush=True)
tau1, tauh = BETA / 24, BETA / 2
conf1 = float(np.sin(np.pi / 24) ** (-2 * DELTA) / np.sin(np.pi / 2) ** (-2 * DELTA))
print(f"conformal (C->inf) Ghat(1) = {conf1:.6f}; "
      f"SYK mean Ghat(1) = {np.mean(syk[24][:, 0] / syk[24][:, -1]):.6f}")
print(f"{'C':>10} {'beta/C':>10} {'Ghat(1)':>10} {'conv_err':>9}")
sweep = [200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0,
         C_DECL * 5, 0.5, C_DECL, C_DECL * 0.2]
vals = []
for C in sorted(sweep, reverse=True):
    r1 = compute_G(tau1, BETA, C)
    rh = compute_G(tauh, BETA, C)
    g1 = r1["G"] / rh["G"]
    vals.append((BETA / C, g1))
    print(f"{C:>10.4f} {BETA/C:>10.3f} {g1:>10.6f} "
          f"{max(r1['conv_err'], rh['conv_err']):>9.1e}", flush=True)
bocs, g1s = zip(*vals)
mono = bool(np.all(np.diff(g1s) > 0))
print(f"monotone increasing in beta/C: {mono}; "
      f"min Ghat(1) over sweep = {min(g1s):.6f} "
      f"(conformal {conf1:.6f}; SYK 1.30)")

# ---------------------------------------------------------------- item 3a
print("\n=== 3a. n_tau SWEEP (12 / 24 / 48), declared C + conformal ===",
      flush=True)
for n_tau in NTAUS:
    if n_tau == 24:
        er, conv = er24, conv24
    else:
        er, conv = er_profile(n_tau, C_DECL)
    dgm_er = run_ph(circle_distance_matrix(er, n_tau))
    dgm_cf = run_ph(circle_distance_matrix(conformal_profile(n_tau), n_tau))
    tau_min = BETA / n_tau
    W0e, W1e, W0c, W1c, births = [], [], [], [], []
    for r in range(R):
        dg = run_ph(circle_distance_matrix(syk[n_tau][r], n_tau))
        births.append(float(dg[1][0][0]) if len(dg[1]) else np.nan)
        a, b = w01(dg, dgm_er)
        W0e.append(a); W1e.append(b)
        a, b = w01(dg, dgm_cf)
        W0c.append(a); W1c.append(b)
    b_er = dgm_er[1][0] if len(dgm_er[1]) else (np.nan, np.nan)
    print(f"n_tau={n_tau:>2} (tau_min={tau_min:.4f}, n-1={n_tau-1} H0 pairs, "
          f"H1 death at m={n_tau//3}): conv_err={conv:.1e}")
    print(f"   ER H1 bar = [{b_er[0]:.6f}, {b_er[1]:.6f}]; "
          f"SYK H1 birth mean = {np.nanmean(births):.6f}")
    print(f"   vs declared:  W_H0 = {np.mean(W0e):8.4f} +- {np.std(W0e):.4f}   "
          f"W_H1 = {np.mean(W1e):.4f} +- {np.std(W1e):.4f}")
    print(f"   vs conformal: W_H0 = {np.mean(W0c):8.4f} +- {np.std(W0c):.4f}   "
          f"W_H1 = {np.mean(W1c):.4f} +- {np.std(W1c):.4f}")

# ---------------------------------------------------------------- item 3b
print("\n=== 3b. FILTRATION RESTRICTED: m >= m_min (n_tau=24, clamped) ===",
      flush=True)


def clamped_D(prof, n_tau, m_min):
    p = prof.copy()
    p[:m_min - 1] = p[m_min - 1]     # short seps enter with m_min
    return circle_distance_matrix(p, n_tau)


dgm_cf24 = run_ph(circle_distance_matrix(conformal_profile(24), 24))
for m_min in (1, 2, 3):
    dgm_er = run_ph(clamped_D(er24, 24, m_min))
    dgm_cf = run_ph(clamped_D(conformal_profile(24), 24, m_min))
    W0e, W1e, W0c, W1c = [], [], [], []
    for r in range(R):
        dg = run_ph(clamped_D(syk[24][r], 24, m_min))
        a, b = w01(dg, dgm_er)
        W0e.append(a); W1e.append(b)
        a, b = w01(dg, dgm_cf)
        W0c.append(a); W1c.append(b)
    b_er = dgm_er[1][0] if len(dgm_er[1]) else (np.nan, np.nan)
    print(f"m_min={m_min}: ER H1 bar = [{b_er[0]:.6f}, {b_er[1]:.6f}]")
    print(f"   vs declared:  W_H0 = {np.mean(W0e):8.4f} +- {np.std(W0e):.4f}   "
          f"W_H1 = {np.mean(W1e):.4f} +- {np.std(W1e):.4f}")
    print(f"   vs conformal: W_H0 = {np.mean(W0c):8.4f} +- {np.std(W0c):.4f}   "
          f"W_H1 = {np.mean(W1c):.4f} +- {np.std(W1c):.4f}")
