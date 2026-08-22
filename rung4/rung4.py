"""
ER=EPR Project — Rung 4: SYK vs the exact Schwarzian at physically
matched parameters
========================================================================

Rung 3's ER side was a frozen classical JT slice: 1-dimensional,
temperature-trivial, provably H1-free — a structurally lopsided
comparison. Rung 4 replaces it with the exact Schwarzian boundary
two-point function (rung4/schwarzian.py, validated by Checks A-E), and
reindexes BOTH sides by Euclidean time on the thermal circle.

The load-bearing design choice (approved before any code): the two
sides' parameters must correspond PHYSICALLY. The ER side is controlled
by beta/C; the SYK side by (beta, N) — C is not an input but an
emergent low-energy coupling of SYK. So before any barcode comparison,
this script extracts C from SYK's own spectrum, two independent ways:

  Route 1 (primary) — low-energy density of states. The Schwarzian DOS
    is rho(E) ∝ sinh(2 pi sqrt(2 C (E - E_edge))) — the SAME
    rho(k) = k sinh(2 pi k) validated on the ER side by Check A, under
    E = k^2/2C. IDENTIFIABILITY (learned from the smoke test): a direct
    nonlinear fit of the cumulative count is degenerate near the edge —
    for u = 2 pi sqrt(2 C eps) << 1 the count is ∝ A sqrt(C) eps^{3/2},
    so A and C ride a perfect ridge. The identifiable form is the
    LINEARIZED fit: in the sinh-dominated regime
    ln Ncum(eps) ~ 2 pi sqrt(2 C eps) + const, so a linear fit of
    ln Ncum vs sqrt(eps) gives C = slope^2/(8 pi^2) with the prefactor
    absorbed entirely into the intercept. C counts as MEASURED only if
    the slope is stable across quantile bands (a plateau); a drifting
    slope means the edge is not sinh-form and no C exists to extract.

  Route 2 (cross-check) — thermodynamics: the linear-in-T coefficient
    of the entropy (the user's original formulation). Schwarzian:
    S(beta) = const - (3/2) ln beta + 4 pi^2 C / beta, so
    y = S + (3/2) ln beta is LINEAR in 1/beta with slope 4 pi^2 C.
    Entropy is invariant under energy shifts, so the edge-energy
    nuisance parameter of an E(beta) fit (which buried the C signal
    under the universal 3/(2 beta) term — also a smoke-test lesson)
    drops out identically. A negative fitted C is data, not an
    artifact, and is reported as such.

  GOE CONTROL (discriminating power, same logic as Check A's
    wrong-power controls): both routes are also run on a GOE ensemble
    of the same dimension and R, rescaled to the same mean bandwidth.
    A generic random-matrix edge has rho ∝ sqrt(eps), which produces
    the SAME (3/2) ln T entropy term as the small-u Schwarzian — so
    resembling the Schwarzian at that level proves nothing. If SYK's
    fit pattern is indistinguishable from GOE's, the extraction has
    detected only generic RMT edge behavior and matched parameters are
    NOT established.

  GATE (hard checkpoint, reviewed by the user before the comparison
  stage runs): (a) Routes 1 and 2 agree within 15% at each N;
  (b) C(N) is linear in N with positive slope; (c) magnitudes are
  reported; (d) SYK separates from the GOE control. If the routes
  disagree at every accessible N, the honest outcome is "could not
  establish matched parameters at accessible N" — no averaging or
  fudging a disagreement into a usable C.

The SYK machinery is imported from the validated syk_model.py
(generate_majoranas, build_syk_hamiltonian, variance 6 J^2/N^3) — not
rebuilt, not re-derived. syk_model draws couplings from the global
NumPy RNG, so realizations are seeded externally: realization r of
size N uses np.random.seed(1000*N + r).

Usage:
    python rung4/rung4.py extract [--smoke]
    (the 'compare' stage is gated on user review of the C(N) results)
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # rung4/schwarzian.py
sys.path.insert(0, str(HERE.parent))   # repo root: syk_model.py

import syk_model  # noqa: E402  (validated builder — reused, never rebuilt)
from schwarzian import compute_G  # noqa: E402

DATA_DIR = HERE / "data"

# --- ensemble design -------------------------------------------------
N_VALUES = [12, 14, 16, 18, 20]
N_REALIZATIONS = {12: 50, 14: 50, 16: 30, 18: 20, 20: 10}
J = 1.0
MAX_WORKERS = 4

# --- Route 1 (DOS) design --------------------------------------------
DOS_BANDS = [(0.02, 0.10), (0.05, 0.15), (0.10, 0.25),
             (0.15, 0.35), (0.25, 0.45)]     # quantile bands of pooled edge
DOS_HEADLINE_BAND = (0.10, 0.25)
DOS_PLATEAU_TOL = 1.25                       # max/min band-C ratio to count as measured

# --- Route 2 (entropy) design ----------------------------------------
BETA_WINDOWS = [(3.0, 8.0), (6.0, 15.0), (10.0, 25.0)]
BETA_HEADLINE = (6.0, 15.0)

N_BOOT = 200
ROUTE_AGREE_TOL = 0.15


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        self.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def progress(msg):
    print(f"  ... {msg}")


# ======================================================================
# SECTION 1 helpers — verification of imported machinery
# ======================================================================
def verify_syk_model(N=12, n_pairs=10, tol=1e-12):
    """{chi_i, chi_j} = delta_ij * I to machine precision, on the
    imported syk_model.generate_majoranas — the same check rung 3 ran on
    its own construction."""
    chis = syk_model.generate_majoranas(N)
    dim = chis[0].shape[0]
    ident = np.eye(dim, dtype=complex)
    rng = np.random.default_rng(0)
    max_err = 0.0
    pairs = {(i, i) for i in range(0, N, 3)}
    while len(pairs) < n_pairs + 4:
        i, j = rng.integers(0, N, size=2)
        pairs.add((min(i, j), max(i, j)))
    for i, j in pairs:
        anti = chis[i] @ chis[j] + chis[j] @ chis[i]
        target = ident if i == j else 0.0
        max_err = max(max_err, float(np.max(np.abs(anti - target))))
    return max_err, dim


# ======================================================================
# SECTION 2 helpers — disorder ensemble (parallel, cached)
# ======================================================================
def _one_realization(args):
    """Worker: build one seeded SYK realization via the imported builder
    and return its spectrum. Seeding is external because syk_model uses
    the global RNG by design."""
    N, r = args
    np.random.seed(1000 * N + r)
    H = syk_model.build_syk_hamiltonian(N, J=J)
    return r, np.linalg.eigvalsh(H)


def build_ensemble(N, R, max_seconds=None):
    """Resumable ensemble build: each realization's spectrum is saved to
    its own raw file the moment it finishes, so a killed run (background
    task lifetime caps are real) loses at most the in-flight
    realizations. The consolidated npz is written once all R exist.
    `max_seconds` makes the call exit cleanly after the deadline with
    whatever finished (for chunked background building)."""
    cache = DATA_DIR / f"spectra_N{N}_R{R}.npz"
    if cache.exists():
        spectra = np.load(cache)["spectra"]
        progress(f"N={N}: loaded {R} cached spectra from {cache.name}")
        return spectra
    DATA_DIR.mkdir(exist_ok=True)
    raw = {r: DATA_DIR / f"raw_N{N}_r{r}.npy" for r in range(R)}
    todo = [r for r in range(R) if not raw[r].exists()]
    t0 = time.time()
    if todo:
        progress(f"N={N}: {R - len(todo)}/{R} already cached raw; building "
                 f"{len(todo)} more")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_one_realization, (N, r)): r for r in todo}
            from concurrent.futures import as_completed
            for fut in as_completed(futures):
                r, ev = fut.result()
                np.save(raw[r], ev)
                progress(f"N={N}: realization {r} done "
                         f"[{time.time()-t0:.0f}s]")
                if max_seconds is not None and time.time() - t0 > max_seconds:
                    progress(f"N={N}: chunk deadline reached, exiting cleanly "
                             f"(resumable)")
                    for f in futures:
                        f.cancel()
                    break
    have = [r for r in range(R) if raw[r].exists()]
    if len(have) < R:
        progress(f"N={N}: {len(have)}/{R} raw spectra so far — not yet "
                 f"consolidated")
        return None
    spectra = np.array([np.load(raw[r]) for r in range(R)])
    np.savez_compressed(cache, spectra=spectra,
                        seeds=np.array([1000 * N + r for r in range(R)]))
    for r in range(R):
        raw[r].unlink()
    progress(f"N={N}: consolidated {R} realizations to {cache.name}")
    return spectra


def build_goe_control(N, R, spectra_syk):
    """GOE ensemble matched in dimension, R, and mean bandwidth. The
    bandwidth rescaling puts both ensembles in the same energy units so
    the band-resolved fits are directly comparable."""
    dim = 2 ** (N // 2)
    rng = np.random.default_rng([9, N])
    spectra = np.zeros((R, dim))
    for r in range(R):
        A = rng.normal(0.0, 1.0, size=(dim, dim))
        spectra[r] = np.linalg.eigvalsh((A + A.T) / np.sqrt(2.0 * dim))
    bw_syk = float((spectra_syk[:, -1] - spectra_syk[:, 0]).mean())
    bw_goe = float((spectra[:, -1] - spectra[:, 0]).mean())
    return spectra * (bw_syk / bw_goe)


def degeneracy_diagnostic(spectra):
    """Fraction of adjacent level pairs that are exactly (numerically)
    degenerate — the N mod 8 Kramers structure. Affects only the DOS
    prefactor, never C; reported for transparency."""
    gaps = np.diff(spectra, axis=1)
    return float(np.mean(gaps < 1e-10))


# ======================================================================
# SECTION 3 helpers — Route 1: linearized DOS edge fit
# ======================================================================
def fit_dos_band(spectra, band):
    """ln Ncum vs sqrt(eps) linear fit over a quantile band of the
    pooled edge-shifted levels: slope = 2 pi sqrt(2C) in the
    sinh-dominated regime, so C = slope^2/(8 pi^2)."""
    R = spectra.shape[0]
    eps = np.sort((spectra - spectra[:, :1]).ravel())
    counts = np.arange(1, eps.size + 1) / R
    lo, hi = int(band[0] * eps.size), int(band[1] * eps.size)
    x = np.sqrt(eps[lo:hi])
    y = np.log(counts[lo:hi])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[0])
    return slope ** 2 / (8.0 * np.pi ** 2), slope


def route1(spectra):
    """Band-resolved linearized DOS fit + plateau assessment + bootstrap
    on the headline band. Returns (C_head, C_boot_std, {band: C},
    plateau_ratio, measured_flag)."""
    bands = {b: fit_dos_band(spectra, b)[0] for b in DOS_BANDS}
    C_head = bands[DOS_HEADLINE_BAND]
    vals = np.array(list(bands.values()))
    plateau_ratio = float(np.max(vals) / max(np.min(vals), 1e-12))
    measured = plateau_ratio < DOS_PLATEAU_TOL
    rng = np.random.default_rng(7)
    R = spectra.shape[0]
    boots = [fit_dos_band(spectra[rng.integers(0, R, size=R)],
                          DOS_HEADLINE_BAND)[0] for _ in range(N_BOOT)]
    return C_head, float(np.std(boots)), bands, plateau_ratio, measured


# ======================================================================
# SECTION 4 helpers — Route 2: entropy linear-in-T coefficient
# ======================================================================
def entropy_curves(spectra, betas):
    """Per-realization thermal entropy S(beta), shape (R, nbeta)."""
    S = np.zeros((spectra.shape[0], betas.size))
    for r, ev in enumerate(spectra):
        x = ev - ev[0]
        w = np.exp(-np.outer(betas, x))
        p = w / w.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
        S[r] = -(p * lp).sum(axis=1)
    return S


def fit_entropy_window(S_mean, betas, window):
    """y = S + (3/2) ln beta regressed on 1/beta: slope = 4 pi^2 C.
    Plain linear least squares; sign of C is an output, not a constraint."""
    m = (betas >= window[0]) & (betas <= window[1])
    x = 1.0 / betas[m]
    y = S_mean[m] + 1.5 * np.log(betas[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]) / (4.0 * np.pi ** 2)


def fit_entropy_free_log(S_mean, betas, window):
    """Transparency fit: S = S0 + a ln T + gamma T with the log
    coefficient free. NOTE: ln T and T are strongly collinear over a
    finite window, so (a, gamma) trade off — this is reported for
    inspection only, never used as a C measurement."""
    m = (betas >= window[0]) & (betas <= window[1])
    T = 1.0 / betas[m]
    A = np.vstack([np.ones_like(T), np.log(T), T]).T
    coef, *_ = np.linalg.lstsq(A, S_mean[m], rcond=None)
    return float(coef[2]) / (4.0 * np.pi ** 2), float(coef[1])


def route2(spectra):
    """Window-resolved entropy fit + bootstrap on the headline window.
    Returns (C_head, C_boot_std, {window: C}, free_log_a)."""
    betas = np.geomspace(1.0, 40.0, 48)
    S = entropy_curves(spectra, betas)
    S_mean = S.mean(axis=0)
    win = {w: fit_entropy_window(S_mean, betas, w) for w in BETA_WINDOWS}
    _, a_free = fit_entropy_free_log(S_mean, betas, (3.0, 15.0))
    rng = np.random.default_rng(11)
    R = spectra.shape[0]
    boots = [fit_entropy_window(
        S[rng.integers(0, R, size=R)].mean(axis=0), betas, BETA_HEADLINE)
        for _ in range(N_BOOT)]
    return win[BETA_HEADLINE], float(np.std(boots)), win, a_free


# ======================================================================
# EXTRACT STAGE
# ======================================================================
def report_routes(label, spectra_by_N, n_values):
    """Run both routes on an ensemble family; print band/window tables;
    return {N: (C1, s1, measured, C2, s2)}."""
    out = {}
    print(f"\n--- {label} ---")
    for N in n_values:
        C1, s1, bands, ratio, measured = route1(spectra_by_N[N])
        C2, s2, win, a_free = route2(spectra_by_N[N])
        out[N] = (C1, s1, measured, C2, s2)
        band_str = "  ".join(f"{b[0]:.2f}-{b[1]:.2f}:{bands[b]:.3f}"
                             for b in DOS_BANDS)
        win_str = "  ".join(f"beta {w[0]:.0f}-{w[1]:.0f}:{win[w]:+.4f}"
                            for w in BETA_WINDOWS)
        print(f"  N={N:>2}  Route1 C={C1:.4f}+-{s1:.4f}  "
              f"plateau max/min={ratio:.2f} "
              f"({'MEASURED' if measured else 'NO PLATEAU — not a measurement'})")
        print(f"        bands: {band_str}")
        print(f"  N={N:>2}  Route2 C={C2:+.4f}+-{s2:.4f}  "
              f"(free-log a={a_free:+.2f}; Schwarzian & RMT edge both "
              f"predict 1.5)")
        print(f"        windows: {win_str}")
    return out


def stage_extract(smoke=False):
    global N_VALUES, N_REALIZATIONS
    if smoke:
        N_VALUES = [12]
        N_REALIZATIONS = {12: 6}

    section("1. MACHINERY VERIFICATION — imported, not rebuilt")
    err, dim = verify_syk_model()
    print(f"syk_model.generate_majoranas(12): max |{{chi_i,chi_j}} - "
          f"delta_ij I| = {err:.3e} (dim {dim})")
    assert err < 1e-12, "imported Majorana construction failed algebra check"
    print("syk_model algebra verified to machine precision.")
    print("Coupling convention: variance(J_ijkl) = 6 J^2/N^3, J =", J)
    res = compute_G(0.1, 1.0, 80.0)
    print(f"schwarzian.compute_G standing gate: G(0.1,1,80) = "
          f"{res['G']:.12e} (expected 3.980425717581e+02), "
          f"conv_err = {res['conv_err']:.1e}")
    assert abs(res["G"] / 3.980425717581e+02 - 1.0) < 1e-9

    section("2. SYK DISORDER ENSEMBLES + GOE CONTROLS")
    print(f"N values: {N_VALUES}; realizations: "
          f"{[N_REALIZATIONS[n] for n in N_VALUES]}; seeds 1000*N + r")
    spectra_by_N, goe_by_N = {}, {}
    for N in list(N_VALUES):
        sp = build_ensemble(N, N_REALIZATIONS[N])
        if sp is None:
            print(f"  N={N:>2}: ensemble incomplete (background build in "
                  f"progress) — EXCLUDED from this pass")
            N_VALUES = [n for n in N_VALUES if n != N]
            continue
        spectra_by_N[N] = sp
        goe_by_N[N] = build_goe_control(N, N_REALIZATIONS[N], spectra_by_N[N])
        print(f"  N={N:>2} (dim {sp.shape[1]:>4}): "
              f"E_0 = {sp[:, 0].mean():.4f} +- {sp[:, 0].std():.4f}, "
              f"bandwidth = {(sp[:, -1]-sp[:, 0]).mean():.4f}, "
              f"exact-degenerate adjacent pairs: "
              f"{100*degeneracy_diagnostic(sp):.1f}% (N mod 8 = {N % 8})")

    section("3. ROUTES 1 AND 2 — SYK, then the GOE control")
    syk = report_routes("SYK", spectra_by_N, N_VALUES)
    goe = report_routes("GOE control (same dim/R, bandwidth-matched)",
                        goe_by_N, N_VALUES)

    section("4. THE C(N) GATE")
    print(f"{'N':>3} | {'SYK C_DOS':>16} | {'SYK C_thermo':>16} | "
          f"{'disagree':>8} | {'agree':>5} | {'GOE C_DOS':>9} | {'GOE C_th':>8}")
    print("-" * 84)
    agrees = {}
    for N in N_VALUES:
        C1, s1, meas, C2, s2 = syk[N]
        g1, _, gmeas, g2, _ = goe[N]
        dis = abs(C1 - C2) / max(abs(C1 + C2) / 2.0, 1e-12)
        agrees[N] = (dis < ROUTE_AGREE_TOL) and meas and (C2 > 0)
        flag = "" if meas else "*"
        print(f"{N:>3} | {C1:7.4f}{flag}+-{s1:6.4f} | {C2:+7.4f} +-{s2:6.4f} | "
              f"{100*dis:7.1f}% | {str(agrees[N]):>5} | {g1:9.4f} | {g2:+8.4f}")
    print("(* = Route 1 slope has no plateau across bands: not a measurement.")
    print(" 'agree' requires: plateau exists, Route 2 C positive, and the")
    print(" two routes within 15% — the pre-registered gate conditions.)")

    n_agree = sum(agrees.values())
    if n_agree >= 3:
        Ns = np.array([N for N in N_VALUES if agrees[N]], dtype=float)
        Cs = np.array([syk[N][0] for N in N_VALUES if agrees[N]])
        ws = 1.0 / np.array([max(syk[N][1], 1e-6)
                             for N in N_VALUES if agrees[N]]) ** 2
        xm = np.sum(ws * Ns) / np.sum(ws)
        ym = np.sum(ws * Cs) / np.sum(ws)
        slope = np.sum(ws * (Ns - xm) * (Cs - ym)) / np.sum(ws * (Ns - xm) ** 2)
        se = np.sqrt(1.0 / np.sum(ws * (Ns - xm) ** 2))
        print(f"\nLinearity of C(N) over agreeing N: slope = "
              f"{slope:.5f} +- {se:.5f} per unit N "
              f"({'positive, Schwarzian-consistent' if slope > 0 else 'NOT positive'})")

    print(f"\nRoutes agree (pre-registered criteria) at {n_agree}/"
          f"{len(N_VALUES)} N values.")
    if n_agree == 0:
        print("""
VERDICT (reported straight, per the gate conditions):
Matched parameters could NOT be established at N <= 20. No averaged or
fudged C will be produced. See the GOE control rows: if SYK's fit
pattern (drifting DOS slope, non-positive entropy coefficient) tracks
the GOE control, the accessible spectral edges are generic
random-matrix behavior and the Schwarzian coupling has not emerged at
these sizes. The barcode comparison is not justified with this
dictionary.""")
    else:
        print("\nGATE: stopping here for user review of (a) route agreement, "
              "(b) linearity in N, (c) magnitudes, (d) SYK-vs-GOE "
              "separation — before any barcode comparison runs.")
    return syk, goe, agrees


# ======================================================================
if __name__ == "__main__":
    args = list(sys.argv[1:])
    smoke = "--smoke" in args
    for a in args:
        if a.startswith("--N="):  # e.g. --N=12,14,16,18
            N_VALUES = [int(x) for x in a.split("=", 1)[1].split(",")]
    stage = next((a for a in args if not a.startswith("--")), "extract")
    if stage != "extract":
        sys.exit("only the 'extract' stage exists; 'compare' is gated on "
                 "user review of the C(N) results")

    tag = "_smoke" if smoke else ""
    results_path = HERE / f"rung4_results{tag}.txt"
    log = open(results_path, "a")
    _stdout = sys.stdout
    sys.stdout = Tee(_stdout, log)
    print(f"\n############ rung4 {stage}{tag} "
          f"({time.strftime('%Y-%m-%d %H:%M:%S')}) ############")
    t0 = time.time()
    stage_extract(smoke=smoke)
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Results appended to: {results_path}")
    sys.stdout = _stdout
    log.close()
