# Rung 4 Status Report — for Arjun

*Handoff document, 2026-08-30. Collaborator report, not a paper draft. All
numbers below are copied from actual run output (committed in
`rung4/rung4_results.txt` through commit `3ee3035`, or from the post-closure
diagnostic runs of 2026-08-29/30 reproduced in full in this report — see §6
for exactly what is committed where).*

---

## 1. TL;DR

Rung 4 is **closed as a decisive negative**: the declared Maldacena–Stanford
dictionary C(N) = √2·α_S·N is falsified at accessible N (≤ 18), two
independent ways — the pre-registered extraction gate failed 0/4 (spectra are
RMT-like, no Schwarzian coupling measurable), and the correlator-level
comparison shows the SYK decay shape is unreachable by *any* Schwarzian C for
β ≤ 20 and fits inconsistently (temperature-dependent, separation-dependent,
wrong N-scaling) at β = 40. Everything is committed locally through
**`3ee3035`** ("Rung 4 CLOSED as an honest negative"); the verbatim teed runs
live in `rung4/rung4_results.txt`, and the post-closure diagnostics are
tabulated in §3 of this report. **Note: as of this writing the local branch
is 8 commits ahead of `origin/main` — the entire rung-4 body of work exists
only on this machine until pushed.**

## 2. What we tested and why

Rung 4 replaced rung 3's frozen 1-D JT slice (provably H1-free) with a
structurally symmetric comparison: both sides reindexed by Euclidean time on
the same 24-point thermal circle.

- **EPR side**: SYK₄ autocorrelator G(τ) = Tr[e^{−(β−τ)H} χ e^{−τH} χ]/Z,
  per-fermion average, 10 seeded realizations per N ∈ {12,14,16,18},
  β ∈ {5,10,20,40} (your `syk_model.py` builder, seeds `1000·N + r`).
- **ER side**: the exact Schwarzian boundary two-point function
  (Mertens–Turiaci–Verlinde 1705.08408), implemented in
  `rung4/schwarzian.py`, validated by Checks A–E at machine precision within
  scope C ≤ 200.
- Both profiles → distance matrix d = 1/Ghat (Ghat = G/G(β/2)) on the τ
  circle → persistent homology (ripser) → Wasserstein (H0+H1) against the
  ER barcode at declared C plus ×5 / ÷5 mismatched-C controls and the
  conformal C→∞ shape.

**Why declared C**: the load-bearing design rule was that the two sides'
parameters must correspond physically, so we first tried to *measure* C from
SYK's own spectra (pre-registered gate, two routes: DOS sinh-edge slope and
entropy linear-in-T). The gate **failed 0/4** — no Route-1 plateau at any N,
*negative* Route-2 C at every N, routes disagreeing 230–433%, and the fit
pattern tracking a bandwidth-matched GOE control. Per explicit decision
(2026-08-24), C was then **declared from the literature instead of
measured**: C(N) = α_S·N/𝒥 with 𝒥 = J/√2 (MS eq 2.16) and α_S = 0.00709.
Provenance (source-verified 2026-08-29, at the PDF-font level for the
J-vs-𝒥 distinction): MS never print α_S numerically; it is derived from
their stated q=4 specific heat c ≈ 0.396 N/J (plain J, after eq 5.181) via
c = 4π²α_S N/𝒥 (calligraphic 𝒥, eq 5.181). **α_S is single-source** — every
published q=4 value traces back to MS's fitted numerics — a declared input
may be single-source, but its independence is limited and we say so.

The committed verdict (2/16 cells favored vs a ~1/3 chance rate, misses
systematic in the conformal direction, no N-trend at β=40, convention-robust
under both d = 1/Ghat and d = log(Gmax/G): identical favored set, zero
flips) is in `rung4_results.txt` §6 and the closure section. What follows
sharpened it.

## 3. The three findings

### 3.1 Two-scalar reduction: the barcodes carry exactly two numbers per side

On this construction the distance matrix is circulant (depends only on
separation m) and every G profile is strictly monotone in m — so at any
threshold the Vietoris–Rips complex is the clique complex of a circulant
graph C₂₄(1..k), and Adamaszek–Adams (arXiv:1503.03669, Thm 4.3) forces the
homotopy type: S¹ for k/n < 1/3, H1 dead at k/n = 1/3, higher spheres
invisible at maxdim=1. Consequences, verified grid-wide (all 224 diagrams:
160 SYK + 64 ER):

- monotonicity violations: **0** (min forward step +4.5e-4 SYK, +7.1e-4 ER);
- every H1 diagram is **exactly one bar** = [1/Ghat(1), 1/Ghat(8)] to
  2.97e-8 (0 empty, 0 extra);
- every H0 diagram is 23 copies of [0, 1/Ghat(1)] (deaths identical to
  machine precision);
- persim's Wasserstein is p=1 with L2 ground metric (verified from source);
  direct matching always beats diagonal deletion here, so
  W_H0 = 23·|Δ(1/Ghat(1))| (verified to 7.7e-7) and W_H1 =
  √(Δ(1/Ghat(1))² + Δ(1/Ghat(8))²).

**So every Wasserstein number in the committed table is an exact function of
(Ghat(1), Ghat(8)) on the two sides, weighted ~23:1 toward Ghat(1)** — the
nearest-neighbor correlation ratio. The persistent homology carries no
topological information beyond those two correlator values. (The "H1 > 0 on
both sides" observation is architecturally guaranteed and proves nothing —
the same lesson as rungs 1–2's Wasserstein-0.) An H0/H1 split rescore also
confirmed the committed favored set {N=14 β=40, N=16 β=40} is identical on
W_H0 alone, W_H1 alone, and the total — no cell's status is an artifact of
the summed metric.

### 3.2 Per-cell reachability: SYK is outside the Schwarzian family for all β ≤ 20

Ghat_Schwarzian(1) is a function of β/C alone (checked: exact collapse
across β at equal β/C, spread 0.00e+00), monotone increasing in β/C, with
**infimum = the conformal value 2.767905** (approached from above as
β/C → 0). Measured SYK values (mean ± std over 10 realizations):

| Ghat_SYK(1) | β=5 | β=10 | β=20 | β=40 |
|---|---|---|---|---|
| N=12 | 1.291 ± 0.016 | 1.760 ± 0.046 | 2.473 ± 0.141 | 3.191 ± 0.410 |
| N=14 | 1.305 ± 0.010 | 1.765 ± 0.030 | 2.631 ± 0.141 | 4.550 ± 0.744 |
| N=16 | 1.302 ± 0.008 | 1.749 ± 0.031 | 2.626 ± 0.154 | 4.274 ± 0.573 |
| N=18 | 1.303 ± 0.005 | 1.729 ± 0.012 | 2.474 ± 0.037 | 3.554 ± 0.149 |

Every β ≤ 20 cell sits **below the conformal infimum** — flatter at short
separation than the conformal limit itself — so *no value of C* (the
declared one or any other) can match it: 12 of 16 cells are unreachable,
structurally. SYK crosses the conformal value between β=20 and β=40 at every
N (interpolated crossing β ≈ 26.6, 21.0, 21.2, 24.1 for N = 12, 14, 16, 18),
entering the Schwarzian family's range only at β=40. For context, the
declared dictionary puts the whole grid at β/C between 27.7 and 332 — every
cell deep in the strongly-quantum Schwarzian regime; nothing here is
semiclassical (a consequence of C(N) ≈ 0.01·N itself, not of our β choices).

### 3.3 Where a fit exists (β=40), the fitted C falsifies the dictionary

C_fit = argmin_C W_H0 over C ∈ (0, 200], reported only where the curves
actually cross (residuals ≤ 2e-7, i.e. exact crossings; monotone-to-boundary
cells reported as NO FIT, never as a boundary value):

| N | β | C_fit [Ghat(1)] | C_fit/C_declared | C_fit [Ghat(8)] |
|---|---|---|---|---|
| 12 | 20 | NO FIT | — | 0.6607 |
| 12 | 40 | 1.2951 ± 0.4923 | 10.76 | 0.8828 |
| 14 | 20 | NO FIT | — | 0.2297 |
| 14 | 40 | 0.2124 ± 0.0338 | 1.51 | 0.0272 |
| 16 | 20 | NO FIT | — | 0.2287 |
| 16 | 40 | 0.2618 ± 0.0423 | 1.63 | 0.0834 |
| 18 | 20 | NO FIT | — | 0.4233 |
| 18 | 40 | 0.5552 ± 0.0416 | 3.08 | 0.2987 |

(All β ≤ 10 cells: NO FIT on both targets. β=20: NO FIT on Ghat(1),
crossings on Ghat(8) only.)

Three independent failures of dictionary behavior:

1. **Temperature dependence** (C must be β-independent): on the Ghat(8)
   target, where two betas fit, the spread across β at fixed N is
   **28.8% / 157.7% / 93.1% / 34.5%** of the mean for N = 12/14/16/18.
2. **Separation inconsistency** (one C must fit the whole profile):
   C_fit(Ghat1)/C_fit(Ghat8) = **1.47 / 7.81 / 3.14 / 1.86** — fitting the
   nearest-neighbor scale and the β/3 scale demands different couplings.
3. **No MS power law**: fitting C_fit = a·N^b over the four β=40 cells gives
   **b = 2.16 ± 2.40, a = 0.00098 (×/÷ 873), χ²/dof = 34.8/2** — the points
   are not even monotone in N (1.295 → 0.212 → 0.262 → 0.555). MS predicts
   b = 1, a = 0.01003. Nothing resembling C ∝ N is present.

## 4. Mechanism

The disagreement is UV-localized, and three independent probes say the same
thing. At the shortest circle separation (τ = β/24) SYK at accessible N is
still nearly free — its correlator barely decays (Ghat(1) ≈ 1.3 at β=5)
while both the conformal limit and every Schwarzian C predict steep decay
(Ghat(1) ≥ 2.77) — and since the barcode metric is ~23:1 weighted toward
exactly that scale, the comparison is dominated by the one regime where the
low-energy dictionary should not apply. Consistently: refining the τ grid
(n_τ = 12 → 24 → 48) makes the per-pair gap *worse* (0.340 → 0.445 → 0.523)
because finer grids probe deeper UV, while excluding the shortest
separations (m ≥ 2, m ≥ 3) monotonically *improves* agreement
(W_H0 10.24 → 7.82 → 6.05 at the N=18, β=5 cell). This is the same physics
the extraction gate saw spectrally (RMT-like edges, no Schwarzian scale) and
the β/C table shows parametrically (entire grid strongly-quantum), now seen
a third way at the correlator level.

## 5. What this means for the project

Rung 4 is a clean result, not a failure: a decisive, convention-robust,
mechanistically understood negative, plus a methodological theorem about
what this class of PH pipeline can measure.

**Proven** (machine-checked, teed in `rung4_results.txt`):
- `schwarzian.py` computes the exact MTV two-point τ-dependence to
  quadrature precision (Checks A–E; Jacobian A/B-verified; absolute
  normalization implementation-anchored, shape-gauged everywhere).
- Extraction gate failed 0/4; declared dictionary favored at 2/16 under
  both distance conventions (identical cells, zero flips); the two hits
  clear the ×5 control by only ~0.4–0.5σ in the validated log convention.
- The two-scalar reduction (§3.1), the reachability bound and per-cell
  status (§3.2), and the C_fit inconsistencies (§3.3), with the numbers
  above.

**Conjectured / not established here**:
- That the Schwarzian regime emerges at larger N (standard expectation;
  untested here).
- That the missing soft mode is the *whole* story of the conformal-ward
  misses (small-N artifacts — spectral discreteness at β=40, finite-size
  edges — were not disentangled).
- Anything about ER=EPR itself: rung 4 tested one specific SYK↔Schwarzian
  dictionary at small N and found it not applicable there; it neither
  supports nor undermines the conjecture.

**Future work only** (explicitly not pursued): N = 20–24 via parity-block
diagonalization (pipeline already supports N=20 in `N_VALUES`); a
best-fit-C-by-barcode variant would need pre-registration — and note §3.3
already shows what an unregistered version would find: β- and
separation-dependent C with no N-law.

## 6. Division of labor and where everything lives

**Yours (SYK numerics + pipeline):** `syk_model.py` (Majorana construction,
variance 6J²/N³ — source-verified against MS eq 2.3), the ensemble/caching
machinery, and the spectra caches `rung4/data/spectra_N{12,14,16,18}_R*.npz`
(seeded rebuilds reproduce them to <1e-8; seeds `1000·N + r`).

**This side (ER math + validation + closure):** `rung4/schwarzian.py` (exact
MTV implementation + Checks A–E in `rung4_checks.py`), `rung4/rung4.py`
(both stages), `rung4/rung4_rescore.py` (convention-robustness rescore), the
physics-provenance audit, and the closure analysis.

**Committed** (local `main`, tip `3ee3035`): all of the above plus
`rung4_results.txt` — the verbatim teed record of every extract run, the
compare run, the two-convention rescore, and the authored closure section
(read that file end-to-end and you have the full committed story).
**Not yet committed:** the 2026-08-29/30 post-closure diagnostics (H0/H1
split rescore, UV diagnostics, reachability sweeps, C_fit battery) — their
complete numbers are in §3–§4 of this report, and the scripts can be added
to the repo on request. **And again: local `main` is 8 commits ahead of
`origin/main` (github.com/ArjunParadkar/ER-EPR) — nothing from rung 4 is on
the remote yet.**
