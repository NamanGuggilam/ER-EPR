# Post-closure diagnostics (2026-08-29/30)

Scripts and verbatim outputs for the diagnostics run after the rung-4
closure commit, summarized with full numbers in
`../RUNG4_REPORT_FOR_ARJUN.md` §3–§4:

- `jacobian_ab_test.py` — A/B test of the (s,d) LOG_JACOBIAN: log G shifts
  by exactly log 2, Ghat and barcodes unchanged.
- `ghat_edge_check.py` — positivity/monotonicity of d = 1/Ghat on real
  compare inputs (N=12 spot check; full grid in split_wasserstein).
- `conformal_corner_h1.py` / `conformal_corner_out.txt` — H1 bar
  coordinates at the grid's minimum-β/C cell (N=18, β=5). NOTE: the
  script's docstring calls this the "conformal corner"; that label was
  later retracted — β/C = 27.7 is deep in the strongly-quantum regime.
- `split_wasserstein.py` / `split_out.txt` — retroactive H0/H1 split of
  the committed table, all 16 cells; grid-wide monotonicity and one-bar
  prediction checks (224 diagrams, 0 violations).
- `uv_diagnostics.py` / `uv_out.txt` — persim H0 prediction check,
  Ghat_Schwarzian(1) vs β/C sweep (β=5), n_τ = 12/24/48 sweep, m ≥ 2/3
  clamped filtrations.
- `cfit_battery.py` / `cfit_out.txt` (crashed at the C_fit bracket) and
  `cfit_out2.txt` (patched rerun, complete) — per-β reachability, the
  16-cell Ghat_SYK(1)/Ghat(8) table, conformal crossings, and the C_fit
  suite (a)–(e).

Caveats: these were run from a session scratchpad, so the scripts contain
absolute paths (`/Users/naman/projects/ER-EPR` in `sys.path` inserts and
cache paths) — adjust before rerunning elsewhere. They import the committed
`rung4.py`/`schwarzian.py` machinery and recompute from the same seeds;
`split_out.txt` includes the regression check reproducing the committed
table to 4.9e-5. The .txt files are the verbatim stdout of the runs.
