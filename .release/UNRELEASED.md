# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **timeseries: damped-trend ETS fits no longer stall (damped fits improve
  or stay identical; every other fit changes only if it previously ran out
  of optimiser budget).** Two compounding defects in
  `timeseries/_ets_fit.py`: (1) R's `initparam` starts `phi` at 0.9782 —
  99% of the way to the 0.98 usual-region bound — which is fine for R's
  derivative-free Nelder-Mead but saturates our logit transform, so the
  numerical gradient in the phi direction was ~4% of mid-range (sigmoid
  derivative 0.0099 vs 0.25) and L-BFGS-B crawled; (2) scipy's default
  `maxfun=15000` allows only ~830 iterations for a 17-parameter seasonal
  model (numerical gradient = n+1 evals/iteration), so the requested
  `maxiter=1000` was unreachable and co2 `MAdM` died with "EVALUATIONS
  EXCEEDS LIMIT" at `converged=False`, ~1.7 AICc worse than its true
  optimum. Fix: damped models now optimise from BOTH a mid-range phi start
  (0.9) and R's initparam start (0.9782), keeping the better optimum, and
  `maxfun = max(15000, max_iter*(n_free+1)*2)` — floored at scipy's old
  default so no fit ever gets a smaller budget than 4.6.2 gave it. That
  floor makes the guarantee provable: the R-start leg reproduces the 4.6.2
  trajectory (deterministic, with at least the old budget), so damped
  results are never worse, and any fit that terminated within the old
  budget (every reference fit does) is bit-identical unless damped. The
  phi bounds stay R's (0.8, 0.98). Verified at full precision on 15 damped
  reference fits: 7 improved (co2 MAdM aicc 172.592→170.898 and now
  converged, beating R's own damped optimum; usaccdeaths MAdA −0.25), 8
  bit-identical, zero worse, all `converged=True`; all non-damped
  reference fits bit-identical; all 10 reference ZZZ selections unchanged
  (on non-reference series a damped candidate whose fit improved can now
  legitimately win a selection it previously lost — that is the point of
  the fix).
- **timeseries: `converged=False` false-negative on damped ETS fits fixed**
  — it was the evaluation-budget exhaustion above, not a flag-logic bug; a
  damped fit that reaches its optimum now reports `converged=True` (new
  regression tests pin this for co2/airpassengers/usaccdeaths/lynx/nile/
  wwwusage damped fits).
- **timeseries/tests: honest cross-engine ETS verification encoded as a
  regression gate.** A prior review claimed our ZZZ selections were worse
  than R's because refitting our picked model string in R gave a higher
  AICc — but that only re-runs R's own Nelder-Mead (the optimiser being
  compared) and reproduces its stall. The fair test — evaluating OUR fitted
  parameters with R's own likelihood code `forecast:::pegelsresid.C`
  (seasonal states reversed into R's ordering) — shows our pick has strictly
  lower AICc under R's own criterion on all 6 divergent reference datasets
  (e.g. co2: our M,A,M scores 1697.2 in R vs 1722.6 for R's chosen M,Ad,M)
  and equal-or-better on the 4 agreeing ones, with every pick admissible
  under R's `admissible()`. The fixture regeneration is now two-stage
  (`tests/fixtures/generate_ets_py_params.py` dumps our fitted parameters,
  `generate_ets_r_reference.R` cross-scores them; the transplant harness
  self-validates by reproducing R's own fits' log-likelihoods before
  scoring foreign parameters, and stores `py_pick`/`py_pick_aicc_in_r`/
  `py_pick_admissible` per dataset). New tests
  (`test_selection_dominates_r_under_r_own_likelihood`, drift-guarded on
  the stored pick) fail if any future change degrades a selection to
  something R's own numbers call worse. `_ets_select.py`'s parity
  documentation now states this verification method explicitly instead of
  a vague "better than R". The selection/parity/damped-convergence tests
  moved to a new `tests/timeseries/test_ets_selection.py` (`test_ets.py`
  had exceeded the 500-LoC limit; now 382 + 262). The R generator now
  refuses to run without the stage-1 params dump instead of silently
  writing a fixture missing the fields ten tests require.
