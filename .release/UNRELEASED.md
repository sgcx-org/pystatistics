# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- CPU PMM (`mice` predictive mean matching) now scales to large datasets.
  Replaced the dense `(n_mis, n_obs)` distance matrix in
  `pystatistics/mice/methods/pmm.py:_match_donors` with a sorted-array windowed
  k-NN (sort the observed predictions once, search a width-`2k` window around
  each missing value's insertion point), matching R `mice`'s `matchindex`
  approach. Cost drops from `O(n_mis·n_obs)` to `O(n_obs log n_obs + n_mis·k)`
  in both time and memory. Measured CPU PMM (p=10-15, m=20, maxit=8): n=3000
  25.4s -> 0.65s (~39x); n=20000 went from minutes/hours to ~6.5s. The window is
  provably exact (the global k nearest neighbours lie in `[pos-k, pos+k-1]` of
  the sorted array), so the donor pool is unchanged; imputations remain
  distributionally identical and still match R. Exact per-seed outputs differ
  from 3.4.0 because the donor ordering within the k-NN set changed.
