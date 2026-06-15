# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- GPU PMM donor search rewritten to scale. `pystatistics/mice/backends/
  _gpu_methods.py:_match_donors_windowed` replaces the dense
  `(m, n_mis, n_obs)` distance tensor with a batched sorted-window k-NN (per-
  chain `torch.sort` + `torch.searchsorted` + a width-`2k` window), the GPU port
  of the 3.4.1 CPU matcher. Memory drops from `O(m·n_mis·n_obs)` to
  `O(m·n_mis·k)`; the donor set is unchanged (the window provably contains the
  global k nearest). Fixes a CUDA OOM at large n (e.g. n=20000, m=50 tried to
  allocate ~12 GB; now ~0.24 GB peak) and the dense tensor was also bandwidth-
  bound, so removing it made the GPU both scalable and much faster.
- Re-measured GPU vs CPU PMM on an RTX 5070 Ti (p=10, maxit=10), against the
  3.4.1 CPU: ~8.5x at n=2000/m=20, ~30x at n=20000/m=20, ~50x at n=20000/m=50,
  ~38x at n=100000/m=20 (peak GPU memory < 0.5 GB throughout). n=100000 did not
  run at all before. NOTE: the speedup figures quoted for 3.4.0 (~39x at n=1000,
  ~135x at n=3000) were measured against the *pre-3.4.1* quadratic CPU and
  overstate the advantage over the current CPU; these windowed numbers are the
  current, honest comparison.
- Added GPU scaling regression tests (`tests/mice/test_gpu.py::TestGpuScales`):
  large-n PMM must stay memory-frugal (guards against reintroducing the dense
  matrix) and still match the CPU distribution.
