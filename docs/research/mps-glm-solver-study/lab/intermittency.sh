#!/bin/bash
# Cross-process intermittency probe: launch single_run.py in a FRESH process per
# trial and tally verdicts. The only honest way to measure the PD-edge flip
# (within one process the outcome is fixed). High-N on the borderline cases,
# low-N controls to confirm the others are deterministic.
# Usage: intermittency.sh <PY> <out.jsonl> <device> <NHIGH> <NLOW>
set -u
PY="$1"; OUT="$2"; DEV="${3:-mps}"; NHIGH="${4:-24}"; NLOW="${5:-6}"
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=0
export PYSTATS_LAB_DEVICE="$DEV"
: > "$OUT"
run_combo () { # solver regime N
  local solver="$1" regime="$2" N="$3" i
  for i in $(seq 1 "$N"); do
    "$PY" -W ignore -m lab.single_run "$solver" "$regime" 2>/dev/null >> "$OUT"
  done
  python3 - "$OUT" "$solver" "$regime" <<'PY'
import sys, json
from collections import Counter
out, solver, regime = sys.argv[1:4]
rows=[json.loads(l) for l in open(out) if l.strip()]
rows=[r for r in rows if r["solver"]==solver and r["regime"]==regime]
c=Counter(r["verdict"] for r in rows)
print(f"  {solver:9s} {regime:9s} n={len(rows):2d}  conv={c.get('converged',0)} "
      f"fail_loud={c.get('fail_loud',0)} silent_wrong={c.get('silent_wrong',0)}", flush=True)
PY
}
# Borderline (PD-edge) -- high N
run_combo cholesky quarterly "$NHIGH"
run_combo hybrid   quarterly "$NHIGH"
# Controls -- confirm determinism
run_combo cg       quarterly "$NLOW"
run_combo lm       quarterly "$NLOW"
run_combo cholesky yearly    "$NLOW"
run_combo cholesky monthly   "$NLOW"
run_combo cholesky biweekly  "$NLOW"
echo "wrote $OUT"
