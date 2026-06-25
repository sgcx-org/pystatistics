"""Generic bridge for timing an R reference implementation in a subprocess.

One job: the reusable mechanics of handing a numeric matrix to an R script, running
it under a timeout, and parsing a JSON result — with the timing done *inside* R so
R's interpreter startup is excluded. Subsystem-specific pieces (which R script, how
to turn its JSON into a record) live with the subsystem driver, not here.

Requires ``Rscript`` on PATH; callers should check :func:`have_rscript` and record a
skip rather than treating its absence as fatal.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def have_rscript() -> bool:
    """True if ``Rscript`` is available on PATH."""
    return shutil.which("Rscript") is not None


def matrix_to_na_csv(X: NDArray[np.floating], path: str | Path) -> None:
    """Write ``X`` as headerless CSV with R-style ``NA`` for missing (NaN) cells."""
    arr = np.asarray(X, dtype=np.float64)
    np.savetxt(path, arr, delimiter=",", fmt="%.10g")  # NaN -> "nan"
    p = Path(path)
    p.write_text(p.read_text().replace("nan", "NA"))


def run_r_json(r_script: str | Path, args: list[str], *,
               timeout_s: float = 1800.0) -> tuple[bool, dict[str, Any] | str]:
    """Run ``Rscript r_script args...`` where the script writes JSON to ``args[1]``.

    Convention: ``args[0]`` is the input path, ``args[1]`` is the output JSON path
    the R script writes. Returns ``(ok, payload)`` — ``payload`` is the parsed dict
    on success, or an error string on failure (non-zero exit, timeout, bad JSON).
    Never raises for the expected non-fatal cases — the caller records the skip.
    """
    r_script = Path(r_script)
    if not r_script.is_file():
        raise FileNotFoundError(f"R script not found: {r_script}")
    out_path = Path(args[1])
    try:
        proc = subprocess.run(
            ["Rscript", str(r_script), *args],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s:.0f}s"
    if proc.returncode != 0:
        return False, f"R error (rc={proc.returncode}): {proc.stderr.strip()[:300]}"
    try:
        return True, json.loads(out_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"could not read R output {out_path}: {exc}"


def with_tempdir(prefix: str = "pystatsval_r_"):
    """Context-manager-free temp dir helper: returns (dir, cleanup-callable)."""
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    return tmp, (lambda: shutil.rmtree(tmp, ignore_errors=True))
