"""Freeze a benchmark run to the validation artifact schema.

One job: assemble ``{schema, env, config, records}`` (the ``validation-run/v1``
shape consumed by the pystatistics-validation renderer) and write it to disk as
JSON. The env block is mandatory for a newly-generated run — a run with no
reproducibility manifest is not a valid artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUN_SCHEMA = "validation-run/v1"


def build_run(*, env: dict[str, Any],
              config: dict[str, Any],
              records: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble a run object. Validates the output contract before returning.

    Raises
    ------
    ValueError
        If ``env`` lacks the required reproducibility fields, or ``records`` is
        empty (an artifact with no measurements is meaningless).
    """
    for key in ("pystatistics_version", "install_source", "device"):
        if key not in env:
            raise ValueError(f"env is missing required field {key!r}")
    if not records:
        raise ValueError("records is empty; refusing to write an evidence-free run")
    return {
        "schema": RUN_SCHEMA,
        "env": env,
        "config": config,
        "records": records,
    }


def write_run(path: str | Path, run: dict[str, Any]) -> Path:
    """Write a run dict to ``path`` as pretty JSON, creating parents. Returns path."""
    if run.get("schema") != RUN_SCHEMA:
        raise ValueError(f"not a {RUN_SCHEMA} object (schema={run.get('schema')!r})")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2))
    return path
