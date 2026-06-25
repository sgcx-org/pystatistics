"""Capture the environment manifest behind a benchmark run.

One job: record exactly what produced a set of measurements — the pystatistics
version, how it was installed (PyPI vs editable), the interpreter/library stack,
and the compute device — so a frozen artifact is reproducible and a canonical
report can refuse to be built from an unreleased local checkout.

Two hard rules, both learned from real bugs:

1. The pystatistics version is read from the **imported module**
   (``pystatistics.__version__``), never from ``importlib.metadata``. An editable
   install freezes the metadata version at install time, so metadata can report a
   stale version while the imported code is newer (the documented stale-label bug).

2. ``install_source`` distinguishes ``pypi`` from ``editable``. A *canonical*
   validation report must be built from a PyPI release; :func:`require_pypi`
   enforces that and fails loud otherwise.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from typing import Any


def _version(modname: str) -> str | None:
    try:
        mod = __import__(modname)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def detect_install_source(dist_name: str = "pystatistics") -> str:
    """Return ``"pypi"``, ``"editable"``, or ``"unknown"`` for an installed dist.

    Reads PEP 610 ``direct_url.json``: an editable install records
    ``dir_info.editable = true``; a plain ``pip install <name>`` from PyPI has no
    ``direct_url.json`` at all. A non-editable local/VCS install is reported as
    ``"editable"`` too (it is equally non-canonical for a published report).
    """
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return "unknown"
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        raw = None
    if not raw:
        return "pypi"  # installed from an index (no direct URL recorded)
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return "unknown"
    if info.get("dir_info", {}).get("editable"):
        return "editable"
    # direct_url present but not editable: a local dir / VCS install — not canonical.
    return "editable"


def _torch_device_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch
    except Exception:
        return info
    info["torch"] = getattr(torch, "__version__", None)
    if torch.cuda.is_available():
        info["available_devices"] = ["cuda", "cpu"]
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        except Exception:
            pass
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info["available_devices"] = ["mps", "cpu"]
    else:
        info["available_devices"] = ["cpu"]
    return info


def env_manifest(*, device: str, host: str | None = None,
                 dist_name: str = "pystatistics",
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the reproducibility manifest for a run on ``device``.

    Parameters
    ----------
    device
        The compute device this run used: ``"cpu"``, ``"mps"``, or ``"cuda"``.
    host
        Logical host id (matches the artifact manifest's ``hosts`` map), optional.

    Raises
    ------
    ValueError
        If ``device`` is not one of cpu/mps/cuda, or pystatistics is not importable
        (a run with no library under test is meaningless — fail loud).
    """
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError(f"device must be cpu/mps/cuda, got {device!r}")
    pyst_version = _version(dist_name)
    if pyst_version is None:
        raise ValueError(
            f"{dist_name} is not importable; cannot build an env manifest for a "
            "run that has no library under test.")

    env: dict[str, Any] = {
        "pystatistics_version": pyst_version,
        "install_source": detect_install_source(dist_name),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "device": device,
    }
    env.update(_torch_device_info())
    if host:
        env["host"] = host
    if extra:
        env.update(extra)
    return env


def require_pypi(env: dict[str, Any]) -> None:
    """Fail loud unless ``env`` came from a PyPI install (canonical-report guard).

    Raises
    ------
    RuntimeError
        If ``install_source`` is not ``"pypi"``. A canonical, version-pinned report
        must validate a *released* artifact; producing one from an editable/local
        checkout is a reproducibility violation.
    """
    src = env.get("install_source")
    if src != "pypi":
        raise RuntimeError(
            f"refusing to produce canonical evidence from install_source={src!r}: "
            f"pystatistics must be installed from PyPI (pip install "
            f"pystatistics=={env.get('pystatistics_version')}). If the code does "
            "not install from PyPI, the code is wrong.")
