"""Hatchling build hook: compile the Cython extension modules.

Pilot scope: builds ``pystatistics/timeseries/_arima_kalman_kernel`` only. As
the Numba->Cython migration proceeds, add each new ``.pyx`` to ``PYX_MODULES``.

FP determinism is a first-class build concern (see docs/CYTHON_MIGRATION_PROPOSAL.md
§6.1): every extension is compiled with ``-ffp-contract=off`` and never
``-ffast-math``, so the compiled kernels reproduce the pure-numpy reference to
the last bit on every platform. The parity tests enforce this in CI.

Implementation note: we drive setuptools' ``build_ext`` through a bare
``Distribution`` (never ``setup()``), which deliberately skips
``parse_config_files`` — that avoids setuptools re-reading this project's
``pyproject.toml`` (and tripping on unrelated metadata) during the extension
build.
"""

from __future__ import annotations

import os
import sysconfig

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# dotted module name -> source .pyx (relative to project root)
PYX_MODULES = {
    "pystatistics.timeseries._arima_kalman_kernel":
        "pystatistics/timeseries/_arima_kalman_kernel.pyx",
    "pystatistics.survival._concordance_fenwick":
        "pystatistics/survival/_concordance_fenwick.pyx",
    "pystatistics.timeseries._ets_recursion":
        "pystatistics/timeseries/_ets_recursion.pyx",
}

# No -ffast-math, no -Ofast: those enable reassociation and break last-bit
# parity. -ffp-contract=off additionally forbids FMA contraction.
UNIX_FLAGS = ["-O3", "-ffp-contract=off"]
MSVC_FLAGS = ["/O2", "/fp:precise"]  # only used if a Windows target is added


class CythonBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return

        import numpy as np
        from Cython.Build import cythonize
        from setuptools import Distribution, Extension

        is_msvc = os.name == "nt" and sysconfig.get_platform().startswith("win")
        flags = MSVC_FLAGS if is_msvc else UNIX_FLAGS

        extensions = [
            Extension(
                name,
                [src],
                include_dirs=[np.get_include()],
                extra_compile_args=flags,
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
            for name, src in PYX_MODULES.items()
        ]
        ext_modules = cythonize(
            extensions,
            compiler_directives={"language_level": "3"},
            force=True,
        )

        # Bare Distribution (NOT setup()) -> no config-file parsing.
        dist = Distribution({"ext_modules": ext_modules})
        cmd = dist.get_command_obj("build_ext")
        cmd.inplace = 1
        cmd.ensure_finalized()
        cmd.run()

        # The wheel is now platform-specific and must carry the compiled
        # objects. infer_tag stamps the correct platform/ABI tag.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        for name in PYX_MODULES:
            rel = name.replace(".", "/") + _ext_suffix()
            build_data.setdefault("force_include", {})[rel] = rel


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"
