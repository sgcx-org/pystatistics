"""Unit tests for the pystatsval harness core.

Normal / edge / failure coverage per the project's Rule 7. No GPU or R required:
timing uses trivial callables; device tests exercise install-source logic via a
fake distribution; serialize/record validate their contracts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pystatsval import device, estimates, record, serialize, timing


# --- timing --------------------------------------------------------------------

def test_time_call_returns_summary_and_result():
    summary, result = timing.time_call(lambda: 42, warmup=1, reps=3)
    assert result == 42
    assert summary["reps"] == 3
    assert summary["min_s"] <= summary["median_s"] <= summary["max_s"]
    assert len(summary["times_s"]) == 3


@pytest.mark.parametrize("kw", [{"warmup": -1}, {"reps": 0}])
def test_time_call_rejects_bad_args(kw):
    with pytest.raises(ValueError):
        timing.time_call(lambda: None, **kw)


# --- record --------------------------------------------------------------------

def test_make_record_core_fields_and_merge():
    rec = record.make_record(
        engine="pystatistics:cpu", dataset="wvs", n=100, p=5, loglik=-1.0,
        n_iter=3, converged=True, wall={"median_s": 0.5, "times_s": [0.5]},
        backend_name="cpu_cholesky_fp64", precision="fp64",
        summary={"sigma_fro": 2.0}, extra={"note": "x"})
    assert rec["engine"] == "pystatistics:cpu"
    assert rec["wall_median_s"] == 0.5
    assert rec["sigma_fro"] == 2.0 and rec["note"] == "x"
    assert rec["converged"] is True


def test_make_record_failure_path_no_raise():
    rec = record.make_record(engine="r:mvnmle", dataset="wvs", p=99,
                             wall=None, extra={"error": "p exceeds R limit"})
    assert rec["wall_median_s"] is None and rec["error"].startswith("p exceeds")


@pytest.mark.parametrize("bad", [{"engine": "", "dataset": "d"},
                                 {"engine": "e", "dataset": ""}])
def test_make_record_rejects_empty_ids(bad):
    with pytest.raises(ValueError):
        record.make_record(**bad)


# --- estimates -----------------------------------------------------------------

def test_summarize_covariance_pd():
    S = np.array([[2.0, 0.0], [0.0, 8.0]])
    out = estimates.summarize_covariance(S)
    assert out["sigma_diag"] == [2.0, 8.0]
    assert out["sigma_fro"] == pytest.approx(np.linalg.norm(S, "fro"))
    assert out["sigma_logdet"] == pytest.approx(np.log(16.0))
    assert out["sigma_full"] is not None


def test_summarize_covariance_none_and_large_p():
    assert estimates.summarize_covariance(None)["sigma_fro"] is None
    big = np.eye(5)
    assert estimates.summarize_covariance(big, full_max_p=2)["sigma_full"] is None


def test_summarize_covariance_rejects_nonsquare():
    with pytest.raises(ValueError):
        estimates.summarize_covariance(np.zeros((2, 3)))


# --- device / install-source guard --------------------------------------------

class _FakeDist:
    def __init__(self, direct_url):
        self._d = direct_url

    def read_text(self, name):
        return self._d if name == "direct_url.json" else None


def test_detect_install_source_pypi(monkeypatch):
    monkeypatch.setattr(device.metadata, "distribution", lambda n: _FakeDist(None))
    assert device.detect_install_source("pystatistics") == "pypi"


def test_detect_install_source_editable(monkeypatch):
    du = json.dumps({"url": "file:///x", "dir_info": {"editable": True}})
    monkeypatch.setattr(device.metadata, "distribution", lambda n: _FakeDist(du))
    assert device.detect_install_source("pystatistics") == "editable"


def test_detect_install_source_local_noneditable(monkeypatch):
    du = json.dumps({"url": "file:///x", "dir_info": {}})
    monkeypatch.setattr(device.metadata, "distribution", lambda n: _FakeDist(du))
    assert device.detect_install_source("pystatistics") == "editable"


def test_require_pypi_raises_on_editable():
    with pytest.raises(RuntimeError, match="PyPI"):
        device.require_pypi({"install_source": "editable",
                             "pystatistics_version": "3.18.0"})


def test_require_pypi_ok_on_pypi():
    device.require_pypi({"install_source": "pypi", "pystatistics_version": "3.18.0"})


def test_env_manifest_rejects_bad_device():
    with pytest.raises(ValueError):
        device.env_manifest(device="tpu")


# --- serialize -----------------------------------------------------------------

def _good_env():
    return {"pystatistics_version": "3.18.0", "install_source": "pypi", "device": "cpu"}


def test_build_run_ok(tmp_path):
    rec = record.make_record(engine="pystatistics:cpu", dataset="d", wall=None)
    run = serialize.build_run(env=_good_env(), config={"reps": 5}, records=[rec])
    assert run["schema"] == "validation-run/v1"
    out = serialize.write_run(tmp_path / "runs" / "r.json", run)
    assert json.loads(out.read_text())["env"]["install_source"] == "pypi"


def test_build_run_rejects_empty_records():
    with pytest.raises(ValueError, match="evidence-free"):
        serialize.build_run(env=_good_env(), config={}, records=[])


def test_build_run_rejects_env_missing_fields():
    with pytest.raises(ValueError, match="device"):
        serialize.build_run(env={"pystatistics_version": "3.18.0",
                                 "install_source": "pypi"},
                            config={}, records=[{"engine": "e"}])
