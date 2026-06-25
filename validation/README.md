# pystatsval — pystatistics validation harness

The reusable machinery that produces **frozen validation evidence** for
`pystatistics`. It lives here, with the library it validates, but is **not part of
the shipped `pystatistics` wheel** (the wheel packages only `pystatistics`). That is
deliberate:

- the harness is developed alongside the library (one source of truth), **and**
- it evolves on its own cadence — regenerating evidence or fixing the harness must
  never require cutting a `pystatistics` release.

It is **run against a PyPI-installed** `pystatistics` of the exact version under
validation — never an editable checkout. `device.require_pypi()` enforces this.

## What's here (subsystem-agnostic core)

| module | job |
|---|---|
| `timing` | warmup + timed repeats; median headline (NON-DETERMINISTIC by design) |
| `record` | the uniform benchmark-record envelope (`validation-run/v1` record shape) |
| `estimates` | reusable estimate summaries (e.g. covariance Frobenius norm / log-det) |
| `device` | environment manifest; version read from the imported module; PyPI guard |
| `serialize` | assemble + write a run to the artifact schema |
| `rrunner` | generic R-subprocess bridge (timing done inside R) |

Subsystem-specific drivers (which estimator to fit, which R script to call, the data
curation) live in the **pystatistics-validation** repo's `drivers/` and import this
package. The harness never modifies the library; it only consumes it.

## Use

```bash
# in the validation env, alongside a PyPI pystatistics of the target version:
pip install pystatistics==<X.Y.Z>
pip install -e /path/to/pystatistics/validation   # this package
pytest                                              # harness unit tests
```

```python
from pystatsval import timing, record, estimates, device, serialize

env = device.env_manifest(device="cpu", host="mac")
device.require_pypi(env)            # refuse non-PyPI installs for canonical evidence
wall, fit = timing.time_call(lambda: my_fit(), warmup=1, reps=5)
rec = record.make_record(engine="pystatistics:cpu", dataset="wvs", n=n, p=p,
                         loglik=fit.loglik, wall=wall,
                         summary=estimates.summarize_covariance(fit.sigma))
run = serialize.build_run(env=env, config={...}, records=[rec])
serialize.write_run("artifacts/<subsystem>/v<X.Y.Z>/runs/<run>.json", run)
```
