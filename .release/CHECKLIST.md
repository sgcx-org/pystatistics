# Release Checklist

> Every release must touch every item below. Some are automated by
> `python .release/release.py <version>`; the rest are manual and
> easy to forget. Work top to bottom.

## To Release

- [ ] **Init file** — `__version__` in `<package>/__init__.py`
      *(automated by `release.py`)*
- [ ] **publish.yml** — version references in
      `.github/workflows/publish.yml`; check before tagging
- [ ] **Git tag bump** ⚠️ — create and push the `vX.Y.Z` tag; the
      GitHub release / PyPI publish depends on it
- [ ] **Update CHANGELOG** — prepend the new version entry from
      `UNRELEASED.md` *(automated by `release.py`)*
- [ ] **Update README** ⚠️⚠️ — version badges, install snippets,
      "what's new" sections; **not** automated, easy to miss
- [ ] **Generate release on GitHub** — `gh release create vX.Y.Z`
      (this is what triggers the PyPI publish workflow)

## Notes

- `release.py` handles `pyproject.toml`, `__init__.py`, `CHANGELOG.md`,
  and resets `UNRELEASED.md`. Everything else above is on you.
- The ⚠️ items are the historical foot-guns — PyPI publish has
  failed at least three times from missing one of these. Double-check
  them before announcing the release.
