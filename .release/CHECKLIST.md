# Release Checklist

> `release.py` only automates the version bump in `pyproject.toml` and
> `<package>/__init__.py`. Everything else is by hand — on purpose.
> The CHANGELOG entry in particular is authorship work, not mechanical:
> `UNRELEASED.md` is written in internal voice, `CHANGELOG.md` is public.

## Major releases (X.0.0) only — clear scheduled removals FIRST

Before cutting a major, open `docs/ROADMAP.md` → "Deprecations & scheduled
removals" and **remove every alias/shim whose "Remove in" is this major** (delete
the code path, drop it from any `Literal`/type, delete the deprecation warning and
its test, and delete the table row). A deprecation that is never removed is just
permanent cruft with a warning. Do this as its own commit before the release bump.

## Release flow

1. **Sanity check:** `python .release/release.py --status`

2. **Write the `CHANGELOG.md` entry by hand.** Open `UNRELEASED.md` for
   reference, then translate it into user-facing prose at the top of
   `CHANGELOG.md` under `## X.Y.Z`. Do NOT copy verbatim. Strip:
   - internal rule numbers (e.g. "Rule 9", "Rule 10")
   - internal project names
   - test counts / coverage numbers / process rationale
   - references to `CLAUDE.md` by name or implication

   Keep:
   - API additions / removals
   - behaviour changes a user would notice
   - bug fixes (in terms of the user-visible symptom)
   - performance numbers (with shape context)

3. **Update `README.md` "What's New".** Same translation rules as the
   CHANGELOG. The README should lead with library identity; the changelog
   lives near the bottom.

4. **Bump versions:** `python .release/release.py --bump X.Y.Z` — this
   only touches `pyproject.toml` and `<package>/__init__.py`.

5. **Review:** `git diff`

6. **Commit and tag:** ⚠️⚠️ A commit ships the *index*, not the working tree.
   Stage **every** change that belongs in this release — any tracked file left
   modified-but-unstaged is silently dropped from the release commit. The files
   below are the minimum; your source changes must be staged too. README.md is
   the most commonly forgotten one, but the rule applies to every file.
   ```
   git add pyproject.toml <package>/__init__.py CHANGELOG.md README.md
   git add <source files changed for this release>
   git diff --name-only    # must be empty, or only unrelated WIP
   git commit -m "Release X.Y.Z"
   git tag vX.Y.Z
   git push origin <branch>
   git push origin vX.Y.Z
   ```

7. **Reset UNRELEASED.md** — only now, after the release commit is in:
   ```
   python .release/release.py --reset-unreleased
   git add .release/UNRELEASED.md
   git commit -m "Reset UNRELEASED.md for next cycle"
   git push
   ```

8. **Publish to PyPI via GitHub release:**
   ```
   # Notes = this version's CHANGELOG section only. Do NOT pass the whole
   # CHANGELOG.md — it exceeds GitHub's 125k release-body limit (bit on
   # 4.6.5) and buries the release under old entries.
   awk '/^## X\.Y\.Z$/{f=1} /^## <previous version>$/{f=0} f' CHANGELOG.md > /tmp/notes.md
   gh release create vX.Y.Z --title 'vX.Y.Z' --notes-file /tmp/notes.md
   ```
   This triggers `publish.yml` → PyPI.

9. **Verify:** `pip install <package>==X.Y.Z`

## Why there is no `--commit` mode

A previous version of this script had a `--commit` flag that did the
changelog prepend, the UNRELEASED reset, and the git commit/tag/push in
one shot. Two problems:

- It copied `UNRELEASED.md` verbatim into `CHANGELOG.md`. `UNRELEASED.md`
  is written in internal developer voice; `CHANGELOG.md` is read by
  strangers on PyPI. That's a Rule-10 audience violation, and the
  automation made it the path of least resistance.
- It coupled the mechanical bump with the human authorship steps, so
  there was no natural point to review the translation before the
  commit went out.

Doing the authorship by hand, then using the script only for the literal
version strings, is slower on paper and safer in practice.

## Historical foot-guns

- **v1.6.1 (2026-ish):** release commit missed unstaged source edits
  because only tracked files were staged. Mitigation: always run
  `git status` before `git commit` and confirm the diff contains every
  file the release should ship.
- **Multiple PyPI publish failures:** every one has been a missed manual
  step from this list. Work top to bottom; do not skip.
