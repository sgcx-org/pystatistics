#!/usr/bin/env python3
"""
Release utility for pystatistics / pystatsbio.

Usage:
    python .release/release.py 1.2.0            # bump + changelog + reset
    python .release/release.py --commit 1.2.0   # bump + changelog + reset + git commit/tag/push
    python .release/release.py --check 1.2.0    # show what would change
    python .release/release.py --status         # show current state

What it does (in order):
    1. Reads .release/UNRELEASED.md for the change log
    2. Validates the new version > current version
    3. Updates pyproject.toml version
    4. Updates __init__.py __version__
    5. Prepends the new entry to CHANGELOG.md
    6. Resets UNRELEASED.md for the next cycle
    7. Prints a checklist of remaining manual steps

What it does NOT do (you must do these yourself):
    - Update the README (version badges, "what's new" prose) — pre-stage
      those edits with `git add README.md` before running --commit so they
      ride along in the release commit
    - Create the GitHub release (`gh release create vX.Y.Z`) — this is what
      triggers the publish.yml workflow that pushes to PyPI
    - In plain (non --commit) mode: git add / commit / push / tag

Place this file in .release/ at the repo root.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = REPO_ROOT / ".release"
UNRELEASED_PATH = RELEASE_DIR / "UNRELEASED.md"


# ---------------------------------------------------------------------------
# Discovery: find the package name and version locations
# ---------------------------------------------------------------------------

def find_package() -> str:
    """Find the Python package directory (has __init__.py with __version__)."""
    for d in sorted(REPO_ROOT.iterdir()):
        if d.is_dir() and (d / "__init__.py").exists():
            init_text = (d / "__init__.py").read_text()
            if "__version__" in init_text:
                return d.name
    raise FileNotFoundError("Cannot find package with __version__ in __init__.py")


def get_current_version() -> str:
    """Read current version from pyproject.toml."""
    toml = (REPO_ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', toml, re.MULTILINE)
    if not m:
        raise ValueError("Cannot find version in pyproject.toml")
    return m.group(1)


def get_init_version(package: str) -> str:
    """Read __version__ from __init__.py."""
    init = (REPO_ROOT / package / "__init__.py").read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    if not m:
        raise ValueError(f"Cannot find __version__ in {package}/__init__.py")
    return m.group(1)


def get_changelog_version() -> str | None:
    """Read the latest version from CHANGELOG.md."""
    changelog = REPO_ROOT / "CHANGELOG.md"
    if not changelog.exists():
        return None
    text = changelog.read_text()
    m = re.search(r'^## (\d+\.\d+\.\d+)', text, re.MULTILINE)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Unreleased changes
# ---------------------------------------------------------------------------

def get_unreleased_content() -> str:
    """Read the ## Changes section from UNRELEASED.md.

    Captures everything after `## Changes` to EOF — works with both flat
    bullet lists and structured `### Summary/Added/Changed/Fixed` subsections.
    """
    if not UNRELEASED_PATH.exists():
        return ""
    text = UNRELEASED_PATH.read_text()
    m = re.search(r'^## Changes\s*\n(.*)\Z', text, re.MULTILINE | re.DOTALL)
    if not m:
        return ""
    content = m.group(1).strip()
    if not content or content == "*(empty — no unreleased changes yet)*":
        return ""
    return content


def reset_unreleased() -> None:
    """Reset UNRELEASED.md to empty template."""
    UNRELEASED_PATH.write_text(textwrap.dedent("""\
        # Unreleased Changes

        > This file tracks all changes since the last stable release.
        > Updated by whoever makes a change, on whatever machine.
        > Synced via git so all sessions (Mac, Linux, etc.) see the same state.
        >
        > When ready to release, run: `python .release/release.py <version>`
        > That script uses this file to build the CHANGELOG entry, bumps versions
        > everywhere, and resets this file for the next cycle.

        ## Changes

        *(empty — no unreleased changes yet)*
    """))


# ---------------------------------------------------------------------------
# Version bumping
# ---------------------------------------------------------------------------

def bump_pyproject(new_version: str) -> None:
    path = REPO_ROOT / "pyproject.toml"
    text = path.read_text()
    text = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{new_version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    path.write_text(text)


def bump_init(package: str, new_version: str) -> None:
    path = REPO_ROOT / package / "__init__.py"
    text = path.read_text()
    text = re.sub(
        r'(__version__\s*=\s*)"[^"]+"',
        rf'\g<1>"{new_version}"',
        text,
        count=1,
    )
    path.write_text(text)


def update_changelog(new_version: str, content: str) -> None:
    """Prepend new version entry to CHANGELOG.md."""
    path = REPO_ROOT / "CHANGELOG.md"
    if not path.exists():
        path.write_text("# Changelog\n")

    existing = path.read_text()

    # Build new entry
    entry = f"## {new_version}\n\n{content}\n\n"

    # Insert after "# Changelog\n"
    if existing.startswith("# Changelog"):
        new_text = existing.replace("# Changelog\n", f"# Changelog\n\n{entry}", 1)
    else:
        new_text = f"# Changelog\n\n{entry}{existing}"

    path.write_text(new_text)


# ---------------------------------------------------------------------------
# Version validation
# ---------------------------------------------------------------------------

def parse_version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def validate_version(new: str, current: str) -> None:
    try:
        new_t = parse_version(new)
        cur_t = parse_version(current)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid version format: {e}") from e

    if new_t <= cur_t:
        raise ValueError(
            f"New version {new} must be greater than current {current}"
        )


# ---------------------------------------------------------------------------
# Git operations (for --commit mode)
# ---------------------------------------------------------------------------

def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command at REPO_ROOT. Fails loudly (Rule 1) unless check=False."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def get_current_branch() -> str:
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def tag_exists(tag: str) -> bool:
    return bool(run_git(["tag", "-l", tag], check=False).stdout.strip())


def commit_and_push(new_version: str, package: str) -> None:
    """Stage release files, commit, tag, and push. Fails loud on any error.

    Stages release.py's own mutations on top of whatever the user has already
    staged (e.g. README changes), so pre-staged edits ride along in the
    release commit.
    """
    tag = f"v{new_version}"
    if tag_exists(tag):
        raise RuntimeError(f"Tag {tag} already exists — aborting")

    branch = get_current_branch()
    files = [
        "pyproject.toml",
        f"{package}/__init__.py",
        "CHANGELOG.md",
        ".release/UNRELEASED.md",
    ]
    run_git(["add", *files])
    run_git(["commit", "-m", f"Release {tag}"])
    print(f"  ✓ git commit: Release {tag}")

    run_git(["tag", tag])
    print(f"  ✓ git tag: {tag}")

    run_git(["push", "origin", branch])
    print(f"  ✓ git push: {branch}")

    run_git(["push", "origin", tag])
    print(f"  ✓ git push: {tag}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status() -> None:
    package = find_package()
    v_toml = get_current_version()
    v_init = get_init_version(package)
    v_log = get_changelog_version()
    unreleased = get_unreleased_content()

    print(f"Package:      {package}")
    print(f"pyproject:    {v_toml}")
    print(f"__init__:     {v_init}")
    print(f"CHANGELOG:    {v_log or '(none)'}")

    if v_toml != v_init:
        print(f"\n  WARNING: pyproject ({v_toml}) != __init__ ({v_init})")
    if v_log and v_toml != v_log:
        print(f"\n  WARNING: pyproject ({v_toml}) != CHANGELOG ({v_log})")

    print(f"\nUnreleased changes:")
    if unreleased:
        lines = unreleased.split("\n")
        sections = [l for l in lines if l.startswith("### ")]
        if sections:
            print(f"  {len(lines)} lines across {len(sections)} section(s):")
            for s in sections:
                print(f"    {s}")
        else:
            for line in lines[:10]:
                print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... ({len(lines) - 10} more lines)")
    else:
        print("  (none)")


def cmd_check(new_version: str) -> None:
    package = find_package()
    current = get_current_version()
    validate_version(new_version, current)
    unreleased = get_unreleased_content()

    print(f"Package:    {package}")
    print(f"Current:    {current}")
    print(f"New:        {new_version}")
    print(f"\nWould update:")
    print(f"  pyproject.toml:          {current} → {new_version}")
    print(f"  {package}/__init__.py:   {current} → {new_version}")
    print(f"  CHANGELOG.md:            prepend {new_version} entry")
    print(f"  .release/UNRELEASED.md:  reset to empty")

    if unreleased:
        print(f"\nChangelog content ({len(unreleased.split(chr(10)))} lines):")
        print(textwrap.indent(unreleased[:500], "  "))
    else:
        print("\n  WARNING: UNRELEASED.md is empty — changelog entry will be blank")


def cmd_release(new_version: str, commit: bool = False) -> None:
    package = find_package()
    current = get_current_version()
    validate_version(new_version, current)
    unreleased = get_unreleased_content()

    if not unreleased:
        print("ERROR: .release/UNRELEASED.md has no changes.")
        print("Add changes there before releasing.")
        sys.exit(1)

    # Pre-flight: if --commit, refuse when tag already exists (before any mutation)
    if commit and tag_exists(f"v{new_version}"):
        print(f"ERROR: Tag v{new_version} already exists — aborting before any file changes.")
        sys.exit(1)

    # Pre-flight: in --commit mode, refuse if there are unstaged source
    # edits in the package. The release commit only stages pyproject.toml,
    # __init__.py, CHANGELOG.md, UNRELEASED.md, plus anything the user
    # pre-staged with `git add`. Unstaged source edits would be left out
    # of the release commit, producing a tagged wheel that is missing the
    # very fixes it is supposed to ship.  (This happened with v1.6.1 and
    # required a same-day 1.6.2 patch-of-a-patch.)
    if commit:
        package_dir = REPO_ROOT / package
        # `git diff --name-only` lists UNSTAGED changes (modified but not
        # added). We only care about changes inside the package directory.
        diff_proc = run_git(["diff", "--name-only"], check=False)
        unstaged = [
            line for line in diff_proc.stdout.splitlines()
            if line.startswith(f"{package}/")
        ]
        if unstaged:
            print(
                f"ERROR: {len(unstaged)} unstaged file(s) inside {package}/ "
                "would be left out of the release commit:"
            )
            for f in unstaged:
                print(f"    {f}")
            print()
            print(
                "Release aborted before any file changes. Either:\n"
                f"  - `git add` these files to include them in Release v{new_version}, or\n"
                "  - `git stash` them if they should not ship in this release.\n"
                "See: .release/CHECKLIST.md — 'Historical context' (v1.6.1 foot-gun)."
            )
            sys.exit(1)

    print(f"Releasing {package} {current} → {new_version}\n")

    # 1. Bump versions
    bump_pyproject(new_version)
    print(f"  ✓ pyproject.toml → {new_version}")

    bump_init(package, new_version)
    print(f"  ✓ {package}/__init__.py → {new_version}")

    # 2. Update changelog
    update_changelog(new_version, unreleased)
    print(f"  ✓ CHANGELOG.md — prepended {new_version} entry")

    # 3. Reset unreleased
    reset_unreleased()
    print(f"  ✓ .release/UNRELEASED.md — reset for next cycle")

    # 4. Commit + tag + push (only in --commit mode)
    if commit:
        print()
        commit_and_push(new_version, package)

    # 5. Checklist
    print(f"\n{'='*50}")
    print(f"  Version bumped to {new_version}")
    print(f"{'='*50}")
    if commit:
        print(f"\nRemaining steps (manual):")
        print(f"  1. Create GitHub Release (triggers PyPI publish):")
        print(f"       gh release create v{new_version} --title 'v{new_version}' --notes-file CHANGELOG.md")
        print(f"  2. Verify on PyPI:")
        print(f"       pip install {package}=={new_version}")
    else:
        print(f"\nRemaining steps (manual):")
        print(f"  1. Review the changes:")
        print(f"       git diff")
        print(f"  2. Commit:")
        print(f"       git add -A && git commit -m 'Release {new_version}'")
        print(f"  3. Push:")
        print(f"       git push origin main")
        print(f"  4. Tag:")
        print(f"       git tag v{new_version} && git push origin v{new_version}")
        print(f"  5. Create GitHub Release (triggers PyPI publish):")
        print(f"       gh release create v{new_version} --title 'v{new_version}' --notes-file CHANGELOG.md")
        print(f"  6. Verify on PyPI:")
        print(f"       pip install {package}=={new_version}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python .release/release.py <version>            # bump + changelog + reset")
        print("  python .release/release.py --commit <version>   # also commit + tag + push")
        print("  python .release/release.py --check <version>    # dry run")
        print("  python .release/release.py --status             # show current state")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--status":
        cmd_status()
    elif arg == "--check":
        if len(sys.argv) < 3:
            print("Usage: python .release/release.py --check <version>")
            sys.exit(1)
        cmd_check(sys.argv[2])
    elif arg == "--commit":
        if len(sys.argv) < 3:
            print("Usage: python .release/release.py --commit <version>")
            sys.exit(1)
        cmd_release(sys.argv[2], commit=True)
    else:
        cmd_release(arg, commit=False)


if __name__ == "__main__":
    main()
