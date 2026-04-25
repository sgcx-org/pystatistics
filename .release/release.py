#!/usr/bin/env python3
"""
Release utility for pystatistics / pystatsbio.

This script ONLY automates the mechanical, judgement-free parts of a
release. Everything that needs human authorship — in particular, the
public-facing CHANGELOG.md entry — is left to you.

Why: UNRELEASED.md is written in internal developer voice (rule numbers,
project names, test counts, development rationale). CHANGELOG.md is a
public file read by strangers on PyPI. Copying one into the other is a
Rule-10 violation. The translation must be done by hand.

Usage:
    python .release/release.py --status              # show version state + unreleased summary
    python .release/release.py --check X.Y.Z         # dry-run: validate the bump
    python .release/release.py --bump X.Y.Z          # bump pyproject.toml + __init__.py
    python .release/release.py --reset-unreleased    # wipe UNRELEASED.md to template
                                                       (only after CHANGELOG entry is written)

What this script does:
    --bump:
      1. Validates new version > current version
      2. Writes new version to pyproject.toml
      3. Writes new version to <package>/__init__.py

    --reset-unreleased:
      1. Overwrites UNRELEASED.md with the empty template

What this script does NOT do (you must do these yourself):
    - Write the CHANGELOG.md entry. Translate UNRELEASED.md into
      user-facing prose by hand. Do not copy verbatim.
    - Update the README "What's New" section.
    - git add / commit / tag / push.
    - gh release create.

Typical flow:
    1. `--status`                                       (sanity check)
    2. Write the CHANGELOG.md entry by hand.            (translation)
    3. Update README.md "What's New" by hand.           (translation)
    4. `--bump X.Y.Z`                                   (mechanical)
    5. `git diff`                                       (review)
    6. git add / commit / tag / push                    (manual)
    7. `--reset-unreleased`                             (only after commit)
    8. Commit the reset, then `gh release create`.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = REPO_ROOT / ".release"
UNRELEASED_PATH = RELEASE_DIR / "UNRELEASED.md"


# ---------------------------------------------------------------------------
# Discovery
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
    toml = (REPO_ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', toml, re.MULTILINE)
    if not m:
        raise ValueError("Cannot find version in pyproject.toml")
    return m.group(1)


def get_init_version(package: str) -> str:
    init = (REPO_ROOT / package / "__init__.py").read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    if not m:
        raise ValueError(f"Cannot find __version__ in {package}/__init__.py")
    return m.group(1)


def get_changelog_version() -> str | None:
    changelog = REPO_ROOT / "CHANGELOG.md"
    if not changelog.exists():
        return None
    text = changelog.read_text()
    m = re.search(r'^## (\d+\.\d+\.\d+)', text, re.MULTILINE)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# UNRELEASED.md
# ---------------------------------------------------------------------------

EMPTY_UNRELEASED = textwrap.dedent("""\
    # Unreleased Changes

    > This file tracks all changes since the last stable release.
    > Updated by whoever makes a change, on whatever machine.
    > Synced via git so all sessions (Mac, Linux, etc.) see the same state.
    >
    > When ready to release, run: `python .release/release.py --status`
    > and follow the manual release flow in the script docstring.

    ## Changes

    *(empty — no unreleased changes yet)*
""")


def get_unreleased_content() -> str:
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
    UNRELEASED_PATH.write_text(EMPTY_UNRELEASED)


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

    print("\nUnreleased changes (internal voice — do NOT paste into CHANGELOG.md):")
    if unreleased:
        lines = unreleased.split("\n")
        for line in lines[:12]:
            print(f"  {line}")
        if len(lines) > 12:
            print(f"  ... ({len(lines) - 12} more lines)")
    else:
        print("  (none)")


def cmd_check(new_version: str) -> None:
    package = find_package()
    current = get_current_version()
    validate_version(new_version, current)

    print(f"Package:    {package}")
    print(f"Current:    {current}")
    print(f"New:        {new_version}")
    print("\nWould update (only these two files):")
    print(f"  pyproject.toml:          {current} → {new_version}")
    print(f"  {package}/__init__.py:   {current} → {new_version}")
    print("\nThis script will NOT touch CHANGELOG.md, README.md, UNRELEASED.md, or git.")


def cmd_bump(new_version: str) -> None:
    package = find_package()
    current = get_current_version()
    validate_version(new_version, current)

    print(f"Bumping {package} {current} → {new_version}\n")

    bump_pyproject(new_version)
    print(f"  ✓ pyproject.toml → {new_version}")

    bump_init(package, new_version)
    print(f"  ✓ {package}/__init__.py → {new_version}")

    print(f"\n{'='*50}")
    print(f"  Version bumped to {new_version}")
    print(f"{'='*50}")
    print("\nRemaining steps (all manual):")
    print("  1. Verify CHANGELOG.md has a hand-written entry for this version")
    print("     (do NOT paste UNRELEASED.md verbatim — translate to user-facing prose).")
    print("  2. Verify README.md 'What's New' reflects the same user-facing summary.")
    print("  3. git diff     # review")
    print(f"  4. git add pyproject.toml {package}/__init__.py CHANGELOG.md README.md")
    print(f"  5. git commit -m 'Release {new_version}'")
    print(f"  6. git tag v{new_version}")
    print(f"  7. git push origin <branch> && git push origin v{new_version}")
    print("  8. python .release/release.py --reset-unreleased")
    print("     git add .release/UNRELEASED.md && git commit -m 'Reset UNRELEASED.md'")
    print("     git push")
    print(f"  9. gh release create v{new_version} --title 'v{new_version}' --notes-file CHANGELOG.md")
    print(f" 10. Verify: pip install {package}=={new_version}")


def cmd_reset_unreleased() -> None:
    reset_unreleased()
    print(f"  ✓ {UNRELEASED_PATH.relative_to(REPO_ROOT)} reset to empty template")
    print("\nRemember to commit the reset:")
    print("  git add .release/UNRELEASED.md")
    print("  git commit -m 'Reset UNRELEASED.md for next cycle'")
    print("  git push")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

USAGE = """\
Usage:
  python .release/release.py --status              # show version state
  python .release/release.py --check X.Y.Z         # dry-run: validate the bump
  python .release/release.py --bump X.Y.Z          # bump pyproject.toml + __init__.py
  python .release/release.py --reset-unreleased    # wipe UNRELEASED.md to template

See the module docstring for the full manual release flow.
"""


def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--status":
        cmd_status()
    elif arg == "--check":
        if len(sys.argv) < 3:
            print("Usage: python .release/release.py --check X.Y.Z")
            sys.exit(1)
        cmd_check(sys.argv[2])
    elif arg == "--bump":
        if len(sys.argv) < 3:
            print("Usage: python .release/release.py --bump X.Y.Z")
            sys.exit(1)
        cmd_bump(sys.argv[2])
    elif arg == "--reset-unreleased":
        cmd_reset_unreleased()
    else:
        print(f"Unknown command: {arg}\n")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
