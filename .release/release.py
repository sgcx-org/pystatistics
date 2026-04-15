#!/usr/bin/env python3
"""
Release utility for pystatistics / pystatsbio.

Usage:
    python .release/release.py 1.2.0          # bump + changelog + reset
    python .release/release.py --check        # show what would change
    python .release/release.py --status       # show current state

What it does (in order):
    1. Reads .release/UNRELEASED.md for the change log
    2. Validates the new version > current version
    3. Updates pyproject.toml version
    4. Updates __init__.py __version__
    5. Prepends the new entry to CHANGELOG.md
    6. Resets UNRELEASED.md for the next cycle
    7. Prints a checklist of remaining manual steps

What it does NOT do (you must do these yourself):
    - git add / commit / push
    - Create the GitHub release / tag
    - The publish.yml workflow handles PyPI automatically on release

Place this file in .release/ at the repo root.
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
    """Read the ## Changes section from UNRELEASED.md."""
    if not UNRELEASED_PATH.exists():
        return ""
    text = UNRELEASED_PATH.read_text()
    # Find everything after "## Changes"
    m = re.search(r'^## Changes\s*\n(.*)', text, re.MULTILINE | re.DOTALL)
    if not m:
        return ""
    content = m.group(1).strip()
    if content == "*(empty — no unreleased changes yet)*":
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
        for line in unreleased.split("\n")[:10]:
            print(f"  {line}")
        n_lines = len(unreleased.split("\n"))
        if n_lines > 10:
            print(f"  ... ({n_lines - 10} more lines)")
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


def cmd_release(new_version: str) -> None:
    package = find_package()
    current = get_current_version()
    validate_version(new_version, current)
    unreleased = get_unreleased_content()

    if not unreleased:
        print("ERROR: .release/UNRELEASED.md has no changes.")
        print("Add changes there before releasing.")
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

    # 4. Checklist
    print(f"\n{'='*50}")
    print(f"  Version bumped to {new_version}")
    print(f"{'='*50}")
    print(f"\nRemaining steps (manual):")
    print(f"  1. Review the changes:")
    print(f"       git diff")
    print(f"  2. Commit:")
    print(f"       git add -A && git commit -m 'Release {new_version}'")
    print(f"  3. Push:")
    print(f"       git push origin main")
    print(f"  4. Create GitHub Release (triggers PyPI publish):")
    print(f"       gh release create v{new_version} --title 'v{new_version}' --notes-file CHANGELOG.md")
    print(f"  5. Verify on PyPI:")
    print(f"       pip install {package}=={new_version}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python .release/release.py <version>   # bump + changelog + reset")
        print("  python .release/release.py --check <v> # dry run")
        print("  python .release/release.py --status    # show current state")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--status":
        cmd_status()
    elif arg == "--check":
        if len(sys.argv) < 3:
            print("Usage: python .release/release.py --check <version>")
            sys.exit(1)
        cmd_check(sys.argv[2])
    else:
        cmd_release(arg)


if __name__ == "__main__":
    main()
