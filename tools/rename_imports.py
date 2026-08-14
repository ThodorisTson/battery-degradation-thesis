"""Rewrite import statements after the module rename.

Run once, from the repository root, after copying the source files across.
Dry run first, read the report, then apply.

    python tools/rename_imports.py            # dry run, changes nothing
    python tools/rename_imports.py --apply    # write the changes

Only import lines are touched. Every rewrite is printed with its file and
line number so the diff can be checked before committing.

Delete this file once the migration is complete.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Old top-level module name -> new dotted path inside the package.
RENAMES = {
    "degradation_xu": "degradation.xu",
    "degradation_shi": "degradation.shi",
    "degradation_subgradient": "degradation.subgradient",
    "degradation_plots_multiyear": "degradation.plots_multiyear",
    "degradation_plots": "degradation.plots",
    "wp2_common": "degradation.site",
    "wp2_econ": "degradation.economics",
    "thesis_style": "degradation.style",
}

SEARCH_DIRS = ["src", "scripts", "figures", "verification"]

# Longest names first, so degradation_plots_multiyear is not partly matched
# by degradation_plots.
_ORDERED = sorted(RENAMES.items(), key=lambda kv: -len(kv[0]))

# Matches:  from <mod> import ...   /   import <mod>   /   import <mod> as x
_PATTERNS = [
    (old, new, re.compile(rf"^(\s*from\s+){re.escape(old)}(\s+import\b)"))
    for old, new in _ORDERED
] + [
    (old, new, re.compile(rf"^(\s*import\s+){re.escape(old)}(\s*(?:as\s+\w+)?\s*)$"))
    for old, new in _ORDERED
]


def rewrite_file(path: Path, apply: bool) -> list[tuple[int, str, str]]:
    """Return the list of (line_number, before, after) for this file."""
    try:
        original = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        print(f"  SKIP (not utf-8): {path}")
        return []

    changes: list[tuple[int, str, str]] = []
    out = list(original)

    for i, line in enumerate(original):
        new_line = line
        for _old, new, pattern in _PATTERNS:
            candidate = pattern.sub(rf"\g<1>{new}\g<2>", new_line)
            if candidate != new_line:
                new_line = candidate
                break
        if new_line != line:
            changes.append((i + 1, line.rstrip("\n"), new_line.rstrip("\n")))
            out[i] = new_line

    if changes and apply:
        path.write_text("".join(out), encoding="utf-8")

    return changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default is a dry run)")
    ap.add_argument("--root", default=".",
                    help="repository root (default: current directory)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "pyproject.toml").exists():
        print(f"No pyproject.toml in {root}. Run from the repository root.")
        return 1

    files: list[Path] = []
    for d in SEARCH_DIRS:
        files.extend(sorted((root / d).rglob("*.py")))

    if not files:
        print(f"No .py files found under {SEARCH_DIRS} in {root}")
        return 1

    total = 0
    touched = 0
    for f in files:
        changes = rewrite_file(f, args.apply)
        if changes:
            touched += 1
            total += len(changes)
            print(f"\n{f.relative_to(root)}")
            for lineno, before, after in changes:
                print(f"  {lineno:>5}  - {before.strip()}")
                print(f"  {'':>5}  + {after.strip()}")

    mode = "APPLIED" if args.apply else "DRY RUN (nothing written)"
    print(f"\n{mode}: {total} import lines in {touched} of {len(files)} files")

    if not args.apply and total:
        print("\nRe-run with --apply once the changes above look correct.")

    # Leftover references that are not import statements, for example strings
    # or comments naming an old module. These are not rewritten automatically.
    print("\nRemaining textual references to old module names:")
    found_any = False
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for old in RENAMES:
            for m in re.finditer(re.escape(old), text):
                line_no = text[:m.start()].count("\n") + 1
                line = text.splitlines()[line_no - 1].strip()
                if line.startswith(("import ", "from ")):
                    continue
                found_any = True
                print(f"  {f.relative_to(root)}:{line_no}  {line[:100]}")
    if not found_any:
        print("  none")

    return 0


if __name__ == "__main__":
    sys.exit(main())
