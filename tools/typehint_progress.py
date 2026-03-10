#!/usr/bin/env python3
"""Report type-hint coverage for Python function signatures.

This script scans Python files under a target directory and reports:
- total number of functions (top-level + methods + async variants)
- number of fully-annotated functions
- number of partially-annotated functions
- number of unannotated functions

A function is considered fully annotated when all parameters (except ``self`` and
``cls``) and the return value have annotations.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FileStats:
    """Type-hint statistics for one file."""

    path: Path
    total: int
    full: int
    partial: int
    none: int

    @property
    def coverage(self) -> float:
        """Return full annotation coverage ratio in [0, 1]."""
        if self.total == 0:
            return 0.0
        return self.full / self.total


class FunctionTypeCounter(ast.NodeVisitor):
    """Count function annotation completeness using AST traversal."""

    def __init__(self) -> None:
        self.total = 0
        self.full = 0
        self.partial = 0
        self.none = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a regular function definition node and update counters."""
        self._consume_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition node and update counters."""
        self._consume_function(node)
        self.generic_visit(node)

    def _consume_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.total += 1
        annotated_params, total_params = _count_annotated_params(node)
        has_return = node.returns is not None

        if total_params == 0:
            if has_return:
                self.full += 1
            else:
                self.none += 1
            return

        if annotated_params == total_params and has_return:
            self.full += 1
        elif annotated_params == 0 and not has_return:
            self.none += 1
        else:
            self.partial += 1


def _iter_python_files(root: Path, excludes: Iterable[str]) -> list[Path]:
    excludes_set = {item.strip() for item in excludes if item.strip()}
    files: list[Path] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(ex) for ex in excludes_set):
            continue
        files.append(path)
    return sorted(files)


def _count_annotated_params(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, int]:
    args = node.args
    params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg is not None:
        params.append(args.vararg)
    if args.kwarg is not None:
        params.append(args.kwarg)

    filtered = [p for p in params if p.arg not in {"self", "cls"}]
    total = len(filtered)
    annotated = sum(1 for p in filtered if p.annotation is not None)
    return annotated, total


def _collect_file_stats(path: Path) -> FileStats:
    source = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(source, filename=str(path))
    counter = FunctionTypeCounter()
    counter.visit(tree)
    return FileStats(
        path=path,
        total=counter.total,
        full=counter.full,
        partial=counter.partial,
        none=counter.none,
    )


def _print_table(stats: list[FileStats], root: Path, min_functions: int) -> None:
    rows = [s for s in stats if s.total >= min_functions]
    rows.sort(key=lambda s: (s.coverage, -s.none, -s.total, str(s.path)))

    print("file,total,full,partial,none,coverage")
    for s in rows:
        rel = s.path.relative_to(root).as_posix()
        print(f"{rel},{s.total},{s.full},{s.partial},{s.none},{s.coverage:.3f}")


def _print_summary(stats: list[FileStats]) -> None:
    total = sum(s.total for s in stats)
    full = sum(s.full for s in stats)
    partial = sum(s.partial for s in stats)
    none = sum(s.none for s in stats)
    coverage = (full / total) if total else 0.0

    print("\nsummary")
    print(f"total_functions: {total}")
    print(f"fully_annotated: {full}")
    print(f"partially_annotated: {partial}")
    print(f"unannotated: {none}")
    print(f"full_coverage: {coverage:.3%}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the type-hint progress report."""
    parser = argparse.ArgumentParser(
        description="Measure per-file type-hint coverage for Python functions."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="phonopy",
        help="Directory to scan (default: phonopy)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["__pycache__"],
        help=(
            "Prefix path relative to target to exclude; may be repeated. "
            "Example: --exclude scripts"
        ),
    )
    parser.add_argument(
        "--min-functions",
        type=int,
        default=1,
        help="Only show files with at least this many functions (default: 1)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the report generation and return a process exit code."""
    args = parse_args()
    root = Path(args.target).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Target directory not found: {root}")
        return 1

    stats: list[FileStats] = []
    for path in _iter_python_files(root, args.exclude):
        try:
            stats.append(_collect_file_stats(path))
        except SyntaxError as exc:
            rel = path.relative_to(root).as_posix()
            print(f"skip_syntax_error: {rel}: {exc}")

    _print_table(stats, root=root, min_functions=args.min_functions)
    _print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
