"""Inspect and explicitly delete Hugging Face model cache entries."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from .models import default_hub_root, hub_name, models_root, resolve_model
from .resources import directory_size_bytes, format_bytes


@dataclass(frozen=True)
class CacheEntry:
    model: str
    path: str
    bytes: int
    symlink: bool


def cache_entries() -> list[CacheEntry]:
    roots = [models_root(), default_hub_root()]
    seen: set[Path] = set()
    entries: list[CacheEntry] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("models--*")):
            resolved = path.resolve() if path.exists() else path
            if resolved in seen:
                continue
            seen.add(resolved)
            entries.append(
                CacheEntry(
                    model=path.name.removeprefix("models--").replace("--", "/"),
                    path=str(path),
                    bytes=directory_size_bytes(path),
                    symlink=path.is_symlink(),
                )
            )
    return sorted(entries, key=lambda entry: entry.bytes, reverse=True)


def delete_cached_model(model_name: str, *, yes: bool = False) -> list[Path]:
    if not model_name:
        raise ValueError("MODEL must be an explicitly named model")
    spec = resolve_model(model_name)
    candidates = [
        models_root() / hub_name(spec),
        default_hub_root() / hub_name(spec),
        models_root() / spec.name,
    ]
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    if not yes:
        raise RuntimeError("pass --yes to delete cached files")
    for path in existing:
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    return existing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show HF cache entries")
    delete = sub.add_parser("delete", help="delete one explicitly named cached model")
    delete.add_argument("model")
    delete.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        entries = cache_entries()
        total = sum(entry.bytes for entry in entries)
        print(f"total: {format_bytes(total)}")
        for entry in entries[:20]:
            marker = " -> symlink" if entry.symlink else ""
            print(f"{format_bytes(entry.bytes):>10}  {entry.model}  {entry.path}{marker}")
        return 0

    if args.command == "delete":
        deleted = delete_cached_model(args.model, yes=args.yes)
        if not deleted:
            print(f"no cache entries found for {args.model}")
        for path in deleted:
            print(f"deleted {path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
