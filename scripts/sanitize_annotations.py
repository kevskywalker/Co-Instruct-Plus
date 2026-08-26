#!/usr/bin/env python3
"""Rewrite absolute lab image paths in annotation JSON to portable relative paths.

Example:
  python scripts/sanitize_annotations.py \\
    --input-dir training_jsons \\
    --output-dir data/annotations \\
    --image-root /path/to/Co-Instruct-plus \\
    --check-exists --sample 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

KNOWN_PREFIXES = (
    "/home/zhw/IQA/Co-Instruct-plus/",
    "/home/zhw/IQA/Co-Instruct-Plus/",
    "/mnt/nvme/zhw/Co-Instruct-plus/",
    "/mnt/nvme/zhw/Co-Instruct-Plus/",
    "/home/yuhan/IQA/Co-Instruct-plus/",
    "/home/yuhan/IQA/Co-Instruct-Plus/",
    "/home/yuhan/Data-DeQA-Score/",
)


def to_relative(path: str) -> str:
    if not isinstance(path, str):
        return path
    if path.startswith("http://") or path.startswith("https://"):
        return path
    for prefix in KNOWN_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    while path.startswith("./"):
        path = path[2:]
    if os.path.isabs(path) and "/data/" in path:
        path = path[path.index("data/") :]
    return path


def scrub_lab_paths(text: str) -> str:
    """Strip lab absolute roots anywhere in a string (e.g. composite id fields)."""
    if not isinstance(text, str):
        return text
    for prefix in KNOWN_PREFIXES:
        text = text.replace(prefix, "")
    text = text.replace("./data/", "data/")
    return text


def rewrite_value(value):
    if isinstance(value, str):
        # Path-like or composite ids that embed paths
        if (
            "/" in value
            or value.startswith("data/")
            or any(p.rstrip("/") in value for p in KNOWN_PREFIXES)
        ):
            if "|" in value or "->" in value:
                return scrub_lab_paths(value)
            return to_relative(scrub_lab_paths(value))
        return value
    if isinstance(value, list):
        return [rewrite_value(v) for v in value]
    if isinstance(value, dict):
        return {k: rewrite_value(v) for k, v in value.items()}
    return value


def rewrite_sample(sample: dict) -> dict:
    return rewrite_value(sample)


def contains_machine_path(data) -> bool:
    """Return whether serialized annotation data still contains local roots."""
    serialized = json.dumps(data, ensure_ascii=False)
    return bool(re.search(r"/(?:home|mnt)/", serialized))


def check_paths(samples: list, image_root: str, sample_n: int) -> tuple[int, int, list]:
    ok = 0
    bad = 0
    examples = []
    n = min(sample_n, len(samples))
    step = max(1, len(samples) // n) if n else 1
    checked = 0
    for i in range(0, len(samples), step):
        if checked >= n:
            break
        s = samples[i]
        im = s.get("image") or s.get("images")
        if im is None:
            continue
        items = im if isinstance(im, list) else [im]
        for p in items:
            if not isinstance(p, str) or p.startswith("http"):
                continue
            full = p if os.path.isabs(p) else os.path.normpath(os.path.join(image_root, p))
            checked += 1
            if os.path.exists(full) and os.path.getsize(full) > 0:
                ok += 1
            else:
                bad += 1
                if len(examples) < 5:
                    examples.append(full)
            if checked >= n:
                break
    return ok, bad, examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--check-exists", action="store_true")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--glob", type=str, default="*.json")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        print(f"[error] no files matching {args.glob} in {in_dir}", file=sys.stderr)
        return 1

    for fp in files:
        print(f"[sanitize] {fp.name} ...", flush=True)
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            print(f"[warn] skip non-list JSON: {fp}")
            continue
        rewritten = [rewrite_sample(s) for s in data]
        if contains_machine_path(rewritten):
            print(
                f"[error] sanitized output still contains /home/ or /mnt/ paths: {fp}",
                file=sys.stderr,
            )
            return 1
        out_path = out_dir / fp.name
        out_path.write_text(json.dumps(rewritten, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out_path} ({len(rewritten)} samples)")
        if args.check_exists:
            if not args.image_root:
                print("  [warn] --check-exists needs --image-root", file=sys.stderr)
            else:
                ok, bad, examples = check_paths(rewritten, args.image_root, args.sample)
                print(f"  path check: ok={ok} bad={bad} (sampled)")
                for e in examples:
                    print(f"    missing: {e}")

    print("[sanitize] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
