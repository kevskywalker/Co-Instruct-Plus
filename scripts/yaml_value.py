#!/usr/bin/env python3
"""Read one scalar value from a YAML mapping for shell launchers."""

from __future__ import annotations

import argparse
import sys

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("key")
    parser.add_argument("default", nargs="?")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    value = config.get(args.key, args.default)
    if value is None:
        return 0
    if args.list:
        if not isinstance(value, list):
            raise TypeError(f"{args.key} is not a YAML list")
        for item in value:
            print(item)
        return 0
    if isinstance(value, bool):
        print("True" if value else "False")
    else:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
