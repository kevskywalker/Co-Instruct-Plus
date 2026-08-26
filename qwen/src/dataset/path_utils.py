"""Path helpers for portable image / annotation resolution."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

# Historical absolute roots seen in lab annotation JSON.
_ABS_PREFIXES: tuple[str, ...] = (
    "/home/zhw/IQA/Co-Instruct-plus/",
    "/home/zhw/IQA/Co-Instruct-Plus/",
    "/mnt/nvme/zhw/Co-Instruct-plus/",
    "/mnt/nvme/zhw/Co-Instruct-Plus/",
    "/home/yuhan/IQA/Co-Instruct-plus/",
    "/home/yuhan/IQA/Co-Instruct-Plus/",
    "/home/yuhan/Data-DeQA-Score/",
)


def _strip_known_prefix(path: str) -> str:
    for prefix in _ABS_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _normalize_rel(path: str) -> str:
    while path.startswith("./"):
        path = path[2:]
    return path


def resolve_annotation_json(json_path: str, yaml_path: Optional[str] = None) -> str:
    """Resolve a dataset JSON path from a YAML entry."""
    if os.path.isabs(json_path) and os.path.isfile(json_path):
        return json_path

    candidates: List[str] = []
    if yaml_path:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), json_path))

    ann = os.environ.get("ANNOTATIONS_DIR")
    if ann:
        candidates.append(os.path.join(ann, json_path))
        candidates.append(os.path.join(ann, os.path.basename(json_path)))

    # Common layout: configs/qsit_multi.yaml -> ../training_jsons/<file>
    if yaml_path:
        pkg = os.path.abspath(os.path.join(os.path.dirname(yaml_path), ".."))
        candidates.append(os.path.join(pkg, "training_jsons", os.path.basename(json_path)))
        candidates.append(os.path.join(pkg, "data", "annotations", os.path.basename(json_path)))

    candidates.append(json_path)
    candidates.append(os.path.basename(json_path))

    for cand in candidates:
        cand = os.path.normpath(cand)
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        f"Cannot resolve annotation JSON {json_path!r}. "
        f"Set ANNOTATIONS_DIR or place files next to the YAML. Tried: {candidates[:5]}"
    )


def resolve_media_path(
    path: str,
    media_root: Optional[str] = None,
    extra_roots: Optional[Iterable[str]] = None,
) -> str:
    """Map absolute lab paths / relative ./data paths onto media_root."""
    if not path:
        return path
    if path.startswith("http://") or path.startswith("https://"):
        return path

    original = path
    stripped = _normalize_rel(_strip_known_prefix(path))

    candidates: List[str] = []
    if os.path.isabs(original):
        candidates.append(original)

    roots: List[str] = []
    if media_root:
        roots.append(media_root)
    if extra_roots:
        roots.extend([r for r in extra_roots if r])
    env_koniq = os.environ.get("KONIQ_IMAGES_DIR")
    if env_koniq:
        roots.append(env_koniq)

    for root in roots:
        candidates.append(os.path.join(root, stripped))
        if "/data/" in stripped:
            # already includes data/
            pass
        elif stripped.startswith("data/"):
            pass
        else:
            candidates.append(os.path.join(root, "data", os.path.basename(stripped)))
        candidates.append(os.path.join(root, os.path.basename(stripped)))

    if not os.path.isabs(stripped):
        candidates.append(stripped)

    # Remap abs paths that contain /data/ even if prefix unknown
    if os.path.isabs(original) and "/data/" in original:
        rel = original[original.index("data/") :]
        if media_root:
            candidates.append(os.path.join(media_root, rel))

    seen = set()
    for cand in candidates:
        cand = os.path.normpath(cand)
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            return cand

    # Prefer joined relative path for clearer FileNotFoundError
    if media_root and not os.path.isabs(stripped):
        return os.path.normpath(os.path.join(media_root, stripped))
    return os.path.normpath(original)
