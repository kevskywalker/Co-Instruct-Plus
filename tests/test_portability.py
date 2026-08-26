import subprocess
import sys
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sanitize_annotations import contains_machine_path, to_relative  # noqa: E402

path_utils_spec = importlib.util.spec_from_file_location(
    "path_utils", ROOT / "qwen" / "src" / "dataset" / "path_utils.py"
)
path_utils = importlib.util.module_from_spec(path_utils_spec)
path_utils_spec.loader.exec_module(path_utils)
resolve_media_path = path_utils.resolve_media_path


def test_sanitize_removes_legacy_deqa_root():
    assert to_relative("/home/yuhan/Data-DeQA-Score/data/example.jpg") == "data/example.jpg"


def test_sanitize_rejects_unmapped_machine_paths():
    assert contains_machine_path({"image": "/home/unknown/data/example.jpg"})
    assert not contains_machine_path({"image": "data/example.jpg"})


def test_runtime_path_resolution_uses_media_root(tmp_path):
    image = tmp_path / "data" / "example.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    assert resolve_media_path("data/example.jpg", str(tmp_path)) == str(image)


def test_yaml_value_reads_scalar_and_list():
    scalar = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "yaml_value.py"),
            str(ROOT / "configs" / "stage1_sft.yaml"),
            "global_batch_size",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert scalar.stdout.strip() == "128"

    labels = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "yaml_value.py"),
            str(ROOT / "configs" / "stage2_score.yaml"),
            "level_names",
            "--list",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert labels.stdout.splitlines() == ["Excellent", "Good", "Fair", "Poor", "Bad"]

    prebuilt = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "yaml_value.py"),
            str(ROOT / "configs" / "stage2_score_koniq.yaml"),
            "prebuilt_pairs",
            "true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert prebuilt.stdout.strip() == "False"
