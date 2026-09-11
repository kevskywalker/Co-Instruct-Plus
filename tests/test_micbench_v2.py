import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen" / "src"))

from evaluate.micbench_v2 import extract_choice, metrics, normalize_rows


def test_extract_choice_handles_letter_and_text():
    candidates = ["first", "second", "third"]
    assert extract_choice("Answer: B", candidates) == "B"
    assert extract_choice("The third", candidates) == "C"
    assert extract_choice("unknown", candidates) is None


def test_normalize_rows_validates_canonical_size(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected 1998"):
        normalize_rows(path, 1998)


def test_metrics_reports_overall_and_image_counts():
    value = metrics([
        {"correct": True, "prediction": "A", "num_images": 3},
        {"correct": False, "prediction": None, "num_images": 4},
    ])
    assert value["samples"] == 2
    assert value["categories"]["Overall"]["accuracy"] == 0.5
    assert value["categories"]["3"]["accuracy"] == 1.0
