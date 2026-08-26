import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_soft_label import (  # noqa: E402
    adjust_gaussian_bar,
    expand_cfg_paths,
    format_answer,
    get_binary_probs,
    get_level,
    run_config,
)


def test_binary_probs_sum_to_one_and_two_bins():
    for mos in (1.0, 1.7, 2.0, 3.0, 3.4, 4.0, 4.6, 5.0):
        probs = get_binary_probs(mos)
        assert len(probs) == 5
        assert abs(sum(probs) - 1.0) < 1e-6
        assert sum(p == 0 for p in probs) == 3
        expected = np.inner(np.array(probs), np.array([5, 4, 3, 2, 1]))
        assert abs(expected - mos) < 1e-6


def test_binary_probs_order_excellent_to_bad():
    # MOS=5 sits on excellent (index 0); MOS=1 on bad (index 4).
    # Endpoint bins are 1-eps because get_binary_probs expands edges by 1e-8.
    hi = get_binary_probs(5.0)
    lo = get_binary_probs(1.0)
    assert hi[0] == pytest.approx(1.0, abs=1e-7)
    assert hi[1:] == pytest.approx([0, 0, 0, 0], abs=1e-7)
    assert lo[-1] == pytest.approx(1.0, abs=1e-7)
    assert lo[:-1] == pytest.approx([0, 0, 0, 0], abs=1e-7)
    assert get_binary_probs(3.0) == pytest.approx([0, 0, 1.0, 0, 0], abs=1e-7)


def test_adjust_gaussian_bar_recovers_score():
    # Raw Gaussian pdf mass is not 1; that is the regime DeQA solves for.
    probs = [0.40069179, 0.50447227, 0.06229573, 0.00075452, 8.96e-7]
    score = 4.400806947958697
    alpha, beta = adjust_gaussian_bar(probs, score)
    adjusted = [p * alpha + beta for p in probs]
    rec = float(np.inner(np.array(adjusted), np.array([5, 4, 3, 2, 1])))
    mass = float(sum(adjusted))
    assert abs(mass - 1.0) < 1e-6
    assert abs(rec - score) < 1e-6


def test_get_level_quintiles():
    assert get_level(1.0, 1.0, 5.0) == "bad"
    assert get_level(5.0, 1.0, 5.0) == "excellent"
    assert get_level(3.0, 1.0, 5.0) == "fair"


def test_format_answer_single_and_triple_slots():
    assert (
        format_answer("The quality of the image is {}.", "good")
        == "The quality of the image is good."
    )
    pair = (
        "The quality of the first image is {}, the quality of the second image "
        "is {}, the first image is {} than the second image."
    )
    assert format_answer(pair, "good") == (
        "The quality of the first image is good, the quality of the second image "
        "is good, the first image is good than the second image."
    )


def test_expand_cfg_paths_replaces_root_data():
    cfg = expand_cfg_paths(
        {
            "split_json": "${ROOT_DATA}/KONIQ/metas/split.json",
            "mos_json": "${ROOT_DATA}/KONIQ/metas/mos.json",
            "save_train": "${ROOT_DATA}/koniq10k/metas/train_koniq_7k.json",
            "save_test": "${ROOT_DATA}/KONIQ/metas/test_koniq_2k.generated.json",
            "img_dir": "koniq10k/512x384",
        },
        "/data/root",
    )
    assert cfg["mos_json"] == "/data/root/KONIQ/metas/mos.json"
    assert cfg["img_dir"] == "koniq10k/512x384"


def test_run_config_writes_train_meta(tmp_path):
    root = tmp_path / "root"
    (root / "KONIQ" / "metas").mkdir(parents=True)
    mos = {
        "a.jpg": {"mos": "1.0", "std": "0.5"},
        "b.jpg": {"mos": "5.0", "std": "0.5"},
        "c.jpg": {"mos": "3.0", "std": "0.05"},
        "d.jpg": {"mos": "4.0", "std": "0.5"},
    }
    split = {"train": ["a.jpg", "b.jpg", "c.jpg"], "test": ["d.jpg"]}
    (root / "KONIQ" / "metas" / "mos.json").write_text(json.dumps(mos))
    (root / "KONIQ" / "metas" / "split.json").write_text(json.dumps(split))

    cfg = {
        "answer": "The quality of the image is {}.",
        "dataset_params": {
            "koniq": {
                "split_json": "${ROOT_DATA}/KONIQ/metas/split.json",
                "mos_json": "${ROOT_DATA}/KONIQ/metas/mos.json",
                "save_train": "${ROOT_DATA}/koniq10k/metas/train_koniq_7k.json",
                "save_test": "${ROOT_DATA}/KONIQ/metas/test_koniq_2k.generated.json",
                "img_dir": "koniq10k/512x384",
                "density_type": "pdf",
                "thre_std": 0.2,
                "thre_diff": 0.1,
            }
        },
    }
    run_config(cfg, root_data=str(root))

    train_path = root / "koniq10k" / "metas" / "train_koniq_7k.json"
    test_path = root / "KONIQ" / "metas" / "test_koniq_2k.generated.json"
    train = json.loads(train_path.read_text())
    test = json.loads(test_path.read_text())
    assert {row["image"] for row in train} == {
        "koniq10k/512x384/a.jpg",
        "koniq10k/512x384/b.jpg",
        "koniq10k/512x384/c.jpg",
    }
    assert len(test) == 1
    assert test[0]["image"] == "koniq10k/512x384/d.jpg"
    assert "level_probs" not in test[0]
    for row in train:
        assert "gt_score" in row and "std" in row
        assert len(row["level_probs"]) == 5
        assert all(p >= 0 for p in row["level_probs"])
        # Clipping negatives can leave mass slightly off 1; that matches DeQA.
        assert 0.9 <= sum(row["level_probs"]) <= 1.1
        assert row["conversations"][1]["value"].startswith("The quality of the image is")

    # Tiny std on c.jpg must take the binary fallback.
    tiny = next(row for row in train if row["image"].endswith("c.jpg"))
    assert sum(p == 0 for p in tiny["level_probs"]) == 3
