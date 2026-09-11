"""Standalone MICBench-v2 multiple-choice evaluator for Qwen3-VL checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CANONICAL_SAMPLES = 1998


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choice_letters(count: int) -> list[str]:
    if not 2 <= count <= 26:
        raise ValueError(f"Expected 2--26 choices, got {count}")
    return [chr(ord("A") + index) for index in range(count)]


def extract_choice(raw: str, candidates: list[str]) -> str | None:
    letters = choice_letters(len(candidates))
    valid = "".join(letters)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S).strip()
    match = re.match(rf"^([{valid}])(?:\s|[\.:：,;\)])", text, flags=re.I)
    if match:
        return match.group(1).upper()
    for pattern in (rf"(?:OPTION|ANSWER|CHOICE)\s*[:：]?\s*([{valid}])\b", rf"\b([{valid}])\b"):
        match = re.search(pattern, text.upper())
        if match:
            return match.group(1)
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    matches = [letter for letter, value in zip(letters, candidates) if value.strip().lower() in normalized]
    return matches[0] if len(matches) == 1 else None


def prompt(question: str, candidates: list[str]) -> str:
    options = "\n".join(f"{letter}. {value}" for letter, value in zip(choice_letters(len(candidates)), candidates))
    return (
        "You are an image quality assessment expert.\n"
        "Answer the multiple-choice question based on the image(s) provided.\n"
        f"Return ONLY one letter: {'/'.join(choice_letters(len(candidates)))}.\n\n"
        f"Question: {question.replace('<image>', '').strip()}\n{options}"
    )


def normalize_rows(path: str | Path, expected_samples: int | None) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("MICBench annotation must be a JSON list")
    if expected_samples is not None and len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} MICBench samples, found {len(rows)}")
    ids = [str(row.get("id", index)) for index, row in enumerate(rows)]
    if len(ids) != len(set(ids)):
        raise ValueError("MICBench annotation has duplicate ids")
    for index, row in enumerate(rows):
        images = row.get("images", row.get("img_path"))
        candidates = row.get("candidates")
        if not isinstance(images, list) or not 3 <= len(images) <= 6:
            raise ValueError(f"Sample {index}: expected 3--6 images")
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 26:
            raise ValueError(f"Sample {index}: invalid candidates")
        gold = str(row.get("correct_choice", "")).upper()
        if gold not in choice_letters(len(candidates)):
            raise ValueError(f"Sample {index}: invalid correct_choice")
    return rows


class Runner:
    def __init__(self, model_path: str, preprocessor_path: str | None, device: str, dtype: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        processor_path = preprocessor_path or model_path
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa"
        ).to(device).eval()

    def generate(self, paths: list[str], question: str, candidates: list[str]) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [{"type": "image", "image": path} for path in paths] + [{"type": "text", "text": prompt(question, candidates)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
        return self.processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in predictions:
        correct = int(row["correct"])
        count = str(row["num_images"])
        for key in ("Overall", count):
            values[key][0] += correct
            values[key][1] += 1
    def entry(value: list[int]) -> dict[str, Any]:
        return {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1] if value[1] else None}
    return {"benchmark": "micbench-v2", "samples": len(predictions), "parsed": sum(bool(item["prediction"]) for item in predictions), "categories": {key: entry(values[key]) for key in ("3", "4", "5", "6", "Overall")}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3-VL checkpoint on MICBench-v2")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="results/micbench-v2")
    parser.add_argument("--preprocessor-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--allow-noncanonical", action="store_true")
    args = parser.parse_args()
    rows = normalize_rows(args.annotation, None if args.allow_noncanonical else CANONICAL_SAMPLES)
    image_root = Path(args.image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    runner = Runner(args.model_path, args.preprocessor_path, args.device, args.dtype, args.max_new_tokens)
    predictions = []
    for index, row in enumerate(rows):
        images = row.get("images", row.get("img_path"))
        paths = [image_root / value for value in images]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        candidates = [str(value) for value in row["candidates"]]
        raw = runner.generate([str(path) for path in paths], str(row["question"]), candidates)
        prediction = extract_choice(raw, candidates)
        gold = str(row["correct_choice"]).upper()
        predictions.append({"id": str(row.get("id", index)), "prediction": prediction, "gold": gold, "correct": prediction == gold, "raw_output": raw, "num_images": len(images)})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    value = metrics(predictions)
    (output / "metrics.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    manifest = {"benchmark": "micbench-v2", "created_unix": time.time(), "annotation": str(Path(args.annotation).resolve()), "annotation_sha256": file_sha256(args.annotation), "model_path": str(Path(args.model_path).resolve()), "samples": len(rows), "max_new_tokens": args.max_new_tokens}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
