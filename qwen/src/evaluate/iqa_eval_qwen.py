"""
IQA evaluation script for Qwen3-VL-based DeQA-Score model.

Ported from src/evaluate/iqa_eval.py, replacing mPLUG-Owl2 model loading
and image processing with Qwen3-VL's AutoProcessor pipeline.
"""

import argparse
import json
import os
from collections import defaultdict
from io import BytesIO

import requests
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    device = args.device

    # Load model and processor
    processor = AutoProcessor.from_pretrained(
        args.preprocessor_path or args.model_path,
    )
    # Load config from original pretrained model (checkpoint may have incomplete config)
    config_path = args.preprocessor_path or args.model_path
    config = AutoConfig.from_pretrained(config_path)
    config._attn_implementation = "sdpa"

    # First try loading directly; if keys have "base_model." prefix (from
    # DeQAQwenModel wrapper), strip the prefix and load manually.
    import safetensors.torch as st
    import glob as _glob

    ckpt_files = sorted(_glob.glob(os.path.join(args.model_path, "*.safetensors")))
    if ckpt_files:
        raw_sd = {}
        for f in ckpt_files:
            raw_sd.update(st.load_file(f))
        # Check if keys have "base_model." prefix
        needs_strip = any(k.startswith("base_model.") for k in raw_sd)
        if needs_strip:
            raw_sd = {k.replace("base_model.", "", 1): v for k, v in raw_sd.items()}

        # Init model from config, then load stripped weights
        model = Qwen3VLForConditionalGeneration(config)
        model.load_state_dict(raw_sd, strict=True)
        model = model.to(dtype=torch.bfloat16, device=device).eval()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()

    meta_paths = args.meta_paths
    root_dir = args.root_dir
    save_dir = args.save_dir
    spaq_images_dir = args.spaq_images_dir or os.environ.get("SPAQ_IMAGES_DIR")
    koniq_images_dir = args.koniq_images_dir or os.environ.get("KONIQ_IMAGES_DIR")
    os.makedirs(save_dir, exist_ok=True)
    with_prob = args.with_prob

    # Build level token IDs (with leading space to match in-context tokenization)
    toks = args.level_names
    print("Level names:", toks)
    ids_ = []
    for name in toks:
        token_ids = processor.tokenizer(" " + name, add_special_tokens=False).input_ids
        assert len(token_ids) == 1, (
            f"Level name ' {name}' must tokenize to exactly 1 token, got {token_ids}"
        )
        ids_.append(token_ids[0])
    print("Level token IDs:", ids_)

    # Build prompt text (single image quality assessment)
    # Use Qwen chat template format
    prompt_text = (
        "<|im_start|>user\n"
        "How would you rate the quality of this image?\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "The quality of the image is"
    )

    image_min_pixels = getattr(args, "image_min_pixels", 524288)
    image_max_pixels = getattr(args, "image_max_pixels", 1310720)

    for meta_path in meta_paths:
        with open(meta_path) as f:
            iqadata = json.load(f)

        missing_images = 0
        processed_images = 0
        first_missing_image_path = None

        imgs_handled = []
        save_path = os.path.join(save_dir, os.path.basename(meta_path))
        open(save_path, "a").close()
        if os.path.exists(save_path):
            with open(save_path) as fr:
                for line in fr:
                    meta_res = json.loads(line)
                    imgs_handled.append(meta_res["image"])

        # Collect items for batching
        batch_images = []
        batch_data = []
        batch_size = args.batch_size

        meta_name = os.path.basename(meta_path)
        for i, llddata in enumerate(tqdm(iqadata, desc=f"Evaluating [{meta_name}]")):
            try:
                filename = llddata["image"]
            except:
                filename = llddata["img_path"]
            if filename in imgs_handled:
                continue

            candidate_paths = []
            primary_path = os.path.join(root_dir, filename)
            if os.path.exists(primary_path) and os.path.getsize(primary_path) > 0:
                candidate_paths.append(primary_path)
            if spaq_images_dir is not None and filename.startswith("SPAQ/images/"):
                alt = os.path.join(spaq_images_dir, os.path.basename(filename))
                if os.path.exists(alt) and os.path.getsize(alt) > 0:
                    candidate_paths.append(alt)
            if koniq_images_dir is not None and filename.startswith("KONIQ/images/"):
                alt = os.path.join(koniq_images_dir, os.path.basename(filename))
                if os.path.exists(alt) and os.path.getsize(alt) > 0:
                    candidate_paths.append(alt)

            image_path = None
            image = None
            for cand in candidate_paths:
                try:
                    image = load_image(cand)
                    image_path = cand
                    break
                except (UnidentifiedImageError, OSError):
                    continue

            if image_path is None:
                missing_images += 1
                if first_missing_image_path is None:
                    first_missing_image_path = primary_path
                continue

            llddata["logits"] = defaultdict(float)
            llddata["probs"] = defaultdict(float)

            batch_images.append(image)
            batch_data.append(llddata)
            processed_images += 1

            if len(batch_images) >= batch_size or (
                i == len(iqadata) - 1 and len(batch_images) > 0
            ):
                # Process batch: each image gets the same prompt
                texts = [prompt_text] * len(batch_images)
                inputs = processor(
                    text=texts,
                    images=batch_images,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                with torch.inference_mode():
                    outputs = model(**inputs)
                    # Get logits at the last non-pad position for each sample
                    output_logits = outputs.logits  # [B, seq_len, vocab]

                    # For each sample, find the last token position
                    attention_mask = inputs["attention_mask"]
                    # Last valid position per sample
                    seq_lengths = attention_mask.sum(dim=1) - 1  # [B]

                    last_logits = output_logits[
                        torch.arange(output_logits.size(0), device=device),
                        seq_lengths,
                    ]  # [B, vocab]

                if with_prob:
                    output_probs = torch.softmax(last_logits, dim=1)

                for j, xllddata in enumerate(batch_data):
                    for tok, id_ in zip(toks, ids_):
                        xllddata["logits"][tok] += last_logits[j, id_].item()
                        if with_prob:
                            xllddata["probs"][tok] += output_probs[j, id_].item()
                    meta_res = {
                        "id": xllddata["id"],
                        "image": xllddata.get("image", xllddata.get("img_path")),
                        "gt_score": xllddata["gt_score"],
                        "logits": xllddata["logits"],
                    }
                    if with_prob:
                        meta_res["probs"] = xllddata["probs"]
                    with open(save_path, "a") as fw:
                        fw.write(json.dumps(meta_res) + "\n")

                batch_images = []
                batch_data = []

        print(
            f"[{meta_name}] done: processed={processed_images}, missing={missing_images}, "
            f"saved_to={save_path}"
        )
        if first_missing_image_path is not None:
            print(f"[{meta_name}] first_missing_image={first_missing_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--preprocessor-path", type=str, default=None)
    parser.add_argument("--meta-paths", type=str, required=True, nargs="+")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--level-names", type=str, required=True, nargs="+")
    parser.add_argument("--spaq-images-dir", type=str, default=os.environ.get("SPAQ_IMAGES_DIR"))
    parser.add_argument("--koniq-images-dir", type=str, default=os.environ.get("KONIQ_IMAGES_DIR"))
    parser.add_argument("--with-prob", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-min-pixels", type=int, default=524288)
    parser.add_argument("--image-max-pixels", type=int, default=1310720)
    args = parser.parse_args()
    main(args)
