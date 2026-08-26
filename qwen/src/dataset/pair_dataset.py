"""
Pair dataset for DeQA-Score training with Qwen3-VL.

Ports PairDataset, PairDatasetPrebuilt, and DataCollatorForPairDataset from
src/datasets/pair_dataset.py, replacing LLaMA-specific image processing and
tokenization with Qwen's processor-based pipeline.
"""

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from constants import IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .data_utils import (
    get_image_info,
    get_qwen_multimodal_settings,
    llava_to_openai,
    pad_sequence,
    get_mm_token_type_ids,
)
from .path_utils import resolve_media_path

LLAVA_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN_SRC = "<image>"  # token used in conversation strings; llava_to_openai converts this to Qwen format

QUESTION_TEMPLATES = [
    "What do you think about the quality of Image 1 and Image 2?",
    "Can you rate the quality of Image 1 and Image 2?",
    "Can you judge the quality of Image 1 and Image 2?",
    "How would you rate the quality of Image 1 and Image 2?",
    "How would you judge the quality of Image 1 and Image 2?",
    "What is your quality rating for Image 1 and Image 2?",
    "What's your opinion on the quality of Image 1 and Image 2?",
    "Rate the quality of Image 1 and Image 2.",
    "Could you evaluate the quality of Image 1 and Image 2?",
    "How do you assess the quality of Image 1 and Image 2?",
]


def rank0_print(*args):
    import torch.distributed as dist
    try:
        if dist.get_rank() == 0:
            print(*args)
    except Exception:
        print(*args)


def _process_conversations_qwen(
    conversations: List[Dict],
    images: List,  # pre-loaded image tensors from get_image_info
    processor,
    image_min_pixel: int,
    image_max_pixel: int,
    image_patch_size: int,
    image_folder: str,
) -> Dict[str, torch.Tensor]:
    """
    Tokenise a single (human, gpt) conversation turn with Qwen's chat template.

    Returns dict with input_ids, labels, attention_mask, pixel_values,
    image_grid_thw, mm_token_type_ids.
    """
    # Convert LLaVA-style {"from": "human"/"gpt", "value": ...} to OpenAI format.
    # llava_to_openai converts "<image>" → "<|vision_start|><|image_pad|><|vision_end|>".
    openai_msgs = llava_to_openai(conversations, is_video=False)

    # Build the chat-formatted prompt/response strings turn by turn
    all_input_ids = []
    all_labels = []
    all_mm_token_type_ids = []
    all_pixel_values = []
    all_image_grid_thw = []

    from constants import DEFAULT_IMAGE_TOKEN  # Qwen's <|image_pad|> form

    image_curr_count = 0

    for j in range(0, len(openai_msgs), 2):
        user_msg = openai_msgs[j]
        gpt_msg = openai_msgs[j + 1]

        user_text = (
            f"{DEFAULT_IM_START_TOKEN}{user_msg['role']}\n"
            f"{user_msg['content']}{DEFAULT_IM_END_TOKEN}\n"
            f"{DEFAULT_IM_START_TOKEN}{gpt_msg['role']}\n"
        )
        gpt_text = f"{gpt_msg['content']}{DEFAULT_IM_END_TOKEN}\n"

        if DEFAULT_IMAGE_TOKEN in user_text:
            num_images = user_text.count(DEFAULT_IMAGE_TOKEN)
            images_for_turn = images[image_curr_count: image_curr_count + num_images]
            inputs = processor(
                text=[user_text],
                images=images_for_turn,
                padding=False,
                do_resize=False,
                return_tensors="pt",
            )
            prompt_input_ids = inputs["input_ids"]
            prompt_mm_ids = get_mm_token_type_ids(inputs, prompt_input_ids)
            all_pixel_values.append(inputs["pixel_values"])
            all_image_grid_thw.append(inputs["image_grid_thw"])
            image_curr_count += num_images
        else:
            prompt_input_ids = processor.tokenizer(
                user_text,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt",
            )["input_ids"]
            prompt_mm_ids = torch.zeros_like(prompt_input_ids, dtype=torch.long)

        response_input_ids = processor.tokenizer(
            gpt_text,
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        )["input_ids"]
        response_mm_ids = torch.zeros_like(response_input_ids, dtype=torch.long)

        input_ids = torch.cat(
            [prompt_input_ids, response_input_ids], dim=1
        ).squeeze(0)
        mm_token_type_ids = torch.cat(
            [prompt_mm_ids, response_mm_ids], dim=1
        ).squeeze(0)
        labels = torch.cat(
            [
                torch.full(
                    (prompt_input_ids.shape[1],), IGNORE_INDEX, dtype=torch.long
                ),
                response_input_ids.squeeze(0),
            ],
            dim=0,
        )

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_mm_token_type_ids.append(mm_token_type_ids)

    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
    labels = torch.cat(all_labels, dim=0).to(torch.long)
    mm_token_type_ids = torch.cat(all_mm_token_type_ids, dim=0).to(torch.long)
    attention_mask = (input_ids > -1_000_000).to(torch.long)

    result: Dict[str, torch.Tensor] = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
    )
    if all_pixel_values:
        result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        result["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
    return result


def _load_images_qwen(
    image_files: List[str],
    image_folder: str,
    processor,
    image_min_pixel: int,
    image_max_pixel: int,
    image_patch_size: int,
) -> List:
    """Load image files and preprocess via Qwen's get_image_info."""
    images = []
    for imfile in image_files:
        full_path = resolve_media_path(
            imfile,
            image_folder,
            extra_roots=[os.environ.get("KONIQ_IMAGES_DIR")],
        )
        image_input = get_image_info(
            full_path,
            image_min_pixel,
            image_max_pixel,
            None,
            None,
            image_patch_size,
        )
        images.append(image_input)
    return images


class PairDataset(Dataset):
    """Dynamic online pairing dataset for DeQA training with Qwen3-VL."""

    def __init__(
        self,
        data_paths: List[str],
        data_weights: List[int],
        processor,
        data_args,
        model_id: str,
    ):
        super().__init__()
        dataset_list = []
        for data_path, data_weight in zip(data_paths, data_weights):
            data_list = json.load(open(data_path, "r"))
            dataset_list.append(data_list * data_weight)
        self.dataset_list = dataset_list

        nums_eachdata = [len(d) for d in dataset_list]
        nums_predata = list(nums_eachdata)
        for i in range(1, len(nums_predata)):
            nums_predata[i] += nums_predata[i - 1]

        rank0_print("Formatting inputs... Skip in lazy mode")
        self.processor = processor
        self.data_args = data_args
        self.nums_eachdata = nums_eachdata
        self.nums_predata = nums_predata
        self.score_bounds = self._compute_score_bounds()

        self.model_type, self.image_patch_size, _ = get_qwen_multimodal_settings(model_id)
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels

    def __len__(self):
        return self.nums_predata[-1]

    @property
    def lengths(self):
        length_list = []
        for dataset in self.dataset_list:
            for sample in dataset:
                img_tokens = 128 if "image" in sample else 0
                length_list.append(
                    sum(
                        len(conv["value"].split())
                        for conv in sample["conversations"]
                    )
                    + img_tokens
                )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for dataset in self.dataset_list:
            for sample in dataset:
                cur_len = sum(
                    len(conv["value"].split())
                    for conv in sample["conversations"]
                )
                cur_len = cur_len if "image" in sample else -cur_len
                length_list.append(cur_len)
        return length_list

    def next_rand(self):
        return random.randint(0, len(self) - 1)

    @staticmethod
    def get_level(mos: float, min_mos: float, max_mos: float) -> str:
        eps = 1e-8
        texts = ["Bad", "Poor", "Fair", "Good", "Excellent"]
        if not (isinstance(mos, (int, float)) and math.isfinite(mos)):
            return "fair"
        if not (
            isinstance(min_mos, (int, float)) and isinstance(max_mos, (int, float))
        ):
            return "fair"
        if max_mos - min_mos <= 0:
            return "fair"
        level = len(texts) // 2
        for idx in range(1, len(texts) + 1):
            mos_left = min_mos + (idx - 1) / 5 * (max_mos - min_mos) - eps
            mos_right = min_mos + idx / 5 * (max_mos - min_mos) + eps
            if mos > mos_left and mos <= mos_right:
                level = idx
                break
        return texts[level - 1]

    def _get_level_label(self, sample: Dict, idx_dataset: int) -> str:
        score = sample.get("gt_score", float("nan"))
        min_mos, max_mos = self.score_bounds[idx_dataset]
        return self.get_level(score, min_mos, max_mos)

    def _compute_score_bounds(self):
        bounds = []
        for dataset in self.dataset_list:
            scores = [
                s.get("gt_score")
                for s in dataset
                if isinstance(s.get("gt_score"), (int, float))
                and math.isfinite(s.get("gt_score"))
            ]
            bounds.append((min(scores), max(scores)) if scores else (0.0, 5.0))
        return bounds

    @staticmethod
    def _get_comparison_label(
        score_a: float, std_a: float, score_b: float, std_b: float
    ) -> str:
        q_ij = score_a - score_b
        sigma_ij = math.sqrt(max(std_a, 0.0) ** 2 + max(std_b, 0.0) ** 2)
        if q_ij > 2 * sigma_ij:
            return "superior"
        if q_ij > sigma_ij:
            return "better"
        if q_ij > -sigma_ij:
            return "similar"
        if q_ij > -2 * sigma_ij:
            return "worse"
        return "inferior"

    @staticmethod
    def _is_valid_metric(value) -> bool:
        return (
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value > -10000
        )

    def __getitem__(self, i: int) -> Dict:
        target_labels = ["inferior", "worse", "similar", "better", "superior"]
        target_comparison = target_labels[i % len(target_labels)]

        while True:
            try:
                if i < self.nums_predata[0]:
                    idx_dataset = 0
                    idx_sample = i
                else:
                    for idx_dataset in range(1, len(self.nums_predata)):
                        if (
                            i < self.nums_predata[idx_dataset]
                            and i >= self.nums_predata[idx_dataset - 1]
                        ):
                            idx_sample = i - self.nums_predata[idx_dataset - 1]
                            break

                sample_a = self.dataset_list[idx_dataset][idx_sample]
                if "image" not in sample_a:
                    i = self.next_rand()
                    continue

                score_a = sample_a.get("gt_score", -10000)
                std_a = sample_a.get("std", -10000)
                if not self._is_valid_metric(score_a) or not self._is_valid_metric(
                    std_a
                ):
                    i = self.next_rand()
                    continue

                if self.nums_eachdata[idx_dataset] < 2:
                    i = self.next_rand()
                    continue

                candidate_indices = []
                for idx_sample_b in range(self.nums_eachdata[idx_dataset]):
                    if idx_sample_b == idx_sample:
                        continue
                    temp = self.dataset_list[idx_dataset][idx_sample_b]
                    if "image" not in temp:
                        continue
                    score_b = temp.get("gt_score", -10000)
                    std_b = temp.get("std", -10000)
                    if self._is_valid_metric(score_b) and self._is_valid_metric(std_b):
                        candidate_indices.append(idx_sample_b)

                if not candidate_indices:
                    i = self.next_rand()
                    continue

                found_b = False
                for idx_sample_B in random.sample(
                    candidate_indices, k=min(20, len(candidate_indices))
                ):
                    temp = self.dataset_list[idx_dataset][idx_sample_B]
                    score_b = temp.get("gt_score", -10000)
                    std_b = temp.get("std", -10000)
                    current_comparison = self._get_comparison_label(
                        score_a, std_a, score_b, std_b
                    )
                    if current_comparison == target_comparison:
                        sample_b = temp
                        comparison = current_comparison
                        found_b = True
                        break

                if not found_b:
                    idx_sample_B = random.choice(candidate_indices)
                    sample_b = self.dataset_list[idx_dataset][idx_sample_B]
                    score_b = sample_b.get("gt_score", -10000)
                    std_b = sample_b.get("std", -10000)
                    comparison = self._get_comparison_label(
                        score_a, std_a, score_b, std_b
                    )

                level_a = self._get_level_label(sample_a, idx_dataset)
                level_b = self._get_level_label(sample_b, idx_dataset)

                relation_text = {
                    "superior": "superior to",
                    "better": "better than",
                    "similar": "similar to",
                    "worse": "worse than",
                    "inferior": "inferior to",
                }[comparison]

                question = random.choice(QUESTION_TEMPLATES)

                image_files = [sample_a["image"], sample_b["image"]]
                conversations = [
                    {
                        "from": "human",
                        "value": (
                            f"{question}\n"
                            f"Image 1: {DEFAULT_IMAGE_TOKEN_SRC}\n"
                            f"Image 2: {DEFAULT_IMAGE_TOKEN_SRC}"
                        ),
                    },
                    {
                        "from": "gpt",
                        "value": (
                            f"The quality of Image 1 is {level_a}, "
                            f"the quality of Image 2 is {level_b}, "
                            f"Image 1 is {relation_text} Image 2."
                        ),
                    },
                ]

                images = _load_images_qwen(
                    image_files,
                    self.data_args.image_folder,
                    self.processor,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_patch_size,
                )

                data_dict = _process_conversations_qwen(
                    conversations,
                    images,
                    self.processor,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_patch_size,
                    self.data_args.image_folder,
                )

                data_dict["task_type"] = "score"
                data_dict["gt_scores"] = [score_a, score_b]
                data_dict["stds"] = [std_a, std_b]
                data_dict["level_probs"] = [
                    sample_a.get("level_probs", [-10000] * 5),
                    sample_b.get("level_probs", [-10000] * 5),
                ]
                data_dict["image_file"] = image_files
                return data_dict

            except Exception as ex:
                print(f"Error at index {i}: {ex}")
                i = self.next_rand()
                continue


class PairDatasetPrebuilt(Dataset):
    """Pre-generated fixed image pairs dataset for Qwen3-VL."""

    def __init__(
        self,
        data_paths: List[str],
        processor,
        data_args,
        model_id: str,
    ):
        super().__init__()
        samples = []
        for data_path in data_paths:
            samples.extend(json.load(open(data_path, "r")))
        self.samples = samples
        self.processor = processor
        self.data_args = data_args
        self.dataset_bounds = self._compute_bounds()

        self.model_type, self.image_patch_size, _ = get_qwen_multimodal_settings(model_id)
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels

        rank0_print(
            f"Loaded {len(self.samples)} prebuilt pairs from {len(data_paths)} file(s)"
        )

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        return [1] * len(self.samples)

    @property
    def modality_lengths(self):
        return [1] * len(self.samples)

    def _compute_bounds(self):
        bounds: Dict[str, tuple] = {}
        for s in self.samples:
            ds = s.get("dataset", "default")
            for sc in s.get("gt_scores", []):
                if isinstance(sc, (int, float)) and math.isfinite(sc):
                    lo, hi = bounds.get(ds, (sc, sc))
                    bounds[ds] = (min(lo, sc), max(hi, sc))
        if not bounds:
            bounds["default"] = (0.0, 5.0)
        return bounds

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        image_files = s["image"]
        scores = s.get("gt_scores", [-10000, -10000])
        stds = s.get("stds", [0.0, 0.0])
        level_probs = s.get("level_probs", [[-10000] * 5, [-10000] * 5])
        ds_name = s.get("dataset", "default")
        min_mos, max_mos = self.dataset_bounds.get(ds_name, (0.0, 5.0))

        comparison = s.get("comparison")
        if comparison is None:
            comparison = PairDataset._get_comparison_label(
                scores[0], stds[0], scores[1], stds[1]
            )

        level_a = PairDataset.get_level(scores[0], min_mos, max_mos)
        level_b = PairDataset.get_level(scores[1], min_mos, max_mos)

        relation_text = {
            "superior": "superior to",
            "better": "better than",
            "similar": "similar to",
            "worse": "worse than",
            "inferior": "inferior to",
        }.get(comparison, "similar to")

        question = random.choice(QUESTION_TEMPLATES)

        conversations = [
            {
                "from": "human",
                "value": (
                    f"{question}\n"
                    f"Image 1: {DEFAULT_IMAGE_TOKEN_SRC}\n"
                    f"Image 2: {DEFAULT_IMAGE_TOKEN_SRC}"
                ),
            },
            {
                "from": "gpt",
                "value": (
                    f"The quality of Image 1 is {level_a}, "
                    f"the quality of Image 2 is {level_b}, "
                    f"Image 1 is {relation_text} Image 2."
                ),
            },
        ]

        images = _load_images_qwen(
            image_files,
            self.data_args.image_folder,
            self.processor,
            self.image_min_pixel,
            self.image_max_pixel,
            self.image_patch_size,
        )

        data_dict = _process_conversations_qwen(
            conversations,
            images,
            self.processor,
            self.image_min_pixel,
            self.image_max_pixel,
            self.image_patch_size,
            self.data_args.image_folder,
        )

        data_dict["task_type"] = s.get("task_type", "score")
        data_dict["gt_scores"] = scores
        data_dict["stds"] = stds
        data_dict["level_probs"] = level_probs
        data_dict["image_file"] = image_files
        return data_dict


@dataclass
class DataCollatorForPairDataset:
    """Collate pair-dataset examples for Qwen3-VL training."""

    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        batch = {
            "input_type": "pair",
            "item": self.collate_one(instances),
        }
        return batch

    def collate_one(self, instances: Sequence[Dict]) -> Dict:
        # Pad input_ids / labels
        input_ids = pad_sequence(
            [inst["input_ids"] for inst in instances],
            padding_side="right",
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [inst["labels"] for inst in instances],
            padding_side="right",
            padding_value=IGNORE_INDEX,
        )
        mm_token_type_ids = pad_sequence(
            [inst["mm_token_type_ids"] for inst in instances],
            padding_side="right",
            padding_value=0,
        )
        attention_mask = input_ids != self.pad_token_id

        batch: Dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )

        batch["task_types"] = [inst["task_type"] for inst in instances]
        batch["gt_scores"] = torch.tensor(
            [inst["gt_scores"] for inst in instances], dtype=torch.float32
        )
        batch["stds"] = torch.tensor(
            [inst["stds"] for inst in instances], dtype=torch.float32
        )
        batch["level_probs"] = torch.tensor(
            [inst["level_probs"] for inst in instances], dtype=torch.float32
        )

        # Vision tensors: keep as list-of-tensors so get_subitem can index per sample.
        # forward_single will torch.cat them right before calling the model.
        if "pixel_values" in instances[0]:
            batch["pixel_values"] = [inst["pixel_values"] for inst in instances]
            batch["image_grid_thw"] = [inst["image_grid_thw"] for inst in instances]

        return batch


def make_pair_data_module(
    processor,
    data_args,
    model_id: str,
) -> Dict:
    """Create dataset and collator for DeQA pair fine-tuning."""
    if getattr(data_args, "prebuilt_pairs", False):
        train_dataset = PairDatasetPrebuilt(
            data_paths=data_args.data_paths,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
        )
    else:
        train_dataset = PairDataset(
            data_paths=data_args.data_paths,
            data_weights=data_args.data_weights,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
        )
    data_collator = DataCollatorForPairDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
