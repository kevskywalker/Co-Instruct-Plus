"""
DeQA-Score training entry point for Qwen3-VL.

Combines:
  - Qwen model loading (load_qwen_vl_generation_model)
  - DeQA loss wrapper (DeQAQwenModel)
  - Pair dataset (make_pair_data_module)
  - QwenSFTTrainer

All DeQA algorithms (softkl_loss, rating_loss, forward_pair) are preserved.
Only the model backend changes from mPLUG-Owl2/LLaMA to Qwen3-VL.
"""

import ast
import os
import pathlib
from typing import List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, HfArgumentParser

from model.load_model import get_qwen_vl_generation_backbone, load_qwen_vl_generation_model
from model.modeling_deqa_qwen import DeQAConfig, DeQAQwenModel
from trainer import QwenSFTTrainer
from dataset import make_pair_data_module
from params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


# --------------------------------------------------------------------------
# Helpers from train_sft.py (reused verbatim)
# --------------------------------------------------------------------------

def find_target_linear_names(
    model, num_lora_modules=-1, lora_namespan_exclude=None, verbose=True
):
    if lora_namespan_exclude is None:
        lora_namespan_exclude = []
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    backbone = get_qwen_vl_generation_backbone(model)
    vision_tower = backbone.visual
    vision_tower.to(dtype=compute_dtype, device=device)
    set_requires_grad(backbone.visual.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(backbone.visual.merger.parameters(), not training_args.freeze_merger)
    if hasattr(backbone.visual, "deepstack_merger_list"):
        set_requires_grad(
            backbone.visual.deepstack_merger_list.parameters(),
            not training_args.freeze_merger,
        )


def configure_llm(model, training_args):
    backbone = get_qwen_vl_generation_backbone(model)
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(backbone.language_model.parameters(), not training_args.freeze_llm)


def unfreeze_topk_layers(model, k_llm=0, k_vis=0):
    backbone = get_qwen_vl_generation_backbone(model)
    if k_llm and hasattr(backbone, "language_model") and hasattr(
        backbone.language_model, "layers"
    ):
        for layer in backbone.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True
    if k_vis and hasattr(backbone, "visual") and hasattr(backbone.visual, "blocks"):
        for blk in backbone.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


# --------------------------------------------------------------------------
# Build DeQAConfig from training arguments and tokenizer
# --------------------------------------------------------------------------

def build_deqa_config(training_args, processor, vocab_size: int) -> DeQAConfig:
    """Parse level names / prefix into token IDs and build DeQAConfig."""
    tokenizer = processor.tokenizer

    level_ids = []
    for level_name in training_args.level_names:
        # Tokenize with leading space to match in-context tokenization
        # (e.g., "is Excellent" → [" Excellent"] as a single token)
        ids = tokenizer(" " + level_name, add_special_tokens=False).input_ids
        assert len(ids) == 1, (
            f"Level name ' {level_name}' must tokenize to exactly 1 token, "
            f"got {ids}"
        )
        level_ids.append(ids[0])

    level_prefix_ids = []
    if training_args.level_prefix:
        level_prefix_ids = tokenizer(
            training_args.level_prefix, add_special_tokens=False
        ).input_ids

    return DeQAConfig(
        level_ids=level_ids,
        level_prefix_ids=level_prefix_ids,
        weight_softkl=training_args.weight_softkl,
        weight_desp=training_args.weight_desp,
        weight_rank=training_args.weight_rank,
        weight_next_token=training_args.weight_next_token,
        weight_in_level=training_args.weight_in_level,
        softkl_loss=training_args.softkl_loss,
        continuous_rating_loss=training_args.continuous_rating_loss,
        binary_rating_loss=training_args.binary_rating_loss,
        closeset_rating_loss=training_args.closeset_rating_loss,
        use_fix_std=training_args.use_fix_std,
        detach_pred_std=training_args.detach_pred_std,
        vocab_size=vocab_size,
    )


# --------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------

def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")
    if not training_args.lora_enable:
        assert not training_args.vision_lora, (
            "lora_enable is not enabled but vision_lora is enabled."
        )

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(
            training_args.lora_namespan_exclude
        )
    else:
        training_args.lora_namespan_exclude = []
    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ------------------------------------------------------------------
    # Load base Qwen model
    # ------------------------------------------------------------------
    bnb_kwargs = {}
    if training_args.bits in [4, 8]:
        bnb_kwargs.update(
            dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["visual", "lm_head"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    attn_impl = "sdpa" if training_args.disable_flash_attn2 else "flash_attention_2"
    try:
        config = AutoConfig.from_pretrained(model_args.model_id)
    except ValueError as exc:
        # Trainer checkpoints may not contain a full config.json (with model_type).
        # In that case, point --model_id to the base model and resume from output_dir.
        raise ValueError(
            "Failed to load model config from --model_id. If you passed a checkpoint "
            "directory (e.g., checkpoint-XXXX), use the base model for --model_id and "
            "let training auto-resume from --output_dir checkpoints. "
            f"Got --model_id={model_args.model_id!r}. Original error: {exc}"
        ) from exc
    config._attn_implementation = attn_impl

    base_model = load_qwen_vl_generation_model(
        model_args.model_id,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl,
        **bnb_kwargs,
    )

    # Disable Liger kernel for unsupported types
    if training_args.use_liger_kernel and base_model.config.model_type in {
        "qwen3_5", "qwen3_5_moe"
    }:
        rank0_print(
            f"Disabling Liger kernel for unsupported model_type: "
            f"{base_model.config.model_type}"
        )
        training_args.use_liger_kernel = False
        if hasattr(training_args, "liger_kernel_config"):
            training_args.liger_kernel_config = None

    base_model.config.use_cache = False
    configure_llm(base_model, training_args)
    configure_vision_tower(base_model, training_args, compute_dtype, training_args.device)
    unfreeze_topk_layers(
        base_model,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": not training_args.vision_lora
        }
        base_model.enable_input_require_grads()

    if training_args.bits in [4, 8]:
        base_model.config.dtype = compute_dtype
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        )

    if training_args.lora_enable:
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                base_model,
                lora_namespan_exclude=training_args.lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                base_model.to(torch.bfloat16)
            if training_args.fp16:
                base_model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        base_model = get_peft_model(base_model, peft_config)

        if not training_args.freeze_vision_tower:
            for name, param in base_model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True
        if not training_args.freeze_merger:
            for name, param in base_model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    # ------------------------------------------------------------------
    # Load processor
    # ------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(model_args.model_id)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in base_model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_token" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Build DeQA wrapper
    # ------------------------------------------------------------------
    deqa_cfg = build_deqa_config(
        training_args, processor,
        vocab_size=getattr(base_model.config, "vocab_size", None)
            or base_model.config.text_config.vocab_size
    )
    model = DeQAQwenModel(base_model, deqa_cfg)

    rank0_print(f"Level names: {training_args.level_names}")
    rank0_print(f"Level token IDs: {deqa_cfg.level_ids}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    # Expose DeQA data args on data_args (data_paths, data_weights come from
    # TrainingArguments; copy to data_args for make_pair_data_module)
    data_args.data_paths = training_args.data_paths
    data_args.data_weights = (
        training_args.data_weights
        if training_args.data_weights
        else [1] * len(training_args.data_paths)
    )
    data_args.prebuilt_pairs = training_args.prebuilt_pairs

    data_module = make_pair_data_module(
        processor=processor,
        data_args=data_args,
        model_id=model_args.model_id,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    if training_args.auto_resume:
        checkpoint_dirs = sorted(
            pathlib.Path(training_args.output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
            reverse=True,
        )
        resume_ckpt = None
        for ckpt_dir in checkpoint_dirs:
            if (ckpt_dir / "trainer_state.json").is_file():
                resume_ckpt = str(ckpt_dir)
                break

        if resume_ckpt is not None:
            rank0_print(f"Resuming from checkpoint: {resume_ckpt}")
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            if checkpoint_dirs:
                rank0_print(
                    "Found checkpoint directories but none contains trainer_state.json; "
                    "starting a fresh training run."
                )
            trainer.train()
    else:
        rank0_print("auto_resume=False, starting a fresh training run.")
        trainer.train()
    trainer.save_state()

    base_model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )
        if local_rank == 0 or local_rank == -1:
            base_model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
