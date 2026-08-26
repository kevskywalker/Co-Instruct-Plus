"""
DeQA-Score wrapper around a Qwen3-VL generation model.

Ports the DeQA-specific loss methods from src/model/modeling_mplug_owl2.py,
adapting them to work with Qwen's forward interface (pixel_values + image_grid_thw
instead of a raw image tensor processed by vision_model + visual_abstractor).
"""

from typing import List, Optional, Tuple, Union
import types

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import CausalLMOutputWithPast

IGNORE_INDEX = -100


class DeQAConfig:
    """Lightweight config container for DeQA-specific parameters."""

    def __init__(
        self,
        level_ids: List[int],
        level_prefix_ids: List[int],
        weight_softkl: float = 1.0,
        weight_desp: float = 1.0,
        weight_rank: float = 1.0,
        weight_next_token: float = 1.0,
        weight_in_level: Optional[float] = None,
        softkl_loss: bool = True,
        continuous_rating_loss: bool = True,
        binary_rating_loss: str = "fidelity",
        closeset_rating_loss: bool = False,
        use_fix_std: bool = True,
        detach_pred_std: bool = False,
        vocab_size: int = 0,
    ):
        self.level_ids = level_ids
        self.level_prefix_ids = level_prefix_ids
        self.weight_softkl = weight_softkl
        self.weight_desp = weight_desp
        self.weight_rank = weight_rank
        self.weight_next_token = weight_next_token
        self.weight_in_level = weight_in_level
        self.softkl_loss = softkl_loss
        self.continuous_rating_loss = continuous_rating_loss
        self.binary_rating_loss = binary_rating_loss
        self.closeset_rating_loss = closeset_rating_loss
        self.use_fix_std = use_fix_std
        self.detach_pred_std = detach_pred_std
        self.vocab_size = vocab_size


def _rank0_print(*args):
    try:
        if dist.get_rank() == 0:
            print(*args)
    except Exception:
        print(*args)


def _extend_list(lst, target_len, min_len):
    """Pad list to target_len by repeating first element (for DDP sync)."""
    if len(lst) == 0:
        return lst
    while len(lst) < target_len:
        lst.append(lst[0])
    return lst[:max(target_len, min_len)]


class DeQAQwenModel(nn.Module):
    """
    Wraps a Qwen3-VL generation model with DeQA-specific loss computation.

    Usage::
        base_model = load_qwen_vl_generation_model(model_id, ...)
        deqa_cfg = DeQAConfig(level_ids=[...], ...)
        model = DeQAQwenModel(base_model, deqa_cfg)
        loss = model(input_type="pair", item=batch)
    """

    def __init__(self, base_model: nn.Module, deqa_config: DeQAConfig):
        super().__init__()
        self.base_model = base_model
        self.deqa_config = deqa_config

    def __getattr__(self, name: str):
        """Delegate unknown attributes to base_model (DeepSpeed compat etc.)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    # ------------------------------------------------------------------
    # Delegate common model attributes to base_model so that Trainer,
    # DeepSpeed, etc. work transparently.
    # ------------------------------------------------------------------
    @property
    def config(self):
        return self.base_model.config

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def parameters(self, recurse=True):
        return self.base_model.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return self.base_model.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        return self.base_model.named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )

    def modules(self):
        return self.base_model.modules()

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.base_model.load_state_dict(state_dict, strict=strict)

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)

    def train(self, mode=True):
        self.base_model.train(mode)
        return self

    def eval(self):
        self.base_model.eval()
        return self

    def enable_input_require_grads(self):
        self.base_model.enable_input_require_grads()

    def ds_external_parameters(self):
        """DeepSpeed Zero3 compatibility: delegate to base_model."""
        if hasattr(self.base_model, 'ds_external_parameters'):
            return self.base_model.ds_external_parameters()
        return iter([])

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    # ------------------------------------------------------------------
    # Level-token helpers (identical to mplug_owl2 version)
    # ------------------------------------------------------------------

    def _get_level_token_positions(
        self, labels: torch.LongTensor, num_positions: int
    ) -> torch.LongTensor:
        """Find positions of quality-level tokens in label sequences.

        Returns tensor of shape [B, num_positions].
        """
        batch_size = labels.shape[0]
        level_ids = torch.tensor(
            self.deqa_config.level_ids, device=labels.device
        )
        idx_levels = []
        for idx_batch in range(batch_size):
            idx_found = torch.where(torch.isin(labels[idx_batch], level_ids))[0]
            assert idx_found.shape[0] >= num_positions, (
                f"Need at least {num_positions} level tokens in each sample, "
                f"found {idx_found.shape[0]}"
            )
            idx_levels.append(idx_found[:num_positions])
        return torch.stack(idx_levels, dim=0)  # [B, num_positions]

    # ------------------------------------------------------------------
    # Loss functions (identical logic to mplug_owl2 version)
    # ------------------------------------------------------------------

    def softkl_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        level_probs: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """KL divergence between predicted logit distribution and soft targets.

        Returns (loss_kl, idx_level_label, idx_level_logit).
        """
        if level_probs.dim() == 2:  # [B, 5]
            level_probs = level_probs.unsqueeze(1)  # [B, 1, 5]
        assert level_probs.dim() == 3

        batch_size = logits.shape[0]
        num_positions = level_probs.shape[1]
        idx_level_label = self._get_level_token_positions(labels, num_positions)

        level_ids_label = labels[
            torch.arange(batch_size, device=labels.device).unsqueeze(1),
            idx_level_label,
        ]
        for level_id in level_ids_label.reshape(-1):
            assert level_id.item() in self.deqa_config.level_ids

        # logits and labels share the same sequence dimension for Qwen
        # (processor builds input_ids with image tokens inline, labels match)
        assert logits.shape[1] == labels.shape[1], (
            f"logits seq len {logits.shape[1]} != labels seq len {labels.shape[1]}"
        )
        idx_level_logit = idx_level_label - 1
        logits_level_ids = logits[
            torch.arange(batch_size, device=logits.device).unsqueeze(1),
            idx_level_logit,
        ].contiguous()  # [B, K, V]

        preds = torch.softmax(logits_level_ids, dim=2)  # [B, K, V]
        target = torch.zeros_like(preds)  # [B, K, V]
        target[:, :, self.deqa_config.level_ids] = level_probs
        target = target.detach()

        pred_log = torch.log(preds.clamp_min(1e-12))
        loss_kl = F.kl_div(pred_log, target, reduction="batchmean")
        return loss_kl, idx_level_label, idx_level_logit

    def rating_loss(
        self,
        pred_scores_A: torch.FloatTensor,
        pred_stds_A: torch.FloatTensor,
        gt_scores_A: torch.FloatTensor,
        gt_stds_A: torch.FloatTensor,
        pred_scores_B: torch.FloatTensor,
        pred_stds_B: torch.FloatTensor,
        gt_scores_B: torch.FloatTensor,
        gt_stds_B: torch.FloatTensor,
    ) -> torch.FloatTensor:
        eps = 1e-8
        if self.deqa_config.use_fix_std:
            pred = 0.5 * (
                1 + torch.erf((pred_scores_A - pred_scores_B) / 2)
            )
        else:
            pred_var = (
                pred_stds_A * pred_stds_A
                + pred_stds_B * pred_stds_B
                + eps
            )
            if self.deqa_config.detach_pred_std:
                pred_var = pred_var.detach()
            pred = 0.5 * (
                1
                + torch.erf(
                    (pred_scores_A - pred_scores_B)
                    / torch.sqrt(2 * pred_var)
                )
            )
        gt_var = gt_stds_A * gt_stds_A + gt_stds_B * gt_stds_B + eps
        gt = 0.5 * (
            1
            + torch.erf(
                (gt_scores_A - gt_scores_B) / torch.sqrt(2 * gt_var)
            )
        ).to(pred.device)
        gt = gt.detach()
        loss = (
            1
            - (pred * gt + eps).sqrt()
            - ((1 - pred) * (1 - gt) + eps).sqrt()
        ).mean()
        return loss

    def binary_rating_loss(
        self,
        pred_scores_A: torch.FloatTensor,
        gt_scores_A: torch.FloatTensor,
        pred_scores_B: torch.FloatTensor,
        gt_scores_B: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pred = 0.5 * (
            1 + torch.erf((pred_scores_A - pred_scores_B) / 2)
        )
        gt = (gt_scores_A > gt_scores_B).to(pred.dtype).to(pred.device)
        gt = gt.detach()
        if self.deqa_config.binary_rating_loss == "bce":
            loss = F.binary_cross_entropy(pred, gt)
        elif self.deqa_config.binary_rating_loss == "fidelity":
            loss_1 = 1 - pred[gt == 1].sqrt()
            loss_2 = 1 - (1 - pred[gt == 0]).sqrt()
            loss = (loss_1.sum() + loss_2.sum()) / pred_scores_A.shape[0]
        else:
            raise NotImplementedError(
                f"Unknown binary_rating_loss: {self.deqa_config.binary_rating_loss}"
            )
        return loss

    # ------------------------------------------------------------------
    # Forward (single sample / text only)
    # ------------------------------------------------------------------

    def forward_single(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        use_softkl_loss: bool = False,
        level_probs: Optional[torch.FloatTensor] = None,
    ) -> CausalLMOutputWithPast:
        """Single forward pass.  Calls base Qwen model then applies DeQA losses."""
        # Build kwargs for Qwen forward
        forward_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # compute CE loss manually so we can interleave softkl
            return_dict=True,
        )
        if pixel_values is not None:
            # pixel_values may be a list-of-tensors (one per sample); concat for Qwen
            if isinstance(pixel_values, (list, tuple)):
                forward_kwargs["pixel_values"] = torch.cat(pixel_values, dim=0)
            else:
                forward_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, (list, tuple)):
                forward_kwargs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
            else:
                forward_kwargs["image_grid_thw"] = image_grid_thw
        if mm_token_type_ids is not None:
            forward_kwargs["token_type_ids"] = mm_token_type_ids

        outputs = self.base_model(**forward_kwargs)
        logits = outputs.logits  # [B, N, V]

        loss = None
        loss_kl = None

        if labels is not None:
            if use_softkl_loss and level_probs is not None:
                loss_kl, idx_level_label, idx_level_logit = self.softkl_loss(
                    logits, labels, level_probs
                )

                # Remove level-token positions before CE loss
                def del_elements(source, idx):
                    if idx.dim() == 1:
                        idx = idx.unsqueeze(1)
                    mask = torch.ones(
                        [*source.shape[:2]], dtype=torch.bool, device=source.device
                    )
                    for idx_1 in range(idx.shape[0]):
                        mask[idx_1, idx[idx_1]] = False
                    remain_len = source.size(1) - idx.size(1)
                    if len(source.shape) == 2:
                        return source[mask].view(source.size(0), remain_len)
                    assert len(source.shape) == 3
                    return source[mask].view(
                        source.size(0), remain_len, source.size(2)
                    )

                labels_ce = del_elements(labels, idx_level_label)
                logits_ce = del_elements(logits, idx_level_logit)
            else:
                labels_ce = labels
                logits_ce = logits

            # Standard causal LM CE loss (shifted)
            shift_logits = logits_ce[..., :-1, :].contiguous()
            shift_labels = labels_ce[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.deqa_config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )

            if loss_kl is not None:
                loss = loss + self.deqa_config.weight_softkl * loss_kl

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # get_score: extract quality scores from a scored batch item
    # ------------------------------------------------------------------

    def get_score(self, item):
        """Run forward_single and extract quality scores for a pair item.

        Returns (scores_A, stds_A, scores_B, stds_B, loss_next_token, loss_in_level).
        """
        outputs = self.forward_single(
            input_ids=item["input_ids"],
            attention_mask=item["attention_mask"],
            labels=item["labels"],
            pixel_values=item.get("pixel_values"),
            image_grid_thw=item.get("image_grid_thw"),
            mm_token_type_ids=item.get("mm_token_type_ids"),
            use_softkl_loss=self.deqa_config.softkl_loss,
            level_probs=item.get("level_probs"),
        )

        logits = outputs.logits  # [B, N, V]
        labels = item["labels"]
        batch_size = logits.shape[0]

        # 2 level tokens per sample (score_A and score_B)
        idx_levels_label = self._get_level_token_positions(labels, num_positions=2)
        idx_level_logit = idx_levels_label - 1  # causal shift: logit[t] predicts label[t+1]

        logits_level_ids = logits[
            torch.arange(batch_size, device=logits.device).unsqueeze(1),
            idx_level_logit,
        ].contiguous()  # [B, 2, V]

        probs_org = torch.softmax(logits_level_ids, dim=2)  # [B, 2, V]
        loss_in_level = (
            1 - probs_org[:, :, self.deqa_config.level_ids].contiguous().sum(dim=2)
        )  # [B, 2]
        bound = torch.tensor(1e-2).to(loss_in_level)
        loss_in_level = torch.max(bound, loss_in_level.mean())

        if self.deqa_config.closeset_rating_loss:
            logits_levels = logits_level_ids[
                :, :, self.deqa_config.level_ids
            ].contiguous()
            probs = torch.softmax(logits_levels, dim=2)
        else:
            probs = probs_org[:, :, self.deqa_config.level_ids].contiguous()

        weights = torch.tensor([5, 4, 3, 2, 1], dtype=probs.dtype, device=probs.device)
        scores = torch.matmul(probs, weights)  # [B, 2]

        variances = (weights.unsqueeze(0).unsqueeze(0) - scores.unsqueeze(2)) ** 2
        stds = torch.sqrt(torch.sum(probs * variances, dim=2))  # [B, 2]

        return (
            scores[:, 0],
            stds[:, 0],
            scores[:, 1],
            stds[:, 1],
            outputs.loss,
            loss_in_level,
        )

    # ------------------------------------------------------------------
    # get_subitem: split a batch by task_type, syncing across ranks
    # ------------------------------------------------------------------

    def get_subitem(self, item, task_type):
        for key in list(item.keys()):
            if item[key] is None:
                del item[key]

        subitem = {key: [] for key in item}
        for idx in range(len(item["task_types"])):
            if item["task_types"][idx] == task_type:
                for key in item:
                    subitem[key].append(item[key][idx])

        batch_size = torch.tensor(len(subitem["task_types"])).cuda()
        world_size = dist.get_world_size()
        batch_size_allrank = [torch.tensor(0).cuda() for _ in range(world_size)]
        dist.barrier()
        dist.all_gather(batch_size_allrank, batch_size)
        batch_size_max = torch.stack(batch_size_allrank, dim=0).max().item()
        batch_size_min = torch.stack(batch_size_allrank, dim=0).min().item()

        for key in item:
            subitem[key] = _extend_list(subitem[key], batch_size_max, batch_size_min)
            if torch.is_tensor(item[key]) and len(subitem[key]):
                subitem[key] = torch.stack(subitem[key], dim=0)
            # pixel_values / image_grid_thw are kept as list-of-tensors; leave as list
        return subitem

    # ------------------------------------------------------------------
    # forward_pair: main DeQA pair-dataset forward
    # ------------------------------------------------------------------

    def forward_pair(self, item, **kwargs) -> CausalLMOutputWithPast:
        item_desp = self.get_subitem(item, task_type="description")
        item_score = self.get_subitem(item, task_type="score")

        # Description loss
        loss_desp = 0
        if len(item_desp["task_types"]) > 0:
            outputs = self.forward_single(
                input_ids=item_desp["input_ids"],
                attention_mask=item_desp["attention_mask"],
                labels=item_desp["labels"],
                pixel_values=item_desp.get("pixel_values"),
                image_grid_thw=item_desp.get("image_grid_thw"),
                mm_token_type_ids=item_desp.get("mm_token_type_ids"),
                use_softkl_loss=False,
            )
            loss_desp = outputs.loss

        # Score / ranking loss
        loss_score = 0
        if len(item_score["task_types"]) > 0:
            gt_scores_A = item_score["gt_scores"][:, 0]
            gt_scores_B = item_score["gt_scores"][:, 1]
            (
                pred_scores_A,
                pred_stds_A,
                pred_scores_B,
                pred_stds_B,
                loss_next_token,
                loss_in_level,
            ) = self.get_score(item_score)

            if not self.deqa_config.continuous_rating_loss:
                loss_rank = self.binary_rating_loss(
                    pred_scores_A, gt_scores_A, pred_scores_B, gt_scores_B
                )
            else:
                gt_stds_A = item_score["stds"][:, 0]
                gt_stds_B = item_score["stds"][:, 1]
                assert (gt_stds_A >= 0).all() and (gt_stds_B >= 0).all()
                loss_rank = self.rating_loss(
                    pred_scores_A,
                    pred_stds_A,
                    gt_scores_A,
                    gt_stds_A,
                    pred_scores_B,
                    pred_stds_B,
                    gt_scores_B,
                    gt_stds_B,
                )

            _rank0_print(
                f"[score loss (w/o weight) | "
                f"ranking loss: {round(loss_rank.item(), 6)}, "
                f"next token loss: {round(loss_next_token.item(), 6)}, "
                f"in level loss: {round(loss_in_level.item(), 6)}]"
            )

            loss_rank = self.deqa_config.weight_rank * loss_rank
            loss_next_token = (
                self.deqa_config.weight_next_token * loss_next_token
                if self.deqa_config.weight_next_token
                else 0
            )
            loss_in_level = (
                self.deqa_config.weight_in_level * loss_in_level
                if self.deqa_config.weight_in_level
                else 0
            )
            loss_score = loss_rank + loss_next_token + loss_in_level

        loss_desp_item = loss_desp if isinstance(loss_desp, int) else loss_desp.item()
        loss_score_item = loss_score if isinstance(loss_score, int) else loss_score.item()
        _rank0_print(
            f"[loss (w/o weight) | "
            f"description loss: {round(loss_desp_item, 6)}, "
            f"score loss: {round(loss_score_item, 6)}]"
        )

        loss = self.deqa_config.weight_desp * loss_desp + loss_score
        return CausalLMOutputWithPast(loss=loss)

    # ------------------------------------------------------------------
    # Top-level forward dispatcher
    # ------------------------------------------------------------------

    def forward(self, input_type=None, **kwargs) -> CausalLMOutputWithPast:
        if input_type is None or input_type == "single":
            return self.forward_single(**kwargs)
        elif input_type == "pair":
            return self.forward_pair(**kwargs)
        else:
            raise ValueError(f"Unknown input_type: {input_type!r}")
