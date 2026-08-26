# Model card — Co-Instruct-Plus (Qwen3-VL)

## Model details

- **Base model:** Qwen3-VL-8B-Instruct
- **Stage 1:** Full-parameter SFT on mixed M2C / T2C instruction data (`qsit_multi`)
- **Stage 2:** DeQA-style continuous quality score training. Default: prebuilt image pairs (`pairs_180k`). Optional paper protocol: KonIQ-10K train-split online pairing (`configs/stage2_score_koniq.yaml`).

## Intended use

Research on image quality assessment (IQA) and quality-aware multimodal instruction following. Not intended for safety-critical deployment without additional evaluation.

## Training data

Instruction and pair annotations derived from Co-Instruct-style corpora and public IQA sources. Users must obtain underlying images and respect each dataset’s license (see `docs/DATA.md`).

## Evaluation

Scoring benchmarks via `scripts/eval_iqa.sh` (SRCC / PLCC on KonIQ, SPAQ, LIVE Challenge / CLIVE, KADID, etc., depending on available metas).

## Limitations

- Quality labels are discrete level tokens mapped to continuous scores; domain shift (e.g. PIPAL) can degrade correlation.
- FlashAttention is disabled in default scripts for portability.
- Absolute paths in legacy JSON are remapped at runtime; prefer sanitized relative annotations for release.

## Compute

Multi-GPU DeepSpeed ZeRO-2 (Stage 1) / ZeRO-3 (Stage 2). Exact GPU hours depend on hardware; document your run in experiment logs.

## Citation

See `CITATION.cff` / README BibTeX.
