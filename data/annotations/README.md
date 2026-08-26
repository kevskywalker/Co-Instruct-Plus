# Annotations directory (portable / upload copy)

Stage-1 JSON lists with **relative** image paths (`data/...`). This is the copy to upload to Hugging Face / Zenodo.

| Location | Role |
|----------|------|
| `data/annotations/` | Portable relative paths — **public release** |
| `training_jsons/` | Lab originals (may contain absolute paths) — **keep local, do not upload** |

Regenerate from lab JSON:

```bash
python scripts/sanitize_annotations.py \
  --input-dir training_jsons \
  --output-dir data/annotations \
  --image-root "$IMAGE_FOLDER" \
  --check-exists --sample 200
```

Expected files (names match [`configs/qsit_multi.yaml`](../../configs/qsit_multi.yaml)):

- `m2c_coarse_general_textonly.json`
- `m2c_coarse_mcq_textonly.json`
- `m2c_fine_general_textonly.json`
- `m2c_fine_mcq_textonly.json`
- `t2c_coarse_general.json`
- `t2c_coarse_mcq.json`
- `t2c_fine_general.json`
- `t2c_fine_mcq.json`
- `coinstruct_562k_t2c.json`

Also upload `pairs/pairs_180k.json` (already relative: `koniq10k/...`).

JSON files are gitignored. See [docs/DATA.md](../../docs/DATA.md).
