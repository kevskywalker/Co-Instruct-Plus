# Release checklist

- [ ] `LICENSE` present (Apache-2.0)
- [ ] `CITATION.cff` + README BibTeX filled with paper details
- [ ] `MODEL_CARD.md` updated
- [ ] No `/home/zhw`, `/mnt/nvme`, `/home/yuhan` in public sources (`bash scripts/check_release.sh`)
- [ ] No files >100MB outside gitignored data dirs
- [ ] Annotations hosted (HF/Zenodo): upload **`data/annotations/`** (relative paths) + `pairs/pairs_180k.json` — **not** `training_jsons/`
- [ ] `rg '/home/|/mnt/' data/annotations/*.json` returns no hits
- [ ] Fresh env: install → download → Stage1 smoke → Stage2 smoke → `eval_iqa.sh`
- [ ] KonIQ-only: `bash scripts/gen_soft_label_koniq.sh` produces `train_koniq_7k.json` without overwriting `test_koniq_2k.json`
- [ ] Tag `v0.1.0` after gate passes
