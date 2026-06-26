#!/bin/bash
# One-shot finisher: mop up the 3,787 reset-failed logs, then rebuild the 3D dataset and run the
# FINAL B1 training on the complete corpus (the chained loop quit at the 3k checkpoint, short of the
# 4,412 profiles now available). Resumable: ocr_parallel skips done/pending; re-running is safe.
cd /var/home/alp/soilbot
PY=.venv/bin/python
PROG=logs/finish_progress.out
echo "$(date -u) === finish start ===" >> "$PROG"

# Mop-up OCR over the reset failed rows (now manifest-absent -> reprocessed). Loop until the
# unprocessed count stops dropping, so transient failures get one clean retry pass.
while true; do
  "$PY" scripts/ocr_parallel.py --workers 2 --limit 4000 >> logs/finish_ocr.out 2>&1
  sleep 5
  N=$("$PY" scripts/db_count.py spt)
  P=$("$PY" scripts/db_count.py parse)
  TOT=$(ls data/logs/*.pdf 2>/dev/null | wc -l)
  echo "$(date -u +%Y-%m-%dT%H:%M) parsed=${P}/${TOT} spoon_spt_borings=${N}" >> "$PROG"
  if [ "${P:-0}" -ge "${TOT:-0}" ]; then break; fi
done
echo "$(date -u) mop-up OCR complete" >> "$PROG"

# Rebuild the 3D dataset on the complete strata, then the final 5-fold B1.
"$PY" -c "from pipeline.config import Config; from pipeline.logging_setup import new_run_id, setup; from ml.data3d import build_and_cache_3d; cfg=Config.load(None); build_and_cache_3d(cfg, setup(cfg.path('log_dir'),'ml.log',new_run_id(),'ds3d',console=False))" >> logs/finish_ocr.out 2>&1
echo "$(date -u) dataset3d rebuilt" >> "$PROG"
echo "--- FINAL B1 (full corpus) ---" >> "$PROG"
"$PY" -m ml.train3d --folds 5 2>&1 | grep -E "SPT-N:|baseline|USCS@|GW:" >> "$PROG"
echo "$(date -u) === finish done ===" >> "$PROG"
