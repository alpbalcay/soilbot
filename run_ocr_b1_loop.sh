#!/bin/bash
# Chained, resumable full-corpus OCR with auto-B1 re-runs.
# Stage 1: finish the download (resumable). Stage 2: OCR in 30-min time-bounded bursts so the
# DuckDB writer lock is released between bursts; in each gap, re-build the 3D dataset + re-run B1
# whenever the spoon-format SPT-N boring count crosses the next +1500 threshold. Fully resumable:
# OCR/download skip completed work via the manifest, so this is safe to kill and restart.
cd /var/home/alp/soilbot
PY=.venv/bin/python
PROG=logs/b1_progress.out
echo "$(date -u) === loop start ===" >> "$PROG"

# Stage 1 — finish download
"$PY" -m pipeline.run --phase 3 --download-logs >> logs/download_full.out 2>&1
echo "$(date -u) download stage complete ($(ls data/logs/*.pdf 2>/dev/null | wc -l) pdfs)" >> "$PROG"

NEXT=1500
while true; do
  # Stage 2 — one PARALLEL OCR batch (2 GPU workers; the GPU fits exactly 2 easyocr readers at
  # ~4GB each). Processes up to 2000 unparsed PDFs then exits cleanly, releasing the DB lock for
  # the B1 checkpoint below. ~10x the single-core rate; clean exit -> no orphan worker processes.
  "$PY" scripts/ocr_parallel.py --workers 2 --limit 2000 >> logs/ocr_chunk.out 2>&1
  sleep 5
  N=$("$PY" scripts/db_count.py spt)
  P=$("$PY" scripts/db_count.py parse)
  TOT=$(ls data/logs/*.pdf 2>/dev/null | wc -l)
  echo "$(date -u +%Y-%m-%dT%H:%M) parsed=${P}/${TOT} spoon_spt_borings=${N} next_b1=${NEXT}" >> "$PROG"

  if [ "${N:-0}" -ge "$NEXT" ]; then
    echo "--- B1 re-run at ${N} spoon borings ($(date -u +%H:%M)) ---" >> "$PROG"
    "$PY" -c "from pipeline.config import Config; from pipeline.logging_setup import new_run_id, setup; from ml.data3d import build_and_cache_3d; cfg=Config.load(None); build_and_cache_3d(cfg, setup(cfg.path('log_dir'),'ml.log',new_run_id(),'ds3d',console=False))" >> logs/ocr_chunk.out 2>&1
    "$PY" -m ml.train3d --folds 5 2>&1 | grep -E "SPT-N:|baseline|USCS@" >> "$PROG"
    NEXT=$((NEXT + 1500))
  fi

  # done when every downloaded pdf is parsed and the full corpus is downloaded
  if [ "${P:-0}" -ge "${TOT:-0}" ] && [ "${TOT:-0}" -ge 49000 ]; then
    echo "$(date -u) === ALL PARSED — loop done ===" >> "$PROG"
    break
  fi
done
