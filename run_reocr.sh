#!/bin/bash
# Full corpus re-OCR with box caching + corrected USCS heuristic. Resumable (ocr_parallel skips
# done/pending/failed via the manifest); safe to kill and restart.
cd /var/home/alp/soilbot
PY=.venv/bin/python
PROG=logs/reocr_progress.out
echo "$(date -u) === reocr start ===" >> "$PROG"
while true; do
  "$PY" scripts/ocr_parallel.py --workers 2 --limit 4000 >> logs/reocr_chunk.out 2>&1
  sleep 5
  P=$("$PY" scripts/db_count.py parse)
  TOT=$(ls data/logs/*.pdf 2>/dev/null | wc -l)
  CACHE=$(ls data/ocr_cache/*.json 2>/dev/null | wc -l)
  echo "$(date -u +%Y-%m-%dT%H:%M) parsed=${P}/${TOT} cached=${CACHE}" >> "$PROG"
  if [ "${P:-0}" -ge "${TOT:-0}" ] && [ "${TOT:-0}" -ge 49000 ]; then
    echo "$(date -u) === ALL RE-OCR'd ===" >> "$PROG"
    break
  fi
done
"$PY" scripts/classify_audit.py >> "$PROG" 2>&1
echo "$(date -u) === audit refreshed ===" >> "$PROG"
