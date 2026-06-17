#!/bin/bash
# Chained, resumable full-corpus OCR driver for B1 (download-all -> OCR-all).
# Both stages skip completed work via the DuckDB manifest, so this is safe to re-run.
cd /var/home/alp/soilbot
echo "=== STAGE 1: download all 49k logs ($(date -u)) ==="
.venv/bin/python -m pipeline.run --phase 3 --download-logs 2>&1 | grep -E "download_done|download_progress"
echo "=== STAGE 2: OCR all downloaded logs ($(date -u)) ==="
.venv/bin/python -m pipeline.run --phase 3 --ocr 2>&1 | grep -E "parse_done"
echo "=== FULL-CORPUS OCR DONE ($(date -u)) ==="
.venv/bin/python -c "import duckdb;c=duckdb.connect('data/soilbot.duckdb',read_only=True);print('SPT borings:',c.execute('SELECT count(DISTINCT boring_id) FROM strata WHERE spt_n IS NOT NULL').fetchone()[0]);c.close()"
