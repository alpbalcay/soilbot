#!/bin/bash
# auto_phase6_b2.sh — wait for run_ocr_b1_loop.sh to finish the full-corpus OCR, then run
# phase 6 (strata_derived) + rebuild dataset3d (physics) + retrain B1 and B2 + regenerate
# ML_REPORT, exactly once. Detached watcher: survives the Claude session; idempotent (no-ops
# once it has completed). Poll-based so it never holds the DuckDB lock or the GPU while waiting.
#
# Launch:  setsid bash scripts/auto_phase6_b2.sh >/dev/null 2>&1 &
# Cancel:  rm -f logs/.auto_phase6_b2.pid && pkill -f auto_phase6_b2.sh
set -u
cd /var/home/alp/soilbot || exit 1
PY=.venv/bin/python
LOG=logs/auto_phase6_b2.out
DONE=logs/.auto_phase6_b2.done
PIDFILE=logs/.auto_phase6_b2.pid
PROG=logs/b1_progress.out
INTERVAL=300   # poll every 5 min

mkdir -p logs

# one watcher only
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT   # always clear the PID file, even if killed mid-sleep
echo "$(date -u) watcher started (pid $$); polling every ${INTERVAL}s for loop completion" >> "$LOG"

# Complete == the loop logged its done-sentinel AND nothing is still holding the DB/GPU.
# (The DONE-marker short-circuit lives in the main loop below.)
complete() {
  grep -q "ALL PARSED" "$PROG" 2>/dev/null || return 1
  pgrep -f "ocr_parallel.py"     >/dev/null 2>&1 && return 1
  pgrep -f "run_ocr_b1_loop.sh"  >/dev/null 2>&1 && return 1
  return 0
}

# Run one pipeline step, logging start/failure so a late-stage break is diagnosable.
step() {
  echo "$(date -u) step: $1" >> "$LOG"; shift
  "$@" >> "$LOG" 2>&1
}

while true; do
  if [ -f "$DONE" ]; then
    echo "$(date -u) already completed; watcher exiting" >> "$LOG"; break
  fi
  if complete; then
    echo "$(date -u) === OCR loop complete -> phase 6 + B1/B2 retrain ===" >> "$LOG"
    BUILD3D="from pipeline.config import Config; from pipeline.logging_setup import new_run_id, setup; from ml.data3d import build_and_cache_3d; cfg=Config.load(None); build_and_cache_3d(cfg, setup(cfg.path('log_dir'),'ml.log',new_run_id(),'ds3d',console=False))"
    step "phase6 (strata_derived)" "$PY" -m pipeline.run --phase 6 &&
    step "build dataset3d"         "$PY" -c "$BUILD3D" &&
    step "train B1"                "$PY" -m ml.train3d --folds 5 &&
    step "train B2 (--physics)"    "$PY" -m ml.train3d --physics --folds 5 &&
    step "regenerate ML_REPORT"    "$PY" -m ml.report &&
    touch "$DONE"
    if [ -f "$DONE" ]; then
      echo "$(date -u) === AUTO PHASE6+B2 DONE (strata_derived + cv_b1/cv_b1_physics + ML_REPORT) ===" >> "$LOG"
    else
      echo "$(date -u) !!! AUTO PHASE6+B2 FAILED at the step logged above; not retried. strata_derived/dataset3d.pt may be partially rebuilt — re-run manually once resolved. !!!" >> "$LOG"
    fi
    break
  fi
  sleep "$INTERVAL"
done
