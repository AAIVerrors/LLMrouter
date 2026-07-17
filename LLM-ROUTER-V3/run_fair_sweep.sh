#!/usr/bin/env bash
# Three-run fairness sweep, sequential:
#   run1_jsq        : config as-is        (JSQ=True,  FAIR=1)  -> JSQ baseline
#   run2_flair_fair1: JSQ=False           (FAIR=1)             -> FLAIR, fairness on
#   run3_flair_fair0: JSQ=False, FAIR=0, FAIR_TARGET=0         -> FLAIR, fairness off
#
# config.py is backed up first and restored on exit (also on Ctrl-C).
# Each run's exact config is snapshotted next to its log.
# Launch inside tmux/nohup: bash run_fair_sweep.sh

set -u
cd "$(dirname "$0")"

PY=/home/braylon/miniconda3/envs/p2l/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="logs/sweep_${STAMP}"
mkdir -p "$OUT"

cp config.py "${OUT}/config_original.py"
restore() { cp "${OUT}/config_original.py" config.py; echo "[sweep] config.py restored to original."; }
trap restore EXIT

set_jsq() {
    sed -i "s/^\( *JSQ = \).*/\1$1/" config.py
}
set_fair() {
    sed -i "s/^\( *FAIR_TARGET = \)[0-9.]*/\1$1/" config.py
    sed -i "s/^\( *FAIR = \)[0-9.]*/\1$1/" config.py
}

run_one() {
    local name="$1"
    echo "===================================================================="
    echo "[sweep] START ${name}   $(date)"
    "$PY" - <<'EOF'
from config import Config
print(f"[sweep]   FAIRNESS_MODE={getattr(Config, 'FAIRNESS_MODE', 'legacy')}"
      f"  JSQ={Config.JSQ}  FAIR={Config.FAIR}  FAIR_TARGET={Config.FAIR_TARGET}"
      f"  DATASET={Config.DATASET_NAME}")
EOF
    echo "===================================================================="
    cp config.py "${OUT}/config_${name}.py"
    PYTHONUNBUFFERED=1 WANDB_NAME="${STAMP}_${name}" \
        "$PY" main.py 2>&1 | tee "${OUT}/${name}.log"
    local rc=${PIPESTATUS[0]}
    echo "[sweep] END ${name}   exit=${rc}   $(date)"
    echo "${name} exit=${rc}" >> "${OUT}/summary.txt"
    if [ "${rc}" -ne 0 ]; then
        echo "[sweep] WARNING: ${name} exited with ${rc}; continuing to next run."
    fi
}

# ---- run 1: current config as-is (JSQ baseline) ----
run_one "run1_jsq"

# ---- run 2: FLAIR with fairness on (JSQ off, FAIR stays 1) ----
set_jsq False
run_one "run2_flair_fair1"

# ---- run 3: FLAIR with fairness off (JSQ off, FAIR -> 0) ----
set_fair 0
run_one "run3_flair_fair0"

echo "[sweep] all runs finished. Logs and per-run config snapshots: ${OUT}"
cat "${OUT}/summary.txt"