#!/usr/bin/env bash
# Run the manuscript protocol in resumable, independently logged chunks.
# Default JOBS=1 avoids contention in wall-clock benchmarks.
set -uo pipefail
cd -- "$(dirname -- "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=${PYTHON:-python3}
JOBS=${JOBS:-1}
RESULTS_DIR=${LAPG_RESULTS_DIR:-results}
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS must be a positive integer." >&2
    exit 2
fi
mkdir -p -- "$RESULTS_DIR/logs"
pids=()
run() {
    local output=$1
    shift
    local destination="$RESULTS_DIR/$output"
    local marker="$destination.done"
    if [[ -s "$destination" && -f "$marker" ]]; then
        echo "SKIP $output"
        return 0
    fi
    echo "START $output"
    if "$PYTHON" experiments.py "$@" --results-dir "$RESULTS_DIR" --save-inputs > "$RESULTS_DIR/logs/$output.log" 2>&1; then
        if [[ ! -s "$destination" ]]; then
            echo "FAILED: expected output is missing: $destination" >&2
            return 1
        fi
        touch -- "$marker"
        echo "DONE $output"
    else
        echo "FAILED $output; see $RESULTS_DIR/logs/$output.log" >&2
        return 1
    fi
}
submit() {
    while (( $(jobs -rp | wc -l) >= JOBS )); do
        sleep 1
    done
    run "$@" &
    pids+=("$!")
}
submit verification.txt verify
submit example1d.json example
for indices in 0_10 10_20 20_30; do
    submit "synth_500x2000_${indices}.csv" synth --m 500 --n 2000 --seeds "${indices/_/:}"
done
for indices in 0_5 5_10 10_15 15_20; do
    submit "synth_1000x5000_${indices}.csv" synth --m 1000 --n 5000 --seeds "${indices/_/:}"
    for dataset in breast_cancer digits_parity; do
        for noise in 0.0 0.2; do
            submit "real_${dataset}_${noise}_${indices}.csv" real --data "$dataset" --noise "$noise" --reps "${indices/_/:}"
        done
    done
done
submit ablation.csv ablation
submit profile.json profile
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "One or more experiment chunks failed. Inspect logs before generating a report." >&2
    exit "$status"
fi
"$PYTHON" report.py --results-dir "$RESULTS_DIR" --output-dir reports
