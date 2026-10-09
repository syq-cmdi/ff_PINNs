#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-python3}
MODE=${1:-verify}
OUT=${REPRO_OUTPUT:-"$ROOT/reproduced"}
PINN_DEVICE=${PINN_DEVICE:-mps}

verify_code() {
  "$PYTHON" code/build_code_manifest.py --check
  "$PYTHON" -m compileall -q code
  echo "Code-only release verification passed"
}

prepare_output() {
  if [[ -e "$OUT" ]]; then
    echo "Refusing to overwrite existing reproduction directory: $OUT" >&2
    exit 2
  fi
  mkdir -p "$OUT"
}

run_reference() {
  prepare_output
  "$PYTHON" code/fetch_ks_archive.py --output "$OUT/inputs/KS_raissi.mat"
  "$PYTHON" code/ks_scale_reference_audit.py \
    --mat "$OUT/inputs/KS_raissi.mat" --output "$OUT/reference"
  "$PYTHON" code/benchmark_forward_runtime.py \
    --reference "$OUT/reference/ks_reference_corrected.npz" \
    --output "$OUT/reference/etdrk4_runtime_audit.json" --warmup 3 --repeats 11
  echo "Reference workflow completed in $OUT"
}

run_inverse_sweeps() {
  local inverse="$OUT/inverse"
  for seed in 20260712 20260713 20260714 20260715 20260716; do
    for noise in 0 0.005 0.01 0.02; do
      "$PYTHON" code/ks_sparse_inverse.py \
        --method fourier --reference "$OUT/reference/ks_reference_corrected.npz" \
        --output-dir "$inverse" --noise-rel "$noise" --n-times 101 --n-space 128 \
        --fourier-modes 12 --galerkin-modes 8 --seed "$seed" --tag baseline_sweep
    done
  done
  "$PYTHON" code/summarize_inverse_sweep.py \
    --input "$inverse" --output "$inverse/summary"
}

run_full() {
  run_reference

  if [[ -n "${LEGACY_CHECKPOINT:-}" ]]; then
    "$PYTHON" code/audit_legacy_observation_pinn.py \
      --checkpoint "$LEGACY_CHECKPOINT" \
      --reference "$OUT/reference/ks_reference_corrected.npz" \
      --output "$OUT/legacy_audit" --points 4096
  else
    echo "Skipping optional legacy-network audit; set LEGACY_CHECKPOINT to a local weight file"
  fi

  "$PYTHON" code/fetch_fetrig_subset.py --output "$OUT/data/fetrig"
  "$PYTHON" code/analyze_fetrig_subset.py \
    --input "$OUT/data/fetrig" --reference "$OUT/reference/ks_reference_corrected.npz" \
    --output "$OUT/fetrig"
  run_inverse_sweeps

  "$PYTHON" -u code/train_hc_marching_ks_pinn.py \
    --reference "$OUT/reference/ks_reference_corrected.npz" \
    --seeds 0,1,2,3,4 --windows 40 --adam-steps 1000 --lbfgs-iterations 200 \
    --collocation 1024 --validation-collocation 4096 --harmonics 8 \
    --state-modes 24 --state-grid 512 --width 64 --depth 5 \
    --learning-rate 0.001 --device "$PINN_DEVICE" --output "$OUT/marching_probe"
  "$PYTHON" code/refresh_marching_metrics.py "$OUT/marching_probe"
  "$PYTHON" code/train_hc_marching_ks_pinn.py \
    --reference "$OUT/reference/ks_reference_corrected.npz" \
    --output "$OUT/marching_probe" --resume-summary
  "$PYTHON" code/build_publication_metrics.py \
    --directory "$OUT/marching_probe" \
    --reference "$OUT/reference/ks_reference_corrected.npz" \
    --scale-audit "$OUT/reference/scale_reference_audit.json" \
    --convergence "$OUT/reference/etdrk4_convergence.csv" \
    --runtime-audit "$OUT/reference/etdrk4_runtime_audit.json" \
    --output "$OUT/marching_probe/publication_metrics.json"
  "$PYTHON" code/promote_marching_results.py \
    --source "$OUT/marching_probe" --target "$OUT/marching_final" --promote
  echo "Full numerical workflow completed in $OUT"
}

case "$MODE" in
  verify) verify_code ;;
  reference) run_reference ;;
  full) run_full ;;
  *)
    echo "Usage: ./reproduce.sh [verify|reference|full]" >&2
    exit 2
    ;;
esac
