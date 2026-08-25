#!/usr/bin/env bash

# Compare the GLM and UCT MHD schemes on one V100 at a fixed 128^3 resolution.
#
# The 64^3 meshblock case (eight blocks) is run first, followed by one 128^3
# block and sixty-four 32^3 blocks. Each case runs for 100 cycles. Parthenon's
# performance timer and MeshBlock counter are reset after cycle 20, so the
# reported timing covers the final 80 cycles.
#
# Run from anywhere on a V100 node with:
#
#   ./performance_testing/run_v100_128_scheme_sweep.sh
#
# Useful overrides:
#
#   ATHENAPK_BIN=/path/to/athenaPK \
#   RESULT_ROOT=/path/to/results \
#     ./performance_testing/run_v100_128_scheme_sweep.sh

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

ATHENAPK_BIN=${ATHENAPK_BIN:-"${REPO_DIR}/build-gpu-v100/bin/athenaPK"}
INPUT_DECK=${INPUT_DECK:-"${SCRIPT_DIR}/linear_mhd_wave.in"}
RESULT_ROOT=${RESULT_ROOT:-"${SCRIPT_DIR}/results_v100_128_$(date +%Y%m%d_%H%M%S)"}

GLOBAL_N=128
TOTAL_CYCLES=100
WARMUP_CYCLES=20

# Run the requested eight-block case first. The other two entries complete the
# fixed-128^3 meshblock-decomposition sweep.
MESHBLOCK_SIZES=(64 128 32)
SCHEMES=(
  glm_weno3js
  glm_weno5js
  uct2_weno3
  uct4_hlld_point
  uct4_llf_line
)

if [[ ! -x "${ATHENAPK_BIN}" ]]; then
  echo "ERROR: AthenaPK executable is not available: ${ATHENAPK_BIN}" >&2
  echo "Set ATHENAPK_BIN to the V100 build if it is stored elsewhere." >&2
  exit 1
fi

if [[ ! -f "${INPUT_DECK}" ]]; then
  echo "ERROR: input deck is not available: ${INPUT_DECK}" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi was not found. Confirm that this is a V100 node." >&2
else
  echo "GPU selected for this sequential benchmark:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -n 1
fi

mkdir -p "${RESULT_ROOT}"
SUMMARY="${RESULT_ROOT}/summary.csv"
printf '%s\n' \
  "scheme,global_n,meshblock_n,num_blocks,total_cycles,warmup_cycles,measured_cycles,walltime_seconds,zone_cycles_per_wallsecond,status" \
  > "${SUMMARY}"

# AthenaPK/Kokkos uses a single GPU and a single host process for every case.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

for meshblock_n in "${MESHBLOCK_SIZES[@]}"; do
  blocks_per_dim=$((GLOBAL_N / meshblock_n))
  num_blocks=$((blocks_per_dim * blocks_per_dim * blocks_per_dim))

  for scheme in "${SCHEMES[@]}"; do
    fluid=""
    riemann=""
    reconstruction=""
    integrator=""
    convergence_order=""
    nghost=""
    uct_solver="none"

    case "${scheme}" in
      glm_weno3js)
        fluid="glmmhd"
        riemann="hlld"
        reconstruction="weno3js"
        integrator="rk3"
        convergence_order=2
        nghost=2
        ;;
      glm_weno5js)
        fluid="glmmhd"
        riemann="hlld"
        reconstruction="weno5js"
        integrator="rk3"
        convergence_order=2
        nghost=3
        ;;
      uct2_weno3)
        fluid="ucthlldmhd"
        riemann="hlld"
        reconstruction="weno3"
        integrator="rk3"
        convergence_order=2
        nghost=2
        ;;
      uct4_hlld_point)
        fluid="ucthlldmhd"
        riemann="hlld"
        reconstruction="weno5js"
        integrator="rk4"
        convergence_order=4
        nghost=3
        uct_solver="hlld_point"
        ;;
      uct4_llf_line)
        fluid="ucthlldmhd"
        riemann="llf"
        reconstruction="weno5js"
        integrator="rk4"
        convergence_order=4
        nghost=3
        uct_solver="llf_line"
        ;;
      *)
        echo "ERROR: unknown scheme: ${scheme}" >&2
        exit 1
        ;;
    esac

    run_id="${scheme}_N${GLOBAL_N}_MB${meshblock_n}"
    run_dir="${RESULT_ROOT}/${run_id}"
    log_file="${run_dir}/run.out"
    mkdir -p "${run_dir}"

    echo
    echo "Running ${run_id}: ${num_blocks} block(s), cycles ${WARMUP_CYCLES}-${TOTAL_CYCLES} timed"

    (
      cd "${run_dir}"
      "${ATHENAPK_BIN}" -i "${INPUT_DECK}" \
        parthenon/job/problem_id="${run_id}" \
        problem/linear_wave_mhd/compute_error=false \
        parthenon/mesh/refinement=none \
        parthenon/mesh/nghost="${nghost}" \
        parthenon/mesh/nx1="${GLOBAL_N}" \
        parthenon/mesh/nx2="${GLOBAL_N}" \
        parthenon/mesh/nx3="${GLOBAL_N}" \
        parthenon/meshblock/nx1="${meshblock_n}" \
        parthenon/meshblock/nx2="${meshblock_n}" \
        parthenon/meshblock/nx3="${meshblock_n}" \
        parthenon/mesh/packs_per_rank=1 \
        parthenon/time/integrator="${integrator}" \
        parthenon/time/nlim="${TOTAL_CYCLES}" \
        parthenon/time/tlim=1.0e30 \
        parthenon/time/perf_cycle_offset="${WARMUP_CYCLES}" \
        parthenon/time/ncycle_out=20 \
        hydro/fluid="${fluid}" \
        hydro/riemann="${riemann}" \
        hydro/reconstruction="${reconstruction}" \
        hydro/convergence_order="${convergence_order}" \
        hydro/uct_solver="${uct_solver}" \
        hydro/first_order_flux_correct=false \
        hydro/scratch_level=1 \
        parthenon/output0/dt=-1.0 \
        2>&1 | tee "${log_file}"
      exit "${PIPESTATUS[0]}"
    )
    run_status=$?

    if ((run_status == 0)); then
      walltime=$(awk '/^walltime used =/ {value=$4} END {print value}' "${log_file}")
      zcps=$(awk '/^zone-cycles\/wallsecond =/ {value=$3} END {print value}' "${log_file}")
      status="complete"
    else
      walltime=""
      zcps=""
      status="failed_${run_status}"
      echo "WARNING: ${run_id} failed with exit code ${run_status}; continuing the sweep." >&2
    fi

    printf '%s\n' \
      "${scheme},${GLOBAL_N},${meshblock_n},${num_blocks},${TOTAL_CYCLES},${WARMUP_CYCLES},$((TOTAL_CYCLES-WARMUP_CYCLES)),${walltime},${zcps},${status}" \
      >> "${SUMMARY}"
  done
done

echo
echo "Benchmark sweep finished."
echo "Summary: ${SUMMARY}"
column -s, -t "${SUMMARY}" 2>/dev/null || sed -n '1,999p' "${SUMMARY}"
