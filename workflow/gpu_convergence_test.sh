#!/bin/bash

# quick convergence test to run on a GPU

## from athenapk/ directory
# ./workflow/gpu_convergence_test.sh

# will run on the 1 gpu
# results will be in athenapk/gpu-linewave/

set -o pipefail

module purge
module load foss/2023a
module load HDF5/1.14.0-gompi-2023a
module load CUDA/12.3.0

export LD_LIBRARY_PATH="$EBROOTHDF5/lib:${LD_LIBRARY_PATH:-}"

APK="$PWD/build-gpu-v100/bin/athenaPK"
INPUT="$PWD/inputs/ct_glm_compatible/linear_mhd_wave.in"

for N in 8 16 32 64 128; do
    NX1=$((2 * N))
    RUNDIR="gpu-linwave/N${N}"

    mkdir -p "$RUNDIR"

    (
        cd "$RUNDIR"

        "$APK" -i "$INPUT" \
            parthenon/job/problem_id="linear_wave_gpu_N${N}" \
            problem/linear_wave_mhd/wave_flag=6 \
            problem/linear_wave_mhd/ang_2=-999.9 \
            parthenon/mesh/nx1="$NX1" \
            parthenon/mesh/nx2="$N" \
            parthenon/mesh/nx3="$N" \
            parthenon/meshblock/nx1="$NX1" \
            parthenon/meshblock/nx2="$N" \
            parthenon/meshblock/nx3="$N" \
            hydro/fluid=ucthlldmhd
    ) |& tee "$RUNDIR/run.out"
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTROOT="$REPO_ROOT/gpu-linwave"
COMBINED="$OUTROOT/linearwave-errors-6.dat"


for N in 8 16 32 64 128; do
    cat "$OUTROOT/N${N}/linearwave-errors-6.dat" >> "$COMBINED"
done

/mnt/home/wysongj2/.conda/envs/mhd/bin/python \
    "$REPO_ROOT/workflow/diagnostics_scripts/linear_mhd_wave_orders.py" \
    "$COMBINED" 6 \
    > "$OUTROOT/order.txt"

echo "Combined errors: $COMBINED"
echo "Convergence report: $OUTROOT/order.txt"