#!/usr/bin/env bash

# Run the GPU workflow directly on dev-amd20-v100 without submitting Slurm jobs.
# GPU_SELECTION may be one physical GPU index or a comma-separated list.
# Usage: ./workflow/run_snakemake_gpu_local.sh GPU_SELECTION [CONFIG_FILE] [TARGET ...]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 GPU_SELECTION [CONFIG_FILE] [TARGET ...]" >&2
  echo "Examples: GPU_SELECTION=2 or GPU_SELECTION=1,2" >&2
  echo "Run 'nvidia-smi' first and choose unused GPUs (0-3)." >&2
  exit 2
fi

gpu_selection="$1"
config_file="${2:-workflow/config_gpu.yaml}"
if [[ $# -gt 2 ]]; then
  snakemake_targets=("${@:3}")
else
  snakemake_targets=()
fi
if [[ ! "$gpu_selection" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_SELECTION must be an index or comma-separated indices." >&2
  exit 2
fi
IFS=',' read -r -a gpu_indices <<< "$gpu_selection"
gpu_count="${#gpu_indices[@]}"

declare -A selected_gpus=()
for gpu_index in "${gpu_indices[@]}"; do
  if [[ -n "${selected_gpus[$gpu_index]:-}" ]]; then
    echo "GPU $gpu_index was selected more than once." >&2
    exit 2
  fi
  selected_gpus[$gpu_index]=1
done

host_name="$(hostname -s)"
if [[ "$host_name" != "dev-amd20-v100" ]]; then
  echo "Refusing local GPU execution on $host_name." >&2
  echo "Log into dev-amd20-v100 before running this script." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; no NVIDIA GPU is visible." >&2
  exit 1
fi

for gpu_index in "${gpu_indices[@]}"; do
  if ! nvidia-smi -i "$gpu_index" >/dev/null 2>&1; then
    echo "GPU $gpu_index is not available on $host_name." >&2
    exit 1
  fi

  active_pids="$(nvidia-smi -i "$gpu_index" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$active_pids" ]]; then
    echo "GPU $gpu_index already has an active compute process:" >&2
    nvidia-smi -i "$gpu_index" >&2
    exit 1
  fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
snakemake_bin="/mnt/home/wysongj2/.conda/envs/snakemake/bin/snakemake"

if [[ "$config_file" != /* ]]; then
  config_file="$repo_root/$config_file"
fi
if [[ ! -f "$config_file" ]]; then
  echo "Configuration file not found: $config_file" >&2
  exit 1
fi

if [[ ! -x "$snakemake_bin" ]]; then
  echo "Snakemake executable not found at $snakemake_bin." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$gpu_selection"
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
export ATHENAPK_LOCAL_GPU_COUNT="$gpu_count"
export ATHENAPK_LOCAL_MPI_TASKS="$gpu_count"
export OMP_NUM_THREADS=1

mkdir -p "$repo_root/workflow/logs"
timestamp="$(date +%Y%m%d-%H%M%S)"
gpu_log="$repo_root/workflow/logs/gpu-local-${timestamp}.csv"

echo "Running locally on $host_name, physical GPU(s) $gpu_selection."
echo "MPI ranks: $gpu_count (one rank per selected GPU)"
echo "Configuration: $config_file"
if [[ ${#snakemake_targets[@]} -gt 0 ]]; then
  echo "Targets: ${snakemake_targets[*]}"
else
  echo "Targets: workflow default (rule all)"
fi
echo "GPU telemetry: $gpu_log"
for gpu_index in "${gpu_indices[@]}"; do
  nvidia-smi -i "$gpu_index"
done

(
  echo "timestamp,index,memory.used [MiB],memory.total [MiB],utilization.gpu [%]"
  while true; do
    for gpu_index in "${gpu_indices[@]}"; do
      nvidia-smi -i "$gpu_index" \
        --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
    done
    sleep 1
  done
) >"$gpu_log" 2>&1 &
monitor_pid=$!

cleanup_monitor() {
  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" >/dev/null 2>&1 || true
}
trap cleanup_monitor EXIT INT TERM

cd "$repo_root"
"$snakemake_bin" \
  -s workflow/Snakefile \
  --configfile "$config_file" \
  --cores 1 \
  --resources gpu="$gpu_count" turbulence_spectra_slots=2 \
  --rerun-incomplete \
  --keep-going \
  --latency-wait 60 \
  --use-envmodules \
  "${snakemake_targets[@]}"

echo "Local GPU workflow completed."
