#!/bin/bash --login
#SBATCH --job-name=Teissier-lines-128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=07-00:00:00
#SBATCH --chdir=/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance
#SBATCH --output=/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/snakemake-controller-gpu-128-line-orders.%j.out
#SBATCH --error=/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/snakemake-controller-gpu-128-line-orders.%j.err

set -euo pipefail

module purge
module load Miniforge3/25.11.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snakemake

snakemake -s workflow/Snakefile \
  --configfile workflow/config_gpu_128_line_orders.yaml \
  --executor slurm \
  --jobs 16 \
  --resources turbulence_spectra_slots=16 \
  --retries 20 \
  --rerun-triggers mtime input params \
  --consider-ancient run_turbulence=exe \
  --rerun-incomplete \
  --keep-going \
  --latency-wait 60 \
  --use-envmodules \
  --slurm-logdir workflow/logs/slurm-gpu-128-line-orders
