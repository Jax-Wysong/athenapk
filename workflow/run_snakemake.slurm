#!/bin/bash --login
#SBATCH --job-name=cpuTeissier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00-03:59:00
#SBATCH --output=/mnt/home/wysongj2/athenapk-fourth-Teissier24/workflow/logs/snakemake-controller.%j.out
#SBATCH --error=/mnt/home/wysongj2/athenapk-fourth-Teissier24/workflow/logs/snakemake-controller.%j.err

# Dry-run command for copy and paste:
# snakemake -s workflow/Snakefile   --configfile workflow/config.yaml -n

# snakemake -s workflow/Snakefile \
#   --configfile workflow/config_gpu.yaml \
#   --unlock

# activate conda env here if needed
module purge
module load Miniforge3/25.11.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate snakemake

module load foss/2023a
module load HDF5/1.14.0-gompi-2023a

snakemake -s workflow/Snakefile \
  --configfile workflow/config_local_4th_order.yaml \
  --executor slurm \
  --jobs 16 \
  --resources turbulence_spectra_slots=2 \
  --rerun-incomplete \
  --keep-going \
  --latency-wait 60 \
  --use-envmodules \
  --slurm-logdir workflow/logs/slurm
