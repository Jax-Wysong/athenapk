#!/bin/bash --login
# The controller is a lightweight CPU job. Individual simulation rules request
# V100 GPUs through workflow/config_gpu.yaml; diagnostics remain CPU jobs.
#SBATCH --job-name=Teissier-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# The controller must remain alive until the longest simulation and all
# downstream diagnostics finish. The longest configured case currently asks
# for 40 hours.
# Keep the controller alive while restartable simulation chunks queue and run.
#SBATCH --time=07-00:00:00
#SBATCH --output=/mnt/home/wysongj2/athenapk-fourth-Teissier24/workflow/logs/snakemake-gpu-controller.%j.out
#SBATCH --error=/mnt/home/wysongj2/athenapk-fourth-Teissier24/workflow/logs/snakemake-gpu-controller.%j.err

module purge
module load Miniforge3/25.11.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snakemake

cd /mnt/home/wysongj2/athenapk-fourth-Teissier24

snakemake -s workflow/Snakefile \
  --configfile workflow/config_gpu.yaml \
  --executor slurm \
  --jobs 4 \
  --resources turbulence_spectra_slots=2 \
  --retries 6 \
  --rerun-triggers mtime \
  --rerun-incomplete \
  --keep-going \
  --latency-wait 60 \
  --use-envmodules \
  --slurm-logdir workflow/logs/slurm-gpu
