#!/bin/bash --login
#SBATCH --job-name=100turb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00-07:00:00
#SBATCH --output=/mnt/home/wysongj2/athenapk-fourth-Berta24/workflow/logs/snakemake-controller.%j.out
#SBATCH --error=/mnt/home/wysongj2/athenapk-fourth-Berta24/workflow/logs/snakemake-controller.%j.err

cd /mnt/home/wysongj2/athenapk-fourth-Berta24

# dry run command for copy paste
# snakemake -s workflow/Snakefile   --configfile workflow/config.yaml -n
# snakemake -s workflow/Snakefile   --configfile workflow/config.yaml workflow/report_profiles/mhd_wave_static.yaml -n

# activate conda env here if needed
module purge
module load Miniforge3/25.11.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate snakemake

module load foss/2023a
module load HDF5/1.14.0-gompi-2023a

snakemake -s workflow/Snakefile \
  --configfile workflow/config.yaml \
  --executor slurm \
  --jobs 10 \
  --rerun-incomplete \
  --keep-going \
  --latency-wait 60 \
  --use-envmodules \
  --slurm-logdir workflow/logs/slurm


# /mnt/home/wysongj2/.conda/envs/snakemake/bin/snakemake \
#   -s workflow/Snakefile \
#   --configfile workflow/config.yaml \
#   --report test-suite-uct-ct-glm/athenapk_3D_all_tests.zip

# snakemake -s workflow/Snakefile \
#   --configfile workflow/config.yaml workflow/report_profiles/mhd_wave_static.yaml \
#   --executor slurm \
#   --jobs 20 \
#   --keep-going \
#   --rerun-incomplete \
#   --latency-wait 60 \
#   --slurm-logdir workflow/logs/slurm \
#   --consider-ancient \
#     run_linear_wave_static=executable \
#     make_linear_wave_static_mesh_plot=executable

# After the workflow finishes, generate its combined static-wave report with:
# workflow/make_mhd_wave_report.sh
