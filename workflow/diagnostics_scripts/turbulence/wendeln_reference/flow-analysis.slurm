#!/bin/bash --login
#SBATCH --job-name=flow
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32           
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --array=0-199%10           # one sub-job per file (00000 → 00100) with only 10 running at a time
#SBATCH --output=/mnt/scratch/wendelnc/Turbulence/02-Ms-0.2/compare-128-ENO/athenapk-eno3/logs/flow/%x_%A_%a.out
#SBATCH --error=/mnt/scratch/wendelnc/Turbulence/02-Ms-0.2/compare-128-ENO/athenapk-eno3/logs/flow/%x_%A_%a.err

source ~/athenapk.profile
module unload Python
conda activate APK25

echo "Analyzing file index $SLURM_ARRAY_TASK_ID"

idx=$(printf "%05d" $SLURM_ARRAY_TASK_ID)

infile="/mnt/scratch/wendelnc/Turbulence/02-Ms-0.2/compare-128-ENO/athenapk-eno3/parthenon.prim.${idx}.phdf"
outfile="/mnt/scratch/wendelnc/Turbulence/02-Ms-0.2/compare-128-ENO/athenapk-eno3/flow-${idx}-out.hdf5"

srun python ./run_analysis.py \
    --res 128 \
    --data_path "$infile" \
    --data_type AthenaPK \
    --kernels Gauss --type flow \
    --outfile "$outfile" \
    --eos adiabatic --gamma 1.0001  -forced -b

echo "File $infile complete."
