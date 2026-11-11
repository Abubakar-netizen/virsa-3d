#!/bin/bash
#SBATCH --job-name=virsa3d_fullpower
#SBATCH --partition=cpujobs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=virsa3d_fullpower_%j.out
#SBATCH --error=virsa3d_fullpower_%j.err

echo "=== FAST-NUCES CFD HPC CLUSTER: FULL-POWER RUN ==="
echo "Node: $(hostname)"
echo "Assigned Cores: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"
echo "Start Time: $(date)"

module load anaconda3/2024.06

cd /home/f233003/virsa3d

python hpc_full_power_runner.py

echo "End Time: $(date)"

