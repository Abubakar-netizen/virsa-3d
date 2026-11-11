#!/bin/bash
#SBATCH --job-name=virsa3d_gpu
#SBATCH --partition=gpujobs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=virsa3d_%j.out
#SBATCH --error=virsa3d_%j.err

echo "=== FAST-NUCES CFD HPC CLUSTER: VIRSA3D GPU JOB ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

module load anaconda3/2024.06

cd /home/f233003/virsa3d

python hpc_gpu_virsa3d.py

echo "Job execution completed at $(date)"

