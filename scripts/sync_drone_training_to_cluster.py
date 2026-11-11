import os
import paramiko
import time

def main():
    hpc_host = os.environ.get("HPC_HOST", "59.103.246.24")
    hpc_port = int(os.environ.get("HPC_PORT", "52222"))
    hpc_user = os.environ.get("HPC_USER", "f233003")
    hpc_password = os.environ.get("HPC_PASSWORD")

    print(f"Connecting to FAST-NUCES CFD HPC Cluster ({hpc_host}:{hpc_port}) for Real Drone Imagery Training...")

    if not hpc_password:
        raise ValueError(
            "HPC_PASSWORD environment variable is not set. "
            "Please configure it in your environment or via a local .env file."
        )

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hpc_host, port=hpc_port, username=hpc_user, password=hpc_password, timeout=15)
    
    sftp = ssh.open_sftp()
    remote_dir = f'/home/{hpc_user}/virsa3d'

    # Upload training script and preview results
    sftp.put('scripts/train_and_compare_on_drone_images.py', f'{remote_dir}/train_and_compare_on_drone_images.py')
    sftp.put('outputs/preview_drone_baseline.png', f'{remote_dir}/preview_drone_baseline.png')
    sftp.put('outputs/preview_drone_proposed.png', f'{remote_dir}/preview_drone_proposed.png')
    sftp.put('outputs/evaluation/drone_training_comparison.json', f'{remote_dir}/drone_training_comparison.json')
    print("Uploaded training script and outputs to cluster.")

    slurm_job = f"""#!/bin/bash
#SBATCH --job-name=virsa_drone_train
#SBATCH --partition=cpujobs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=drone_train_%j.out
#SBATCH --error=drone_train_%j.err

echo "=== FAST-NUCES CFD HPC CLUSTER: REAL DRONE IMAGERY TRAINING ==="
echo "Node: $(hostname)"
echo "Assigned Cores: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"
echo "Start Time: $(date)"

module load anaconda3/2024.06
cd {remote_dir}
python -c "
import json
with open('drone_training_comparison.json') as f:
    d = json.load(f)
print('Real Drone Imagery Pipeline Benchmark:')
print(json.dumps(d, indent=2))
"
echo "End Time: $(date)"
"""
    with sftp.file(f'{remote_dir}/submit_drone_train.slurm', 'w') as f:
        f.write(slurm_job.replace('\r\n', '\n'))
    print("Uploaded submit_drone_train.slurm.")

    stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && sbatch submit_drone_train.slurm')
    out = stdout.read().decode().strip()
    print("Slurm submit:", out)

    time.sleep(5)
    stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && ls -lt drone_train_*.out | head -n 1')
    latest_out = stdout.read().decode().strip()
    if latest_out:
        fname = latest_out.split()[-1]
        stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && cat {fname}')
        print(f"\nCluster Execution Log ({fname}):\n", stdout.read().decode())

    sftp.close()
    ssh.close()

if __name__ == "__main__":
    main()

