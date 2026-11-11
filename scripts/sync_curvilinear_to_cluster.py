import os
import paramiko
import time

def main():
    hpc_host = os.environ.get("HPC_HOST", "59.103.246.24")
    hpc_port = int(os.environ.get("HPC_PORT", "52222"))
    hpc_user = os.environ.get("HPC_USER", "f233003")
    hpc_password = os.environ.get("HPC_PASSWORD")

    print(f"Connecting to FAST-NUCES CFD HPC Cluster ({hpc_host}:{hpc_port})...")

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
    
    sftp.put('scripts/hpc_curvilinear_runner.py', f'{remote_dir}/hpc_curvilinear_runner.py')
    print("Uploaded hpc_curvilinear_runner.py to cluster.")

    slurm_job = f"""#!/bin/bash
#SBATCH --job-name=virsa3d_curv40
#SBATCH --partition=cpujobs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=virsa_curv_%j.out
#SBATCH --error=virsa_curv_%j.err

echo "=== FAST-NUCES CFD HPC CLUSTER: 40-CORE CURVILINEAR RUN ==="
echo "Node: $(hostname)"
echo "Assigned Cores: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"
echo "Start Time: $(date)"

module load anaconda3/2024.06
cd {remote_dir}
python hpc_curvilinear_runner.py
echo "End Time: $(date)"
"""
    with sftp.file(f'{remote_dir}/submit_curv.slurm', 'w') as f:
        f.write(slurm_job.replace('\r\n', '\n'))
    print("Uploaded submit_curv.slurm.")

    stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && sbatch submit_curv.slurm')
    out = stdout.read().decode().strip()
    print("Slurm submit:", out)

    # Wait for execution
    time.sleep(5)
    stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && ls -lt virsa_curv_*.out | head -n 1')
    latest_out = stdout.read().decode().strip()
    if latest_out:
        fname = latest_out.split()[-1]
        stdin, stdout, stderr = ssh.exec_command(f'cd {remote_dir} && cat {fname}')
        print(f"\nCluster Execution Log ({fname}):\n", stdout.read().decode())

    sftp.close()
    ssh.close()

if __name__ == "__main__":
    main()

