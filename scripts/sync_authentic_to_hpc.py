import os
import paramiko
import time

def sync_and_run_cluster():
    hpc_host = os.environ.get("HPC_HOST", "59.103.246.24")
    hpc_port = int(os.environ.get("HPC_PORT", "52222"))
    hpc_user = os.environ.get("HPC_USER", "f233003")
    hpc_password = os.environ.get("HPC_PASSWORD")

    print("=" * 80)
    print("VIRSA3D: SUBMITTING REAL ARCHITECTURAL RUN TO FAST-NUCES CFD HPC CLUSTER")
    print("Partition: cpujobs / gpujobs | Authentic Front Facade & Flat Roof Model")
    print("=" * 80)

    if not hpc_password:
        raise ValueError(
            "HPC_PASSWORD environment variable is not set. "
            "Please configure it in your environment or via a local .env file."
        )

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hpc_host, port=hpc_port, username=hpc_user, password=hpc_password, timeout=15)

    sftp = ssh.open_sftp()
    remote_dir = f"/home/{hpc_user}/virsa3d"
    
    try:
        sftp.mkdir(f"{remote_dir}/photos")
    except Exception:
        pass

    base_dir = "C:/Users/TAQI KHOKHAR/.gemini/antigravity/brain/16b5837f-f7cc-4279-96ee-8e0858e9b673/.user_uploaded"
    sftp.put(f"{base_dir}/media_1794244826873_fae230c1.jpg", f"{remote_dir}/photos/front_facade.jpg")
    sftp.put(f"{base_dir}/media_1794244894299_6f33e079.png", f"{remote_dir}/photos/roof_plane.png")
    print("-> Uploaded authentic user photos to HPC cluster.")

    # Upload cluster runner script
    sftp.put("scripts/hpc_full_power_runner.py", f"{remote_dir}/hpc_full_power_runner.py")
    
    # Slurm submission script for 40 cores
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=virsa3d_authentic
#SBATCH --partition=cpujobs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=virsa3d_authentic_%j.out
#SBATCH --error=virsa3d_authentic_%j.err

echo "=== FAST-NUCES CFD HPC CLUSTER: AUTHENTIC ARCHITECTURE RUN ==="
echo "Node: $(hostname)"
echo "Assigned Cores: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"
echo "Start Time: $(date)"

module load anaconda3/2024.06
cd {remote_dir}
python hpc_full_power_runner.py
echo "End Time: $(date)"
"""
    with open("scripts/submit_authentic.slurm", "w") as f:
        f.write(slurm_content)
    sftp.put("scripts/submit_authentic.slurm", f"{remote_dir}/submit_authentic.slurm")

    # Submit job
    stdin, stdout, stderr = ssh.exec_command(f"cd {remote_dir} && sbatch submit_authentic.slurm")
    sub_res = stdout.read().decode().strip()
    print(f"-> Slurm Job: {sub_res}")

    time.sleep(3)
    stdin, stdout, stderr = ssh.exec_command(f"ls -lt {remote_dir}/virsa3d_authentic_*.out 2>/dev/null | head -n 1")
    latest_out = stdout.read().decode().strip()
    if latest_out:
        fname = latest_out.split()[-1]
        print(f"\n--- CLUSTER EXECUTION LOG ({fname}) ---")
        stdin, stdout, stderr = ssh.exec_command(f"cat {fname}")
        print(stdout.read().decode())

    sftp.close()
    ssh.close()

if __name__ == "__main__":
    sync_and_run_cluster()
