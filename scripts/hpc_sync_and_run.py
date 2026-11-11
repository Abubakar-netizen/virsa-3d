import os
import sys
import json
import paramiko
from pathlib import Path

def sync_and_run():
    hpc_host = os.environ.get("HPC_HOST", "59.103.246.24")
    hpc_port = int(os.environ.get("HPC_PORT", "52222"))
    hpc_user = os.environ.get("HPC_USER", "f233003")
    hpc_password = os.environ.get("HPC_PASSWORD")

    print("=" * 80)
    print("VIRSA3D: HPC CLUSTER SYNCHRONIZATION & EXECUTION")
    print(f"Connecting to FAST-NUCES CFD Campus HPC ({hpc_host}:{hpc_port})...")
    print("=" * 80)

    if not hpc_password:
        raise ValueError(
            "HPC_PASSWORD environment variable is not set. "
            "Please configure it in your environment or via a local .env file."
        )

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hpc_host, port=hpc_port, username=hpc_user, password=hpc_password, timeout=15)
        print("[SSH SUCCESS] Connected to cluster master node!")

        sftp = ssh.open_sftp()
        print("[SFTP] Opening file transfer channel...")

        # Ensure directory on cluster
        ssh.exec_command(f"mkdir -p /home/{hpc_user}/virsa3d")

        # Upload files
        local_py = "scripts/hpc_gpu_virsa3d.py"
        remote_py = f"/home/{hpc_user}/virsa3d/hpc_gpu_virsa3d.py"
        sftp.put(local_py, remote_py)
        print(f"-> Uploaded {local_py} -> {remote_py}")

        local_slurm = "scripts/submit_gpu.slurm"
        remote_slurm = f"/home/{hpc_user}/virsa3d/submit_gpu.slurm"
        sftp.put(local_slurm, remote_slurm)
        print(f"-> Uploaded {local_slurm} -> {remote_slurm}")

        # Execute direct run with anaconda3 on the cluster
        print("\n[HPC EXECUTION] Running Virsa3D pipeline on cluster node...")
        exec_cmd = (
            f"cd /home/{hpc_user}/virsa3d && "
            "module load anaconda3/2024.06 && "
            "python hpc_gpu_virsa3d.py"
        )
        stdin, stdout, stderr = ssh.exec_command(exec_cmd)
        out_log = stdout.read().decode()
        err_log = stderr.read().decode()

        print("\n--- CLUSTER OUTPUT LOG ---")
        print(out_log)
        if err_log:
            print("--- STDERR ---")
            print(err_log)

        # Submit Slurm batch job for GPU queue
        print("\n[SLURM QUEUE] Submitting job to gpujobs partition...")
        sbatch_cmd = f"cd /home/{hpc_user}/virsa3d && sbatch submit_gpu.slurm"
        stdin, stdout, stderr = ssh.exec_command(sbatch_cmd)
        slurm_res = stdout.read().decode().strip()
        print(f"-> Slurm Submission Result: {slurm_res}")

        # Check queue
        stdin, stdout, stderr = ssh.exec_command(f"squeue -u {hpc_user}")
        print("\n--- CURRENT USER JOBS IN SLURM QUEUE ---")
        print(stdout.read().decode())

        # Download json output if available
        try:
            os.makedirs("outputs/evaluation", exist_ok=True)
            sftp.get("/home/f233003/virsa3d/hpc_execution_output.json", "outputs/evaluation/hpc_execution_output.json")
            print("-> Successfully downloaded hpc_execution_output.json to local outputs/evaluation/")
        except Exception as e:
            print("Note: Could not retrieve hpc_execution_output.json yet:", e)

        sftp.close()

    except Exception as e:
        print("[ERROR] HPC Connection/Execution Error:", e)
    finally:
        ssh.close()
        print("\n[DONE] HPC Cluster synchronization session closed.")

if __name__ == "__main__":
    sync_and_run()

