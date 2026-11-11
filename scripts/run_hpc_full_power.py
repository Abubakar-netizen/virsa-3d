import os
import paramiko
import time

def submit_full_power():
    hpc_host = os.environ.get("HPC_HOST", "59.103.246.24")
    hpc_port = int(os.environ.get("HPC_PORT", "52222"))
    hpc_user = os.environ.get("HPC_USER", "f233003")
    hpc_password = os.environ.get("HPC_PASSWORD")

    print("=" * 80)
    print("SUBMITTING FULL-POWER MULTI-CORE JOB TO FAST-NUCES CFD HPC CLUSTER")
    print("Partition: cpujobs | Node Target: compute3 (80-core 125GB node)")
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
    sftp.put("scripts/hpc_full_power_runner.py", f"{remote_dir}/hpc_full_power_runner.py")
    sftp.put("scripts/submit_hpc_full_power.slurm", f"{remote_dir}/submit_hpc_full_power.slurm")
    print("-> Uploaded runner script & Slurm batch configuration.")

    stdin, stdout, stderr = ssh.exec_command(f"cd {remote_dir} && sbatch submit_hpc_full_power.slurm")
    submission_msg = stdout.read().decode().strip()
    print(f"-> {submission_msg}")

    # Wait 3 seconds and check queue / output
    time.sleep(3)
    stdin, stdout, stderr = ssh.exec_command(f"squeue -u {hpc_user}")
    print("\n--- USER QUEUE STATUS ---")
    print(stdout.read().decode())

    # Check output log
    stdin, stdout, stderr = ssh.exec_command(f"ls -lt {remote_dir}/virsa3d_fullpower_*.out 2>/dev/null | head -n 1")
    latest_out = stdout.read().decode().strip()
    if latest_out:
        fname = latest_out.split()[-1]
        print(f"\n--- READING EXECUTION LOG ({fname}) ---")
        stdin, stdout, stderr = ssh.exec_command(f"cat {fname}")
        print(stdout.read().decode())

    sftp.close()
    ssh.close()

if __name__ == "__main__":
    submit_full_power()

