import os
import paramiko

def check_hpc():
    hpc_host = os.environ.get("HPC_HOST", "59.103.246.24")
    hpc_port = int(os.environ.get("HPC_PORT", "52222"))
    hpc_user = os.environ.get("HPC_USER", "f233003")
    hpc_password = os.environ.get("HPC_PASSWORD")

    if not hpc_password:
        raise ValueError(
            "HPC_PASSWORD environment variable is not set. "
            "Please configure it in your environment or via a local .env file."
        )

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hpc_host, port=hpc_port, username=hpc_user, password=hpc_password, timeout=10)

    commands = [
        "module load anaconda3/2024.06 && python -c 'import torch; print(\"Torch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())'",
        "module avail cuda",
        "cat /etc/os-release | grep PRETTY_NAME",
        "mkdir -p ~/virsa3d"
    ]

    for c in commands:
        print(f"=== Command: {c} ===")
        stdin, stdout, stderr = ssh.exec_command(c)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out: print("OUT:", out)
        if err: print("ERR:", err)

    ssh.close()

if __name__ == "__main__":
    check_hpc()

