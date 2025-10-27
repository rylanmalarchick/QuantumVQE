import subprocess
import paramiko
import os

def submit_job_via_ssh(job_script_path, hpc_host, username):
    # SSH into HPC and submit job
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Password Auth
    ssh.connect(hpc_host, username=username, password='your_password')
    
    # Copy job script to HPC
    sftp = ssh.open_sftp()
    sftp.put(job_script_path, f'/tmp/{os.path.basename(job_script_path)}')
    
    # Submit job
    stdin, stdout, stderr = ssh.exec_command(f'qsub /tmp/{os.path.basename(job_script_path)}')
    job_id = stdout.read().decode().strip()
    
    ssh.close()
    return job_id

# Usage
job_id = submit_job_via_ssh('pbs_scripts/my_job.sh', 'hpc-headnode', 'username')
print(f"Job submitted: {job_id}")
