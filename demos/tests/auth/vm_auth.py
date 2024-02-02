import sys
import paramiko
import time
import os
import re

from googleapiclient import discovery
from google.oauth2.service_account import Credentials


def authenticate_vm(path):
    credentials = Credentials.from_service_account_file(path)
    return discovery.build("compute", "v1", credentials=credentials)


def _start_ssh_session(response, creds, username, passphrase):
    external_ip = response["networkInterfaces"][0]["accessConfigs"][0]["natIP"]
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    max_retries = 3
    retry_delay = 10  # seconds

    for _ in range(max_retries):
        # ssh connection fails non-deterministically
        try:
            ssh.connect(
                external_ip,
                username=username,
                key_filename=creds,
                passphrase=passphrase,
                timeout=10,
                channel_timeout=10,
            )
            print("SSH session successful !")
            break
        except paramiko.ssh_exception.NoValidConnectionsError as e:
            print(f"SSH Exception(NoValidConnectionsError): {e}")
            time.sleep(retry_delay)
        except paramiko.ssh_exception.SSHException as e:
            print(f"SSH Exception(General): {e}")
        except paramiko.ssh_exception.AuthenticationException as e:
            print(f"Authentication failed: {e}")
            return

    # Open an SSH session
    transport = ssh.get_transport()
    channel = transport.open_session()

    # Execute the command on the instance in the background
    command = "cd actions-runner && ./run.sh"
    channel.exec_command(command)

    # Close the SSH session immediately and keep the channel open
    ssh.close()


def start_runner(
    creds,
    ssh_creds,
    ssh_user,
    key_passphrase,
    id="gpu-insatnce",
    zone="us-central1-a",
    instance="demos-tests",
):
    compute = authenticate_vm(creds)
    compute.instances().start(project=id, zone=zone, instance=instance).execute()

    max_wait_time = 600
    wait_interval = 10
    waited_time = 0
    response = None

    while waited_time < max_wait_time:
        response = (
            compute.instances().get(project=id, zone=zone, instance=instance).execute()
        )
        status = response.get("status")

        if status == "RUNNING":
            break

        time.sleep(wait_interval)
        waited_time += wait_interval

    if waited_time >= max_wait_time:
        # try starting once again
        compute.instances().start(project=id, zone=zone, instance=instance).execute()
        raise Exception(f"Instance {instance} did not start within the expected time.")

    print("Startup Successful (VM ON)")

    # Once the instance is running, start the SSH session
    _start_ssh_session(response, ssh_creds, ssh_user, key_passphrase)


def stop_runner(creds, instance="demos-tests"):
    compute = authenticate_vm(creds)
    compute.instances().stop(
        project="gpu-insatnce", zone="us-central1-a", instance=instance
    ).execute()
    time.sleep(25)

    print("Cleanup Successful (VM OFF)")


if __name__ == "__main__":
    ssh_user, key_passphrase, stop_vm, instance = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    gcp_credentials = "gcp_auth.json"
    ssh_credentials = "~/.ssh/id_rsa"
    
    if stop_vm == "true":
        # Stop the instance
        stop_runner(gcp_credentials, instance=instance)
    else:
        # Start the instance
        ssh_key_path = os.path.expanduser(ssh_credentials)
        start_runner(gcp_credentials, ssh_key_path, ssh_user, key_passphrase, instance=instance)
