#!/usr/bin/env bash
# Upload credentials and scripts to ec2 instance.
# Run with `bash -x upload.sh <EC2 public DNS>`
# Example usage: `bash -x upload.sh ubuntu@ec2-3-17-6-177.us-east-2.compute.amazonaws.com`
set -e

LOCAL_CRED_FL=~/.aws/credentials_mlflow
PEM_FILE="~/visp-admin-aws-keypair.pem"
# The first argument to the script is the ec2 instance id
EC2_LOGIN=$1
scp -i ${PEM_FILE} ${LOCAL_CRED_FL} "${EC2_LOGIN}:~/credentials"
# Upload script to setup the ec2 instance
scp -i ${PEM_FILE} setup_mlflow.sh "${EC2_LOGIN}:~/"
scp -i ${PEM_FILE} run_mlflow.sh "${EC2_LOGIN}:~/"
ssh -i ${PEM_FILE} "${EC2_LOGIN}"