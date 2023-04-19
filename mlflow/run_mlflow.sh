#!/usr/bin/env bash
set -e

# Set port and aws profile
PORT=5500 AWS_PROFILE="mlflow"  \
&& sudo docker run --interactive -t \
    -v ~/mlruns:/mlflow/mlruns/ \
    --publish "${PORT}":"${PORT}" \
    --env FILE_DIR="/mlflow" \
    --env PORT="${PORT}" \
    --env BUCKET="s3://numerai-v1/" \
    --env AWS_ACCESS_KEY_ID=`aws configure get ${AWS_PROFILE}.aws_access_key_id` \
    --env AWS_SECRET_ACCESS_KEY=`aws configure get ${AWS_PROFILE}.aws_secret_access_key` \
    --expose="${PORT}" \
    vishnups/mlflow-visp
