#!/usr/bin/env bash
# Run mlflow docker image
# Usage: ./run_mlflow.sh <POSTGRES_DB_PASSWORD>
set -e

# Set port and aws profile
PORT=5500 \
AWS_PROFILE="default"  \
BUCKET="s3://numerai-v1/mlflow/" \
DB_USERNAME="postgres" \
DB_PASSWORD=$1 \
DB_URI="mlflow.cwtakrybmksl.us-east-2.rds.amazonaws.com" \
DB_PORT=5432 \
DB_NAME="mlflow" \
&& sudo docker run --interactive -t \
    -v ~/mlruns:/mlflow/mlruns/ \
    --publish "${PORT}":"${PORT}" \
    --env FILE_DIR="/mlflow" \
    --env PORT="${PORT}" \
    --env DB_USERNAME=${DB_USERNAME} \
    --env DB_PASSWORD=${DB_PASSWORD} \
    --env DB_URI=${DB_URI} \
    --env DB_PORT=${DB_PORT} \
    --env DB_NAME=${DB_NAME} \
    --env BUCKET=${BUCKET} \
    --env AWS_ACCESS_KEY_ID=`aws configure get ${AWS_PROFILE}.aws_access_key_id` \
    --env AWS_SECRET_ACCESS_KEY=`aws configure get ${AWS_PROFILE}.aws_secret_access_key` \
    --expose="${PORT}" \
    vishnups/mlflow-visp