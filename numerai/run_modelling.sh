#!/usr/bin/env bash
# Run mlflow docker image
# Usage: ./run_mlflow.sh <POSTGRES_DB_PASSWORD>
set -e

PORT=8888
sudo docker run --interactive -t \
    -v ~/.aws/credentials:/.aws/credentials
    -v ~/data:/numerai/data/ \
    --publish "${PORT}":"${PORT}" \
    --expose="${PORT}" \
    vishnups/numerai-visp
