# docker build --platform linux/amd64 -t vishnups/mlflow-visp . && docker push vishnups/mlflow-visp
# Taken from https://github.com/aws-samples/amazon-sagemaker-mlflow-fargate/blob/main/container/Dockerfile
FROM python:3.9.0

RUN pip install \
    mlflow==2.0.1 \
    pymysql==1.0.2 \
    boto3 \
    psycopg2-binary && \
    mkdir -p /mlflow/mlruns/

WORKDIR /mlflow/

CMD mlflow server \
   --host 0.0.0.0 \
   --port ${PORT} \
   --default-artifact-root ${BUCKET} \
   --workers 2 \
   --backend-store-uri "postgresql://${DB_USERNAME}:${DB_PASSWORD}@${DB_URI}:${DB_PORT}/${DB_NAME}"
