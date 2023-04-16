# docker build -t vishnups/mlflow-visp . && docker push vishnups/mlflow-visp
# Taken from https://github.com/aws-samples/amazon-sagemaker-mlflow-fargate/blob/main/container/Dockerfile
FROM python:3.9.0

RUN pip install \
    mlflow==2.0.1 \
    pymysql==1.0.2 \
    boto3 && \
    mkdir /mlflow/

CMD mlflow server \
   --host 0.0.0.0 \
   --port ${PORT} \
   --default-artifact-root ${BUCKET} \
   --workers 1
