# Docker image to run the numerai modelling scripts.
# docker build  --platform linux/amd64 -t vishnups/numerai-visp .
#  && docker push vishnups/numerai-visp
FROM python:3.10.0
RUN mkdir /numerai/
COPY requirements-minimal.txt /numerai/
RUN pip install -r /numerai/requirements-minimal.txt
COPY . /numerai/
WORKDIR /numerai/
CMD bash