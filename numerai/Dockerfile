# Docker image to run the numerai modelling scripts.
# docker build  --platform linux/amd64 -t vishnups/numerai-visp .
#  && docker push vishnups/numerai-visp
FROM python:3.10.0
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++ \
        curl \
        git \
        libomp-dev \
        vim \
        git \
        postgresql-client
COPY requirements-minimal.txt /
RUN pip install -r /requirements-minimal.txt
RUN cd / && git clone https://github.com/vispz/numerai.git
WORKDIR /numerai/
CMD bash
