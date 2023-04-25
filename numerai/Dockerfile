# Docker image to run the numerai modelling scripts.
# docker build  --platform linux/amd64 -f Dockerfile -t vishnups/numerai-visp .
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
        postgresql-client \
        tmux
COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN cd / && git clone https://github.com/vispz/numerai.git
# Allow outside access to the notebook server
RUN jupyter notebook --generate-config \
    && echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
RUN mkdir /numerai/data /numerai/.aws/
WORKDIR /numerai/
CMD bash
