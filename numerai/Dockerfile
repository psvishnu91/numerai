# Docker image to run the numerai modelling scripts.
# docker build  --platform linux/amd64 -t vishnups/numerai-visp .
#  && docker push vishnups/numerai-visp
FROM python:3.10.0
RUN mkdir /numerai/
COPY . /numerai/
WORKDIR /numerai/
RUN pip install -r requirements-minimal.txt
CMD bash