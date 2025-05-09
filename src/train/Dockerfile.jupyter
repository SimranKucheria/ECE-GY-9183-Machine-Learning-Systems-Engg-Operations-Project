FROM quay.io/jupyter/pytorch-notebook:cuda12-latest

# USER ${NB_UID}
USER root

# Install all dependencies in a single layer to minimize image size
RUN apt-get update
RUN apt-get install default-jre -y
# RUN apt-get install default-jdk -y
RUN apt install openjdk-11-jdk -y

RUN pip install --pre --no-cache-dir \
    lightning \
    pynvml \
    mlflow \
    transformers==4.16.0 \
    opencv-python \
    timm \
    fairscale \
    pycocoevalcap \
    pycocotools \
    ruamel_yaml==0.16.6 \
    ray && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Additional environment setup if needed
ENV PYTHONPATH "${PYTHONPATH}:/home/${NB_USER}/work"