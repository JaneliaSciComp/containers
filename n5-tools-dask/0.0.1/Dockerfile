# Create final image
FROM condaforge/mambaforge as builder
ARG GIT_TAG=0.0.1

WORKDIR /app
RUN git clone --branch ${GIT_TAG} --depth 1 https://github.com/JaneliaSciComp/n5-dask.git .

RUN mamba env create --file environment.yml -y \
    && mamba clean --tarballs -y \
    && mkdir -p /opt/conda/envs/myenv/etc/conda/activate.d \
    # It's necessary to set TMPDIR for running with Singularity, because /opt/conda will be read-only
    && echo "export TMPDIR=/tmp" > /opt/conda/envs/myenv/etc/conda/activate.d/env_vars.sh

# ps is needed by nextflow to collect task metrics
RUN apt-get install -y procps

RUN echo "#!/bin/bash" >> /entrypoint.sh \
    && echo "source /opt/conda/etc/profile.d/conda.sh" >> /entrypoint.sh \
    && echo "SCRIPT_NAME=\$1; shift" >> /entrypoint.sh \
    && echo "conda activate n5_dask" >> /entrypoint.sh \
    && echo 'python /app/src/n5_dask/${SCRIPT_NAME}.py "$@"' >> /entrypoint.sh \
    && chmod +x /entrypoint.sh
