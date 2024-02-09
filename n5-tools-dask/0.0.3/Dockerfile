# Create final image
FROM condaforge/mambaforge as builder
ARG GIT_TAG=0.0.3

WORKDIR /app
RUN git clone --branch ${GIT_TAG} --depth 1 https://github.com/JaneliaSciComp/n5-tools-dask.git .

RUN mamba env create -n n5-tools-dask --file environment.yml -y \
    && mamba clean -a -y \
    && mkdir -p /opt/conda/envs/myenv/etc/conda/activate.d \
    # It's necessary to set TMPDIR for running with Singularity, because /opt/conda will be read-only
    && echo "export TMPDIR=/tmp" > /opt/conda/envs/n5-tools-dask/etc/conda/activate.d/env_vars.sh

# ps is needed by nextflow to collect task metrics
RUN apt-get install -y procps

RUN echo "#!/bin/bash" >> /entrypoint.sh \
    && echo "source /opt/conda/etc/profile.d/conda.sh" >> /entrypoint.sh \
    && echo "SCRIPT_NAME=\$1; shift" >> /entrypoint.sh \
    && echo "conda activate n5-tools-dask" >> /entrypoint.sh \
    && echo 'python /app/src/${SCRIPT_NAME}.py "$@"' >> /entrypoint.sh \
    && chmod +x /entrypoint.sh
