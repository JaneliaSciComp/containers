ARG DASK_VERSION=2025.1.0

FROM oraclelinux:9

LABEL software="Dask"
LABEL version="1"
LABEL base_image="oraclelinux:9"
LABEL software.version="2025.1.0-py12-ol9"
LABEL about.summary="Dask is a flexible library for parallel computing in Python."
LABEL about.home="https://www.dask.org"
LABEL about.license="BSD-3-Clause"
LABEL about.license_file="https://github.com/dask/dask/blob/main/LICENSE.txt"
LABEL about.documentation="https://docs.dask.org/en/stable/"
LABEL extra.binaries="/opt/mambaforge/bin"
LABEL extra.scripts="/opt/scripts/daskscripts"
LABEL about.tags="implemented-in::python, interface::daemon, role::devel-lib"
LABEL conda_forge.minforge.version="24.11.3-0"
LABEL python.version="3.12"
LABEL bioimagetools.image.authors="goinac@janelia.hhmi.org,rokickik@janelia.hhmi.org"

ARG TARGETPLATFORM

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN dnf update -y && \
    dnf install -y \
        tar wget \
        hostname \
        procps-ng \
        net-tools \
        which

# Install miniconda
RUN wget https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-24.11.3-0-Linux-$(uname -m).sh \
    -O mamba-install.sh && \
    bash mamba-install.sh -b -p /opt/mambaforge

ENV PATH=/opt/mambaforge/bin:${PATH}

COPY conda-env.yml /tmp/

ENV NUMBA_CACHE_DIR=/tmp/numba_cache

RUN mamba env update -n base -f /tmp/conda-env.yml

RUN rm -rf /opt/mambaforge/pkgs

WORKDIR /opt/scripts/daskscripts

ENV DASK_CONFIG=/opt/scripts/daskscripts/config/dask-config.yml

# Add scripts
COPY scripts/* /opt/scripts/daskscripts/
COPY config /opt/scripts/daskscripts/config

RUN chmod 755 /opt/scripts/daskscripts/*.sh
