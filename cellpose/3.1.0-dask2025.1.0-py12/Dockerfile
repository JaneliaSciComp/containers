FROM  ghcr.io/janeliascicomp/dask:2025.1.0-py12-ol9

LABEL software="cellpose"
LABEL version="3"
LABEL base_image="janeliascicomp/dask:2025.1.0-py12-ol9"
LABEL software.version="3.1.0-dask2025.1.0-py12"
LABEL about.summary="A generalist algorithm for cell and nucleus segmentation."
LABEL about.home="https://github.com/MouseLand/cellpose"
LABEL about.license="BSD-3-Clause"
LABEL about.license_file="https://github.com/MouseLand/cellpose/blob/main/LICENSE"
LABEL about.documentation="https://cellpose.readthedocs.io/en/latest/"
LABEL extra.binaries="/opt/mambaforge/bin"
LABEL extra.scripts="/opt/scripts/cellpose"
LABEL about.tags="implemented-in::python"
LABEL conda_forge.minforge.version="24.11.3-0"
LABEL python.version="3.12"
LABEL extra.identifiers.biotools=cellpose
LABEL bioimagetools.image.authors="goinac@janelia.hhmi.org,rokickik@janelia.hhmi.org"

ARG TARGETPLATFORM

RUN dnf group install -y "Development Tools"

COPY conda-env.yml /tmp/

RUN mamba env update -n base -f /tmp/conda-env.yml

ENV CELLPOSE_LOCAL_MODELS_PATH=

WORKDIR /opt/scripts/cellpose

RUN mkdir -p /etc/cellpose/config

COPY configs/* /etc/cellpose/config/
COPY scripts/python /opt/scripts/cellpose

RUN pip install .
