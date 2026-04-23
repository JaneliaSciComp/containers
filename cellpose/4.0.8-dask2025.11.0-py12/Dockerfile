FROM ghcr.io/janeliascicomp/dask:2025.11.0-py12-ol9
ARG CELLPOSE_TOOLS_BRANCH=dev
ARG CELLPOSE_TOOLS_COMMIT=3a1ff5f

LABEL software="cellpose"
LABEL version="4"
LABEL base_image="janeliascicomp/dask:2025.11.0-py12-ol9"
LABEL software.version="4.0.8-dask2025.11.0-py12"
LABEL about.summary="A generalist algorithm for cell and nucleus segmentation."
LABEL about.home="https://github.com/MouseLand/cellpose"
LABEL about.license="BSD-3-Clause"
LABEL about.license_file="https://github.com/MouseLand/cellpose/blob/main/LICENSE"
LABEL about.documentation="https://cellpose.readthedocs.io/en/latest/"
LABEL extra.binaries="/opt/mambaforge/bin"
LABEL extra.scripts="/opt/scripts/cellpose"
LABEL about.tags="implemented-in::python"
LABEL conda_forge.minforge.version="25.11.0-1"
LABEL python.version="3.12"
LABEL extra.identifiers.biotools=cellpose
LABEL bioimagetools.image.authors="goinac@janelia.hhmi.org,rokickik@janelia.hhmi.org"

RUN dnf group install -y "Development Tools"

WORKDIR /opt/scripts/cellpose

RUN git clone --branch ${CELLPOSE_TOOLS_BRANCH} --depth 1 --progress https://github.com/JaneliaSciComp/cellpose-tools.git . \
 && git reset --hard ${CELLPOSE_TOOLS_COMMIT}

ENV PIP_ROOT_USER_ACTION=ignore

RUN  mamba env update -n base -f conda-env.yml

RUN pip install --root-user-action ignore -e .

RUN mamba clean -a -y \
 && rm -rf ~/.cache

RUN mkdir -p /etc/cellpose/config \
 && cp configs/* /etc/cellpose/config

 # Remove dev tools
 RUN dnf group remove -y "Development Tools" \
  && dnf clean all \
  && rm -rf /var/cache/dnf
