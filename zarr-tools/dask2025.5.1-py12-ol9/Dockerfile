# Create final image
FROM  ghcr.io/janeliascicomp/dask:2025.5.1-py12-ol9
ARG ZARR_TOOLS_BRANCH=main
ARG ZARR_TOOLS_COMMIT=c537faf

LABEL software="cellpose"
LABEL version="4"
LABEL base_image="janeliascicomp/dask:2025.5.1-py12-ol9"

RUN dnf group install -y "Development Tools"

WORKDIR /opt/scripts/zarr-tools

RUN git clone --branch ${ZARR_TOOLS_BRANCH} --depth 1 --progress https://github.com/JaneliaSciComp/cellpose-tools.git . \
 && git reset --hard ${ZARR_TOOLS_COMMIT}

ENV PIP_ROOT_USER_ACTION=ignore

RUN  mamba env update -n base -f conda-env.yml

RUN pip install --root-user-action ignore -e .

RUN mkdir -p /etc/zarr-tools/config \
 && cp configs/* /etc/zarr-tools/config
