# Create final image
FROM  ghcr.io/janeliascicomp/dask:2025.11.0-py12-ol9
ARG ZARR_TOOLS_BRANCH=main
ARG ZARR_TOOLS_COMMIT=74dce46

LABEL software="zarr-tools"
LABEL version="4"
LABEL base_image="janeliascicomp/dask:2025.11.0-py12-ol9"
LABEL about.tags="implemented-in::python"
LABEL conda_forge.minforge.version="25.11.0-1"
LABEL python.version="3.12"

RUN dnf group install -y "Development Tools"

WORKDIR /opt/scripts/zarr-tools

RUN git clone --branch ${ZARR_TOOLS_BRANCH} --depth 1 --progress https://github.com/JaneliaSciComp/zarr-tools.git . \
 && git reset --hard ${ZARR_TOOLS_COMMIT}

ENV PIP_ROOT_USER_ACTION=ignore

RUN  mamba env update -n base -f conda-env.yml

RUN pip install --root-user-action ignore -e .

RUN mamba clean -a -y

RUN mkdir -p /etc/zarr-tools/config \
 && cp configs/* /etc/zarr-tools/config

# Remove dev tools
RUN dnf group remove -y "Development Tools" \
  && dnf clean all \
  && rm -rf /var/cache/dnf
