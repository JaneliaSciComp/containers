# Create final image
FROM  ghcr.io/janeliascicomp/dask:2025.5.1-py12-ol9
ARG TARGETPLATFORM
ARG GIT_BRANCH=dev
ARG GIT_COMMIT=982c790

RUN dnf group install -y "Development Tools"

COPY conda-env.yml /tmp/

RUN mamba env update -n base -f /tmp/conda-env.yml

WORKDIR /opt/scripts/zarr-tools

RUN mkdir -p /etc/zarr-tools/config

COPY configs/* /etc/zarr-tools/config/
COPY scripts/python /opt/scripts/zarr-tools

RUN pip install --root-user-action ignore .
