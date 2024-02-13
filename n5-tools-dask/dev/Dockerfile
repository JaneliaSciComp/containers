# Create final image
FROM  ghcr.io/janeliascicomp/dask:2023.10.1-py11-ol9

ARG TARGETPLATFORM
ARG GIT_TAG=dev

RUN dnf install -y git

WORKDIR /tmp-build-dir
RUN git clone --branch ${GIT_TAG} https://github.com/JaneliaSciComp/n5-tools-dask.git . \
 && mamba env update -n base -f environment.yml


WORKDIR /opt/scripts/n5-tools-dask
RUN cp /tmp-build-dir/src/* /opt/scripts/n5-tools-dask \
 && rm -rf /tmp-build-dir
