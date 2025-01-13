# Create final image
FROM  ghcr.io/janeliascicomp/dask:2024.4.1-py11-ol9
ARG TARGETPLATFORM
ARG GIT_BRANCH=dev
ARG GIT_COMMIT=982c790

RUN dnf install -y git

WORKDIR /tmp-build-dir

RUN git clone --branch ${GIT_BRANCH} --depth 1 https://github.com/JaneliaSciComp/n5-tools-dask.git . \
 && git reset --hard ${GIT_COMMIT} \
 && mamba env update -n base -f conda-env.yml

WORKDIR /opt/scripts/n5-tools
RUN cp /tmp-build-dir/src/* /opt/scripts/n5-tools

RUN echo "Remote tmp build dir" && rm -rf /tmp-build-dir
