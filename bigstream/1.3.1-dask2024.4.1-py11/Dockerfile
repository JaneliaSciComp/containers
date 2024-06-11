# Create final image
FROM ghcr.io/janeliascicomp/dask:2024.4.1-py11-ol9
ARG TARGETPLATFORM

RUN dnf install -y \
        git \
        mesa-libGL

WORKDIR /app/bigstream

# install bigstream from the specified commit
ENV BIGSTREAM_BRANCH=goinac-dev
ENV BIGSTREAM_COMMIT=8902277
RUN git clone --branch ${BIGSTREAM_BRANCH} --depth 1 https://github.com/JaneliaSciComp/bigstream.git . && \
    git reset --hard ${BIGSTREAM_COMMIT}

# Remove the data and the git repo to reduce space
RUN echo "Remove test data from the container" && rm -rf resources

# Use the base environment from the baseImage
RUN mamba env update -n base -f conda-env.yaml

RUN pip install .
