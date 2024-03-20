# Create final image
FROM ghcr.io/janeliascicomp/dask:2023.10.1-py11-ol9
ARG TARGETPLATFORM

RUN dnf install -y \
        git \
        mesa-libGL

WORKDIR /app/bigstream

# install bigstream
ENV BIGSTREAM_TAG=goinac-dev
RUN git clone --branch $BIGSTREAM_TAG --depth 1 https://github.com/JaneliaSciComp/bigstream.git .

# Remove the data and the git repo to reduce space
RUN echo "Remove test data from the container" && rm -rf resources

# Use the base environment from the baseImage
RUN mamba env update -n base -f conda-env.yaml

RUN pip install .
