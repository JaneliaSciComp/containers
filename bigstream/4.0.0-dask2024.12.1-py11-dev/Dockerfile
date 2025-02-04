# Create final image
FROM ghcr.io/janeliascicomp/dask:2024.12.1-py11-ol9
ARG TARGETPLATFORM
ARG BIGSTREAM_BRANCH=goinac-dev
ARG BIGSTREAM_COMMIT=e608ace

RUN dnf install -y \
        git \
        mesa-libGL

WORKDIR /app/bigstream

# install bigstream from the specified commit
RUN git clone --branch ${BIGSTREAM_BRANCH} --depth 1 https://github.com/JaneliaSciComp/bigstream.git . \
 && git reset --hard ${BIGSTREAM_COMMIT}

# Remove the data and the git repo to reduce space
RUN echo "Remove test data from the container" && rm -rf resources

ENV ITK_THREADS=
ENV MKL_NUM_THREADS=
ENV NUM_MKL_THREADS=
ENV OPENBLAS_NUM_THREADS=
ENV OPENMP_NUM_THREADS=
ENV OMP_NUM_THREADS=

ENV PYTHONPATH=/app/bigstream

# Use the base environment from the baseImage
RUN mamba env update -n base -f conda-env.yaml
RUN echo ${BIGSTREAM_COMMIT} > .commit

RUN pip install .
