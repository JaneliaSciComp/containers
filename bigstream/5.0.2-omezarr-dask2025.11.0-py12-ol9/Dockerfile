# Create final image
FROM ghcr.io/janeliascicomp/zarr-tools:0.3.0-dask2025.11.0-py12-ol9
ARG BIGSTREAM_BRANCH=ome-dev
ARG BIGSTREAM_COMMIT=5a626f9

RUN dnf install -y \
        git \
        mesa-libGL

WORKDIR /app/bigstream

# install bigstream from the specified commit
RUN git clone --branch ${BIGSTREAM_BRANCH} --depth 1 --progress https://github.com/JaneliaSciComp/bigstream.git . \
 && git reset --hard ${BIGSTREAM_COMMIT}

# Remove the data and the git repo to reduce space
RUN echo "Remove test data from the container" && rm -rf resources

ENV ITK_THREADS=
ENV MKL_NUM_THREADS=
ENV NUM_MKL_THREADS=
ENV OPENBLAS_NUM_THREADS=
ENV OPENMP_NUM_THREADS=
ENV OMP_NUM_THREADS=

ENV PIP_ROOT_USER_ACTION=ignore

ENV PYTHONPATH=/app/bigstream

# Use the base environment from the baseImage
RUN mamba env update -n base -f conda-env.yaml
RUN echo ${BIGSTREAM_COMMIT} > .commit

RUN pip install --root-user-action ignore -e . \
 && pip install "zarr-tools @ git+https://github.com/JaneliaSciComp/zarr-tools.git@16271a6"

# Remove dev tools
RUN dnf clean all \
 && rm -rf /var/cache/dnf
