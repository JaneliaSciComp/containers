# Create final image
FROM ghcr.io/janeliascicomp/dask:2025.11.0-py12-ol9
ARG BIGSTREAM_BRANCH=ome-dev
ARG BIGSTREAM_COMMIT=5c34b3e

LABEL software="BigStream"
LABEL version="5"
LABEL base_image="janeliascicomp/dask:2025.11.0-py12-ol9"
LABEL software.version="5.1.1-omezarr-dask2025.11.0-py12-ol9"
LABEL about.summary="A wrapper of several SimpleITK algorithms for running blockwise image registration."
LABEL about.home="https://github.com/JaneliaSciComp/bigstream"
LABEL about.license="BSD-3-Clause"
LABEL about.tags="implemented-in::python"
LABEL conda_forge.minforge.version="25.11.0-1"
LABEL python.version="3.12"
LABEL extra.identifiers.biotools=SimpleITK,BigStream
LABEL bioimagetools.image.authors="fleishmang@janelia.hhmi.org,goinac@janelia.hhmi.org,rokickik@janelia.hhmi.org"

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

# Use the base environment from the baseImage
RUN mamba env update -n base -f conda-env.yaml
RUN echo ${BIGSTREAM_COMMIT} > .commit

RUN pip install --root-user-action ignore -e . \
 && pip install "zarr-tools @ git+https://github.com/JaneliaSciComp/zarr-tools.git@16271a6"

# Remove dev tools
RUN dnf clean all \
 && rm -rf /var/cache/dnf
