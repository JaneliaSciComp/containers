FROM ghcr.io/janeliascicomp/dask:2025.11.0-py12-ol9
ARG SPOTS_TOOLS_BRANCH=main
ARG SPOTS_TOOLS_COMMIT=42a1ae4

RUN dnf install -y \
    tar git

WORKDIR /opt/scripts/spots-utils

RUN git clone --branch ${SPOTS_TOOLS_BRANCH} --depth 1 --progress https://github.com/JaneliaSciComp/easifish-spots-tools.git . \
 && git reset --hard ${SPOTS_TOOLS_COMMIT}

ENV MKL_NUM_THREADS=
ENV NUM_MKL_THREADS=
ENV OPENBLAS_NUM_THREADS=
ENV OPENMP_NUM_THREADS=
ENV OMP_NUM_THREADS=

ENV PIP_ROOT_USER_ACTION=ignore

# Use the base environment from the baseImage
RUN mamba env update -n base -f conda-env.yml
RUN echo ${SPOTS_TOOLS_COMMIT} > .commit

RUN pip install --root-user-action ignore -e .

# Remove dev tools
RUN dnf clean all \
 && rm -rf /var/cache/dnf
