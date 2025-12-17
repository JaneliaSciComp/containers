FROM oraclelinux:9
ARG SPOTS_TOOLS_BRANCH=main
ARG SPOTS_TOOLS_COMMIT=5365b0a

RUN dnf install -y \
    tar wget git

# Install miniconda
RUN wget https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-25.11.0-1-Linux-$(uname -m).sh \
    -O mamba-install.sh && \
    bash mamba-install.sh -b -p /opt/mambaforge

ENV PATH=/opt/mambaforge/bin:${PATH}

WORKDIR /opt/scripts/spots-utils

RUN git clone --branch ${SPOTS_TOOLS_BRANCH} --depth 1 --progress https://github.com/JaneliaSciComp/easifish-spots-tools.git . \
 && git reset --hard ${SPOTS_TOOLS_COMMIT}

 ENV PIP_ROOT_USER_ACTION=ignore

RUN mamba env update -n base -f conda-env.yml \
 && pip install --root-user-action ignore -e .

RUN mamba clean -a -y \
 && rm -rf ~/.cache \
 && dnf clean all
