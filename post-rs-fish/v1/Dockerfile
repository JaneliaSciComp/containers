FROM oraclelinux:9

ARG TARGETPLATFORM

RUN dnf install -y \
    tar wget

# Install miniconda
RUN wget https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-24.11.3-0-Linux-$(uname -m).sh -O mamba-install.sh \
 && bash mamba-install.sh -b -p /opt/mambaforge

ENV PATH=/opt/mambaforge/bin:${PATH}

COPY conda-env.yml /tmp/

RUN mamba env update -n base -f /tmp/conda-env.yml \
 && rm -rf /opt/mambaforge/pkgs

WORKDIR /opt/scripts/post-rs-fish

# Add scripts
COPY scripts/* /opt/scripts/post-rs-fish
