FROM ghcr.io/janeliascicomp/zarr-tools:dask2025.5.1-py12-ol9

LABEL software="cellpose"
LABEL version="4"
LABEL base_image="janeliascicomp/dask:2025.5.2-py12-ol9"
LABEL software.version="0.5.0-dask2025.5.1-py12"
LABEL about.summary="A distributed algorithm for spots extraction."
LABEL about.home="https://github.com/GFleishman/fishspot"
LABEL about.license="MIT"
LABEL about.license_file="https://github.com/GFleishman/fishspot/blob/main/LICENSE"
LABEL extra.binaries="/opt/mambaforge/bin"
LABEL extra.scripts="/opt/scripts/fishspots"
LABEL about.tags="implemented-in::python"
LABEL conda_forge.minforge.version="25.3.0-3"
LABEL python.version="3.12"
LABEL extra.identifiers.biotools=fishspots
LABEL bioimagetools.image.authors="fleishmang@janelia.hhmi.org,goinac@janelia.hhmi.org"

RUN dnf install -y git

COPY conda-env.yml /tmp/

RUN mamba env update -n base -f /tmp/conda-env.yml \
 && rm -rf /opt/mambaforge/pkgs

WORKDIR /opt/scripts/fishspots

RUN mkdir -p /etc/fishspots/config

COPY configs/* /etc/fishspots/config
COPY scripts/python /opt/scripts/fishspots

RUN pip install --no-cache-dir --root-user-action ignore -e . \
 && mamba clean --all -y
