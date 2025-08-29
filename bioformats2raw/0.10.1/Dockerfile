FROM openjdk:8
ARG VERSION=0.10.1

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update -y -q \
 && apt-get install -y --no-install-recommends -q libblosc1 \
 && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /opt
RUN curl -sLO https://github.com/glencoesoftware/bioformats2raw/releases/download/v${VERSION}/bioformats2raw-${VERSION}.zip \
    && unzip bioformats2raw-${VERSION}.zip \
    && ln -s /opt/bioformats2raw-${VERSION} /opt/bioformats2raw
RUN echo "$VERSION" > /opt/VERSION \
    && ln -s /opt/bioformats2raw/LICENSE.txt /opt/LICENSE

ENV PATH="/opt/bioformats2raw/bin:${PATH}"

LABEL \
    org.opencontainers.image.title="Bioformats2raw" \
    org.opencontainers.image.description="Convert any image readable by Bioformats into an OME-NGFF compatible format" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org" \
    org.opencontainers.image.licenses="GPL-2.0" \
    org.opencontainers.image.version=${VERSION}

WORKDIR /opt/bioformats2raw
ENTRYPOINT ["/opt/bioformats2raw/bin/bioformats2raw"]
