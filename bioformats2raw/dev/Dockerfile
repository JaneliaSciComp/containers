FROM openjdk:8
ARG COMMIT_HASH=ccfaee7

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update -y -q \
 && apt-get install -y --no-install-recommends -q libblosc1 \
 && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER root

WORKDIR /opt
RUN git clone https://github.com/glencoesoftware/bioformats2raw.git \
    && cd bioformats2raw \
    && git checkout ${COMMIT_HASH} \
    && ./gradlew installDist \
    && cd .. \
    && ln -s /opt/bioformats2raw/build/install/bioformats2raw /opt/bioformats2raw-${COMMIT_HASH}
RUN echo "$COMMIT_HASH" > /opt/VERSION \
    && cp /opt/bioformats2raw/LICENSE.txt /opt/LICENSE

ENV PATH="/opt/bioformats2raw/bin:${PATH}"

LABEL \
    org.opencontainers.image.title="Bioformats2raw" \
    org.opencontainers.image.description="Convert any image readable by Bioformats into an OME-NGFF compatible format" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org" \
    org.opencontainers.image.licenses="GPL-2.0" \
    org.opencontainers.image.version=${COMMIT_HASH}

USER 1000
WORKDIR /opt/bioformats2raw
ENTRYPOINT ["/opt/bioformats2raw/bin/bioformats2raw"]
