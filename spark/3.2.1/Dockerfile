ARG SPARK_VERSION=3.2.1

FROM apache/spark:v${SPARK_VERSION}
ARG SPARK_VERSION

# Reset user to root so that we can install missing packages
# and the image works in Singularity
USER root
RUN apt update -y \
 && apt-get install -y net-tools curl

LABEL \
    org.opencontainers.image.title="Apache Spark" \
    org.opencontainers.image.description="Apache Spark is a unified analytics engine for large-scale data processing. This container combines the Spark runtime with additional scripts for simplified creation of Spark clusters." \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.version=${SPARK_VERSION}

# Save version inside container
RUN echo "${SPARK_VERSION}" > /opt/spark/VERSION

# Add scripts for running pipelines
COPY ./scripts /opt/scripts
