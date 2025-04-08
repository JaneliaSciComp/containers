FROM azul/zulu-openjdk:8-latest

ARG SPARK_VERSION=3.3.2
ARG HADOOP_VERSION=3
ARG SCALA_VERSION=2.12


# Reset user to root so that we can install missing packages
# and the image works in Singularity
USER root

RUN apt update -y \
 && apt-get install -y \
        net-tools wget curl tar \
        hostname python3 python3-pip

WORKDIR /tmp

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
 && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
 && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
 && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

WORKDIR /opt

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
