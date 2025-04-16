ARG SPARK_VERSION=3.1.3
ARG RSFISH_GIT_HASH=4d1b901

# Staged build using builder container
FROM azul/zulu-openjdk-debian:8u412-jdk AS builder
ARG SPARK_VERSION

USER root
RUN apt update -y \
 && apt-get install -y \
        git maven 

# Checkout and build the code
WORKDIR /tmp/app
RUN git clone https://github.com/PreibischLab/RS-FISH-Spark.git . \
 && git reset --hard $RSFISH_GIT_HASH \
 && mvn package

RUN echo "$RSFISH_GIT_HASH" > /tmp/app/VERSION

# Create final image
FROM ghcr.io/janeliascicomp/spark:${SPARK_VERSION}
ARG RSFISH_GIT_HASH

LABEL \
    org.opencontainers.image.title="RS-FISH Spark" \
    org.opencontainers.image.description="Spark version of RS-FISH for spot detection" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org" \
    org.opencontainers.image.licenses="BSD-3-Clause" \
    org.opencontainers.image.version=${RSFISH_GIT_HASH}

WORKDIR /app
COPY --from=builder /tmp/app/VERSION /app
COPY --from=builder /tmp/app/target/RS-Fish-jar-with-dependencies.jar /app/app.jar
COPY --from=builder /tmp/app/LICENSE.txt /app

RUN echo "#!/bin/bash" >> /entrypoint.sh \
    && echo 'java -cp /app/app.jar -Xmx20G -Dspark.master=local[8] net.preibisch.rsfish.spark.SparkRSFISH "$@"' >> /entrypoint.sh \
    && chmod +x /entrypoint.sh
