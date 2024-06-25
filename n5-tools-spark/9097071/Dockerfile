ARG SPARK_VERSION=3.1.3
ARG N5_SPARK_GIT_HASH=9097071

# Temporary container for building cleanly
FROM azul/zulu-openjdk-debian:8u412-jdk AS builder
ARG N5_SPARK_GIT_HASH
ARG TARGETPLATFORM

RUN apt update \
 && apt install -y git curl maven \
 && curl -sSL https://bit.ly/install-xq | bash

# Checkout and build the code
WORKDIR /tmp/app
RUN git clone https://github.com/saalfeldlab/n5-spark . \
    && git reset --hard $N5_SPARK_GIT_HASH \
    && mvn package -P fatjar,spark-local -DskipTests

# Find the built jar, based on the version in the pom file
RUN xq -x '/project/artifactId' pom.xml > target-name \
 && xq -x '/project/version' pom.xml > target-version \
 && mv target/$(cat target-name)-$(cat target-version).jar app.jar

RUN echo "$N5_SPARK_GIT_HASH" > /tmp/app/VERSION

# Create final image
FROM ghcr.io/janeliascicomp/spark:${SPARK_VERSION}
ARG N5_SPARK_GIT_HASH
ARG TARGETPLATFORM

LABEL \
    org.opencontainers.image.title="N5 Spark Tools" \
    org.opencontainers.image.description="Spark-driven processing utilities for N5 datasets" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org" \
    org.opencontainers.image.licenses="BSD-3-Clause AND BSD-2-Clause" \
    org.opencontainers.image.version=${N5_SPARK_GIT_HASH}

WORKDIR /app
COPY --from=builder /tmp/app/LICENSE.txt /app/LICENSE
COPY --from=builder /tmp/app/VERSION /app
COPY --from=builder /tmp/app/app.jar /app

