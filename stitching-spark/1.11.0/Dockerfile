ARG SPARK_VERSION=3.1.3
ARG STITCHING_SPARK_GIT_TAG=1.11.0-alpha

# Temporary container for building cleanly
FROM azul/zulu-openjdk-debian:8u412-jdk as builder
ARG STITCHING_SPARK_GIT_TAG
ARG TARGETPLATFORM

RUN apt update \
 && apt install -y git curl \
 && curl -sSL https://bit.ly/install-xq | bash

# Checkout and build the code
WORKDIR /tmp/app
RUN git clone --branch ${STITCHING_SPARK_GIT_TAG} --depth 1 https://github.com/JaneliaScicomp/stitching-spark.git . \
 && git submodule update --init --recursive \
 && ./mvnw clean package -DskipTests

# Find the built jar, based on the version in the pom file
RUN xq -x '/project/artifactId' pom.xml > target-name \
 && xq -x '/project/version' pom.xml > target-version \
 && mv target/$(cat target-name)-$(cat target-version).jar app.jar

RUN echo "$STITCHING_SPARK_GIT_TAG" > /tmp/app/VERSION

# Create final image
FROM ghcr.io/janeliascicomp/spark:${SPARK_VERSION}
ARG STITCHING_SPARK_GIT_TAG
ARG TARGETPLATFORM

LABEL \
    org.opencontainers.image.title="Stitching Spark" \
    org.opencontainers.image.description="Reconstructing large microscopy images from overlapping image tiles on a high-performance Spark cluster" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org" \
    org.opencontainers.image.licenses="BSD-3-Clause AND GPL-2.0" \
    org.opencontainers.image.version=${STITCHING_SPARK_GIT_TAG}

WORKDIR /app
COPY --from=builder /tmp/app/LICENSE.txt /app/LICENSE
COPY --from=builder /tmp/app/VERSION /app
COPY --from=builder /tmp/app/app.jar /app
