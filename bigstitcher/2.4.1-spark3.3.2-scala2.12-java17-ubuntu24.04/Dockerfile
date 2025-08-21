ARG SPARK_VERSION=3.3.2-scala2.12-java17-ubuntu24.04
ARG BIGSTITCHER_BRANCH=zarr-group-arg
ARG BIGSTITCHER_SPARK_GIT_HASH=0d7cf0e

# Staged build using builder container with java8
# because Bigstitcher still needs java8 to build
FROM azul/zulu-openjdk:8-latest AS builder
ARG BIGSTITCHER_BRANCH
ARG BIGSTITCHER_SPARK_GIT_HASH

RUN apt-get update -y; \
    apt-get install -y git curl maven; \
    curl -sSL https://bit.ly/install-xq | bash;

# Checkout and build the code
WORKDIR /tmp/app
RUN git clone --branch ${BIGSTITCHER_BRANCH} https://github.com/JaneliaSciComp/BigStitcher-Spark.git . \
    && git reset --hard ${BIGSTITCHER_SPARK_GIT_HASH} \
    && mvn package -P fatjar -DskipTests

# Find the built jar, based on the version in the pom file
RUN xq -x '/project/artifactId' pom.xml > target-name \
 && xq -x '/project/version' pom.xml > target-version \
 && echo "Rename target/$(cat target-name)-$(cat target-version).jar" \
 && mv target/$(cat target-name)-$(cat target-version).jar app.jar

RUN echo "${BIGSTITCHER_BRANCH}:${BIGSTITCHER_SPARK_GIT_HASH}" > /tmp/app/VERSION

# Create final image
FROM ghcr.io/janeliascicomp/spark:${SPARK_VERSION}
ARG BIGSTITCHER_SPARK_GIT_HASH

LABEL \
    org.opencontainers.image.title="BigStitcher Spark" \
    org.opencontainers.image.description="Spark version of BigStitcher" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org,preibischs@janelia.hhmi.org,goinac@janelia.hhmi.org" \
    org.opencontainers.image.licenses="BSD-3-Clause" \
    org.opencontainers.image.version=${BIGSTITCHER_SPARK_GIT_HASH}

USER root

RUN apt-get update -y; \
    apt-get install -y \
        libblosc1 libblosc-dev \
        libzstd1 libzstd-dev libhdf5-dev;

WORKDIR /app
COPY --from=builder /tmp/app/LICENSE /app/LICENSE
COPY --from=builder /tmp/app/VERSION /app
COPY --from=builder /tmp/app/app.jar /app/app.jar
