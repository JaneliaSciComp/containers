ARG SPARK_VERSION=3.1.3
ARG BIGSTITCHER_SPARK_GIT_HASH=d63678a

# Staged build using builder container
FROM azul/zulu-openjdk:8-latest AS builder
ARG BIGSTITCHER_SPARK_GIT_HASH

RUN apt update -y \
 && apt-get install -y \
        git curl maven \
 && curl -sSL https://bit.ly/install-xq | bash


# Checkout and build the code
WORKDIR /tmp/app
RUN git clone https://github.com/PreibischLab/BigStitcher-Spark.git . \
    && git reset --hard ${BIGSTITCHER_SPARK_GIT_HASH} \
    && mvn package -P fatjar -DskipTests

# Find the built jar, based on the version in the pom file
RUN xq -x '/project/artifactId' pom.xml > target-name \
 && xq -x '/project/version' pom.xml > target-version \
 && mv target/$(cat target-name)-$(cat target-version).jar app.jar

RUN echo "${BIGSTITCHER_SPARK_GIT_HASH}" > /tmp/app/VERSION

# Create final image
FROM ghcr.io/janeliascicomp/spark:${SPARK_VERSION}
ARG TARGETPLATFORM
ARG BIGSTITCHER_SPARK_GIT_HASH

LABEL \
    org.opencontainers.image.title="BigStitcher Spark" \
    org.opencontainers.image.description="Spark version of RS-FISH for spot detection" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org,preibischs@janelia.hhmi.org,goinac@janelia.hhmi.org" \
    org.opencontainers.image.licenses="BSD-3-Clause" \
    org.opencontainers.image.version=${BIGSTITCHER_SPARK_GIT_HASH}

RUN apt update -y \
 && apt-get install -y \
        git \
        libblosc1 libblosc-dev \
        g++ cmake

WORKDIR /tmp/c-blosc

RUN git clone --branch main --depth 1 https://github.com/Blosc/c-blosc.git . \
 && mkdir build \ 
    && cd build \
    && cmake -S .. -B . \
    && make \
    && make install

WORKDIR /app
COPY --from=builder /tmp/app/LICENSE /app/LICENSE
COPY --from=builder /tmp/app/VERSION /app
COPY --from=builder /tmp/app/app.jar /app/app.jar
