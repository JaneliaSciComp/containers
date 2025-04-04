ARG SPARK_VERSION=3.1.3
ARG BIGSTITCHER_SPARK_GIT_HASH=d63678a

# Staged build using builder container
FROM azul/zulu-openjdk:8-latest AS builder
ARG BIGSTITCHER_SPARK_GIT_HASH

RUN apt update -y \
 && apt-get install -y \
        git \
        maven

# Checkout and build the code
WORKDIR /tmp/app
RUN git clone https://github.com/PreibischLab/BigStitcher-Spark.git . \
    && git reset --hard ${BIGSTITCHER_SPARK_GIT_HASH} \
    && mvn package -P fatjar -DskipTests

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
COPY --from=builder /tmp/app/target/BigStitcher-Spark-0.1.0-SNAPSHOT.jar /app/app.jar

RUN echo "#!/bin/bash" >> /entrypoint.sh \
    && echo 'java -cp /app/app.jar -Dspark.master=local[8] "$@"' >> /entrypoint.sh \
    && chmod +x /entrypoint.sh
