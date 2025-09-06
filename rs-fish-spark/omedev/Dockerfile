ARG SPARK_VERSION=3.3.2-scala2.12-java17-ubuntu24.04
ARG RS_FISH_REPO=https://github.com/cgoina/RS-FISH-Spark.git
ARG RS_FISH_BRANCH=ome
ARG RS_FISH_SPARK_GIT_HASH=991935b

# Staged build using builder container with java8
# because Bigstitcher still needs java8 to build
FROM azul/zulu-openjdk:8-latest AS builder
ARG RS_FISH_REPO
ARG RS_FISH_BRANCH
ARG RS_FISH_SPARK_GIT_HASH

RUN apt-get update -y \
 && apt-get install -y git curl maven

RUN curl -sSL https://bit.ly/install-xq | bash;

# Checkout and build the code
WORKDIR /tmp/app
RUN git clone --branch ${RS_FISH_BRANCH} ${RS_FISH_REPO} . \
 && git reset --hard ${RS_FISH_SPARK_GIT_HASH}

RUN mvn package -DskipTests

# Find the built jar, based on the version in the pom file
RUN xq -x '/project/artifactId' pom.xml > target-name \
 && xq -x '/project/version' pom.xml > target-version \
 && echo "Rename target/$(cat target-name)-$(cat target-version)-with-dependencies.jar" \
 && mv target/$(cat target-name)-$(cat target-version)-with-dependencies.jar app.jar

RUN echo "${RS_FISH_BRANCH}:${RS_FISH_SPARK_GIT_HASH}" > /tmp/app/VERSION

FROM ghcr.io/janeliascicomp/spark:${SPARK_VERSION}
ARG RS_FISH_SPARK_GIT_HASH

LABEL \
    org.opencontainers.image.title="RS-FISH Spark" \
    org.opencontainers.image.description="Spark version of RS-FISH" \
    org.opencontainers.image.authors="rokickik@janelia.hhmi.org,preibischs@janelia.hhmi.org,goinac@janelia.hhmi.org" \
    org.opencontainers.image.licenses="GPL-2.0" \
    org.opencontainers.image.version=${RS_FISH_SPARK_GIT_HASH}

USER root

RUN apt update -y; \
    apt-get install -y \
        libblosc1 libblosc-dev \
        libzstd1 libzstd-dev libhdf5-dev;

WORKDIR /app
COPY --from=builder /tmp/app/LICENSE.txt /app/LICENSE
COPY --from=builder /tmp/app/VERSION /app
COPY --from=builder /tmp/app/app.jar /app/app.jar
