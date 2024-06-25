ARG SPARK_VERSION=3.1.3
ARG STITCHING_SPARK_GIT_TAG=1.10.0

# Temporary container for building cleanly
FROM janeliascicomp/builder:1.2.1 AS builder
ARG STITCHING_SPARK_GIT_TAG

# Checkout and build the code
WORKDIR /tmp/app
# TODO: currently this points to a fork which adds a minor feature. It can be switched back to https://github.com/saalfeldlab/stitching-spark after the PR is merged.
RUN git clone --branch ${STITCHING_SPARK_GIT_TAG} --depth 1 https://github.com/krokicki/stitching-spark.git . \
    && git submodule update --init --recursive \
    && /usr/local/bin/buildinfo.sh \
    && mvn package

# Find the built jar, based on the version in the pom file
RUN xq -r '.project.artifactId+"-"+.project.version+".jar"' pom.xml > filename \
    && mv /tmp/app/target/`cat filename` app.jar

RUN echo "$STITCHING_SPARK_GIT_TAG" > /tmp/app/VERSION

# Create final image
FROM ghcr.io/janeliascicomp/spark:${SPARK_VERSION}
ARG STITCHING_SPARK_GIT_TAG

LABEL software="stitching-spark" \
    software.version=${STITCHING_SPARK_GIT_TAG} \
    base_image="apache/spark" \
    about.summary="Reconstructing large microscopy images from overlapping image tiles on a high-performance Spark cluster" \
    about.home="https://github.com/saalfeldlab/stitching-spark" \
    about.license="SPDX:GPL-2.0" \
    about.license_file="/app/LICENSE.txt" \
    about.tags="implemented-in::java, interface::commandline, role::program" \
    maintainer="rokickik@janelia.hhmi.org"

WORKDIR /app
COPY --from=builder /tmp/app/LICENSE.txt /app/LICENSE
COPY --from=builder /tmp/app/VERSION /app
COPY --from=builder /tmp/app/app.jar /app
COPY --from=builder /buildinfo /app
