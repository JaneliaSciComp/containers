ARG SPARK_VERSION=3.1.3
ARG N5_SPARK_GIT_HASH=9097071

# Temporary container for building cleanly
FROM janeliascicomp/builder:1.2.1 as builder
ARG N5_SPARK_GIT_HASH

# Checkout and build the code
WORKDIR /tmp/app
RUN git clone https://github.com/saalfeldlab/n5-spark . \
    && git reset --hard $N5_SPARK_GIT_HASH \
    && /usr/local/bin/buildinfo.sh \
    && mvn package -P fatjar,spark-local

# Find the built jar, based on the version in the pom file
RUN xq -r '.project.artifactId+"-"+.project.version+".jar"' pom.xml > filename \
    && mv /tmp/app/target/`cat filename` app.jar

RUN echo "$N5_SPARK_GIT_HASH" > /tmp/app/VERSION

# Create final image
FROM ghcr.io/janeliascicomp/spark:${SPARK_VERSION}

LABEL software="n5-spark" \
    software.version=${N5_SPARK_GIT_HASH} \
    base_image="apache/spark" \
    about.summary="Spark-driven processing utilities for N5 datasets" \
    about.home="https://github.com/saalfeldlab/stitching-spark" \
    about.license="SPDX:GPL-2.0" \
    about.license_file="/app/LICENSE.txt" \
    about.tags="implemented-in::java, interface::commandline, role::program" \
    maintainer="rokickik@janelia.hhmi.org" \
    org.opencontainers.image.source="https://github.com/JaneliaSciComp/containers"

COPY --from=builder /tmp/app/LICENSE.txt /app/LICENSE
COPY --from=builder /tmp/app/VERSION /app
COPY --from=builder /tmp/app/app.jar /app
COPY --from=builder /buildinfo /app
