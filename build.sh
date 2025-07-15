#!/bin/bash

set -euo pipefail

function build_docker() {
    local path="$1"
    shift
    local container=$(echo $path | cut -d '/' -f 1)
    local version=$(echo $path | cut -d '/' -f 2)

    local now="$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"
    local repo_url="https://github.com/JaneliaSciComp/containers"
    local tag="ghcr.io/janeliascicomp/$container:$version"

    local CMD=(docker buildx build
        --label org.opencontainers.image.source="$repo_url"
        --label org.opencontainers.image.created="$now"
        --tag ${tag}        
    )
    CMD+=("$@")
    CMD+=(./${container}/${version})

    echo "${CMD[@]}"
    exec "${CMD[@]}"
}

function build_podman() {
    local push_arg=$1
    shift
    local path="$1"
    shift
    local container=$(echo $path | cut -d '/' -f 1)
    local version=$(echo $path | cut -d '/' -f 2)

    local now="$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"
    local repo_url="https://github.com/JaneliaSciComp/containers"
    local tag="ghcr.io/janeliascicomp/$container:$version"

    echo "Create manifest: $tag"
    podman manifest create $tag

    local BUILD_CMD=(
	podman build
        --label org.opencontainers.image.source="$repo_url"
        --label org.opencontainers.image.created="$now"
    )

    BUILD_CMD+=("$@")
    BUILD_CMD+=(
        --manifest ${tag}
	./${container}/${version}
    )

    echo "Build image: ${BUILD_CMD[@]}"
    ${BUILD_CMD[@]}

    if [[ "${push_arg}" == "--push" ]]; then
	echo "Push container $tag"
	podman manifest push $tag
    fi
}

args=()
tool=docker
platform_arg=
push_container_arg=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-podman)
      tool=podman
      ;;
    --platform)
      args+=("--platform")
      shift
      platform_arg="$1"
      args+=(${platform_arg})
      ;;
    --push|--load)
      push_container_arg="$1"
      ;;
    *)
      args+=("$1")
      ;;
  esac
  shift
done

if [[ -z "${platform_arg}" ]]; then
    # Default platform if not specified
    echo "Build container for linux/amd64,linux/arm64"
    args+=("--platform" "linux/amd64,linux/arm64")
else
    echo "Build container for ${platform_arg}"
fi

case $tool in
    docker)
	if [[ -z "${push_container_arg}" ]]; then
	    args+=("--load")
	else
	    args+=(${push_container_arg})
	fi
	build_docker "${args[@]}"
	;;
    podman)
	if [[ -z "${push_container_arg}" ]]; then
	    push_container_arg="--load"
	fi
	build_podman ${push_container_arg} "${args[@]}"
	;;
esac
