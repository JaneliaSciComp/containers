#!/bin/bash

set -euo pipefail

function build() {
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

args=()
platform_arg=
push_container_arg=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      args+=("--platform")
      shift
      platform_arg="$1"
      args+=(${platform_arg})
      ;;
    --push|--load)
      args+=("$1")
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

if [[ -z "${platform_arg}" ]]; then
    args+=("--load")
fi

if [[ -z "${push_container_arg}" ]]; then
    args+=("--push")
fi

build "${args[@]}"
