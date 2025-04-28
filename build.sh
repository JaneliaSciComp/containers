#!/bin/bash

set -euo pipefail

function build() {
    local path="$1"
    shift
    container=$(echo $path | cut -d '/' -f 1)
    version=$(echo $path | cut -d '/' -f 2)

    now="$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"
    repo_url="https://github.com/JaneliaSciComp/containers"
    tag="ghcr.io/janeliascicomp/$container:$version"

    CMD=(docker buildx build
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

build "${args[@]}"
