#!/bin/bash

# Script to build and optionally push containers using podman
# Usage: ./build-container.sh <subdirectory> [--push]
# Example: ./build-container.sh bigstream/5.0.2-omezarr-dask2025.11.0-py12-ol9 --push

set -e

# Check if subdirectory is provided
if [ -z "$1" ]; then
    echo "Error: Subdirectory path is required"
    echo "Usage: $0 <subdirectory> [--push]"
    echo "Example: $0 bigstream/5.0.2-omezarr-dask2025.11.0-py12-ol9 --push"
    exit 1
fi

SUBDIR="$1"
PUSH_FLAG=""

# Normalize the subdirectory path by removing leading './' if present
SUBDIR="${SUBDIR#./}"

# Check for --push flag
if [ "$2" == "--push" ]; then
    PUSH_FLAG="--push"
    echo "Will push container after building"
fi

# Extract image name and tag from subdirectory path
# Format: name/tag
IMAGE_NAME=$(echo "$SUBDIR" | cut -d'/' -f1)
TAG=$(echo "$SUBDIR" | cut -d'/' -f2)
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "=========================================="
echo "Building container: $FULL_IMAGE"
echo "Subdirectory: $SUBDIR"
echo "=========================================="
echo

# Step 1: Check for existing image
echo "Checking for existing image: $FULL_IMAGE"
if podman images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${FULL_IMAGE}$"; then
    echo "Found existing image: $FULL_IMAGE"
    echo "Removing image..."
    podman rmi "$FULL_IMAGE"
    echo "Image removed successfully"
fi

# Step 2: Check for existing manifest
echo "Checking for existing manifest: $FULL_IMAGE"
if podman manifest inspect "$FULL_IMAGE" >/dev/null 2>&1; then
    echo "Found existing manifest: $FULL_IMAGE"
    echo "Removing manifest..."
    podman manifest rm "$FULL_IMAGE"
    echo "Manifest removed successfully"
fi

# Step 3: Prune unused images
echo
echo "Pruning unused images..."
podman image prune -f
echo "Prune complete"
echo

# Step 4: Build the container
echo "=========================================="
echo "Building container from $SUBDIR"
echo "=========================================="
./build.sh "$SUBDIR" --with-podman $PUSH_FLAG

echo
echo "=========================================="
echo "Build complete: $FULL_IMAGE"
echo "=========================================="
