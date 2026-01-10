#!/bin/bash

# Script to clear container image and manifest
# Usage: ./clear-container-image.sh <subdirectory>
# Example: ./clear-container-image.sh bigstream/5.0.2-omezarr-dask2025.11.0-py12-ol9

set -e

# Check if subdirectory is provided
if [ -z "$1" ]; then
    echo "Error: Subdirectory path is required"
    echo "Usage: $0 <subdirectory>"
    echo "Example: $0 bigstream/5.0.2-omezarr-dask2025.11.0-py12-ol9"
    exit 1
fi

SUBDIR="$1"

# Normalize the subdirectory path by removing leading './' if present
SUBDIR="${SUBDIR#./}"

# Extract image name and tag from subdirectory path
# Format: name/tag
IMAGE_NAME=$(echo "$SUBDIR" | cut -d'/' -f1)
TAG=$(echo "$SUBDIR" | cut -d'/' -f2)
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "=========================================="
echo "Clear container image: $FULL_IMAGE"
echo "=========================================="
echo

# Step 1: Check for existing image
echo "Checking for existing image: $FULL_IMAGE"
if podman images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${FULL_IMAGE}$"; then
    echo "Found existing image: $FULL_IMAGE"
    echo "Removing image..."
    podman rmi "$FULL_IMAGE" -f
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
