#!/bin/bash
# Build the TelosView RunPod GPU Worker Docker image.
# Assembles the build context from scattered source files.
#
# Usage: ./build.sh [--push]
#   --push  Also push to a registry (requires docker login)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPLAT_SRC="$PROJECT_ROOT/splat-test/src"
WORKER_HTML="$SCRIPT_DIR/../worker.html"
BUILD_CTX="$SCRIPT_DIR/.build-ctx"
IMAGE_NAME="${RUNPOD_DOCKER_IMAGE:-telosview-runpod-worker:latest}"

echo "=== TelosView RunPod Worker Build ==="
echo "Splat.js source: $SPLAT_SRC"
echo "Worker HTML:     $WORKER_HTML"
echo "Image name:      $IMAGE_NAME"
echo ""

# Validate source files exist
if [ ! -d "$SPLAT_SRC" ]; then
  echo "ERROR: splat.js source not found at $SPLAT_SRC"
  exit 1
fi
if [ ! -f "$WORKER_HTML" ]; then
  echo "ERROR: worker.html not found at $WORKER_HTML"
  exit 1
fi

# Clean and create build context
rm -rf "$BUILD_CTX"
mkdir -p "$BUILD_CTX/src"

echo "Assembling build context..."

# Copy Docker build files
cp "$SCRIPT_DIR/Dockerfile" "$BUILD_CTX/"
cp "$SCRIPT_DIR/package.json" "$BUILD_CTX/"
cp "$SCRIPT_DIR/server.mjs" "$BUILD_CTX/"

# Copy worker.html
cp "$WORKER_HTML" "$BUILD_CTX/worker.html"

# Copy splat.js source
cp -r "$SPLAT_SRC"/* "$BUILD_CTX/src/"

echo "Build context assembled ($(du -sh "$BUILD_CTX" | cut -f1))"
echo ""

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$BUILD_CTX"

echo ""
echo "=== Build complete: $IMAGE_NAME ==="
echo ""
echo "To test locally:"
echo "  docker run --gpus all -p 8080:8080 $IMAGE_NAME"
echo ""
echo "To push to Docker Hub:"
echo "  docker push $IMAGE_NAME"
echo ""

# Push if requested
if [ "${1:-}" = "--push" ]; then
  echo "Pushing $IMAGE_NAME..."
  docker push "$IMAGE_NAME"
  echo "Push complete"
fi

# Clean up build context
rm -rf "$BUILD_CTX"
