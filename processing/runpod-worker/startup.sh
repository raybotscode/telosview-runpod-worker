#!/bin/bash
# TelosView RunPod Worker Startup Script
# Downloads and runs the worker server on a RunPod GPU pod

echo "[startup] Installing Node.js 22..."
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null 2>&1
apt-get install -y nodejs git > /dev/null 2>&1 || echo "[startup] Node.js install had issues"

echo "[startup] Installing Chromium + Vulkan..."
apt-get update > /dev/null 2>&1
apt-get install -y chromium-browser libvulkan1 mesa-vulkan-drivers fonts-liberation > /dev/null 2>&1 || echo "[startup] Chromium install had issues"

echo "[startup] Cloning repo..."
cd /tmp
git clone --depth 1 https://github.com/raybotscode/telosview-runpod-worker.git telosview-repo > /dev/null 2>&1
if [ ! -d telosview-repo ]; then
  echo "[startup] Clone failed!"
  exit 1
fi

echo "[startup] Setting up worker directory..."
mkdir -p /app/frames /app/output /app/src

# Copy worker files
cp /tmp/telosview-repo/processing/runpod-worker/server.mjs /app/server.mjs
cp /tmp/telosview-repo/processing/runpod-worker/package.json /app/package.json
cp /tmp/telosview-repo/processing/worker.html /app/worker.html

# Copy splat.js source (entire directory tree)
cp -r /tmp/telosview-repo/splat-test/src/* /app/src/

# Cleanup
rm -rf /tmp/telosview-repo

# Install npm deps
echo "[startup] Installing npm dependencies..."
cd /app
npm install --ignore-scripts > /dev/null 2>&1 || echo "[startup] npm install had issues"

# Install Playwright browsers (Chromium)
echo "[startup] Installing Playwright Chromium..."
npx playwright install chromium > /dev/null 2>&1 || echo "[startup] Playwright install had issues"

echo "[startup] Starting worker server..."
cd /app
exec node server.mjs
