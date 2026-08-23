# TelosView RunPod GPU Worker
# Built on RunPod's PyTorch base (pre-cached on their infra — instant pulls)
# Includes: Node.js 22, Chromium with Vulkan, Playwright, splat.js pipeline
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Node.js 22
RUN apt-get update && \
    apt-get install -y curl wget gnupg ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Chromium + Vulkan + Playwright system deps (Ubuntu 22.04 package names)
RUN apt-get update && apt-get install -y \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libxshmfence1 \
    libx11-xcb1 libxcb-dri3-0 libxext6 libxfixes3 \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    fonts-liberation fonts-noto-color-emoji \
    libnspr4 \
    libegl1 libgl1-mesa-glx libgles2 \
    xvfb x11-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade Vulkan loader to 1.4+ via LunarG (Ubuntu 22.04 ships 1.3.204, which
# rejects NVIDIA 570+ drivers that declare Vulkan 1.4). Chromium's Dawn/WebGPU
# uses the system loader, so this is required for WebGPU to see the NVIDIA GPU.
RUN apt-get update && apt-get install -y wget gnupg && \
    wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
    echo "deb https://packages.lunarg.com/vulkan jammy main" > /etc/apt/sources.list.d/lunarg-vulkan.list && \
    apt-get update && apt-get install -y libvulkan1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure NVIDIA Vulkan ICD is registered (RunPod mounts the driver)
# NOTE: RunPod's base image already ships a correct ICD at
# /etc/vulkan/icd.d/nvidia_icd.json (api_version 1.4.312). Do NOT override
# VK_ICD_FILENAMES or create a duplicate ICD — that breaks Vulkan discovery.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Set up app directory
WORKDIR /app

# Install npm dependencies first (layer caching)
COPY package.json ./
RUN npm install

# Install Playwright Chromium + its system deps
# 'npx playwright install chromium' installs BOTH the full Chromium build
# (supports WebGPU) AND chrome-headless-shell (old headless, no WebGPU).
# server.mjs uses channel:'chromium' to select the full build.
RUN npx playwright install-deps chromium && npx playwright install chromium

# Copy worker files
COPY server.mjs ./
COPY worker.html ./
COPY src/ ./src/

# Create runtime directories
RUN mkdir -p /app/frames /app/output

# Expose processing server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the worker server inside a virtual display (Xvfb) — Chromium's WebGPU/Vulkan
# backend needs a DISPLAY to initialize surface/swapchain, even for headless compute.
# dbus must also be up (Chromium GPU process talks to it). Dawn's blocklist (which
# rejects NVIDIA 570+) is disabled via flags in server.mjs.
CMD ["/bin/bash", "-c", "/etc/init.d/dbus start 2>/dev/null; Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp & export DISPLAY=:99 && node server.mjs"]
