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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up app directory
WORKDIR /app

# Install npm dependencies first (layer caching)
COPY package.json ./
RUN npm install

# Install Playwright Chromium + its system deps
RUN npx playwright install-deps chromium && npx playwright install chromium

# Tell Playwright to prefer its bundled Chromium (has proper GPU support)
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=""

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

# Start the worker server
CMD ["node", "server.mjs"]
