# RunPod GPU Processing Image for TelosView v2
# Runs splat.js Gaussian splatting with real CUDA/WebGPU (no SwiftShader)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Node.js 22
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Chromium dependencies (for headless WebGPU with real GPU)
RUN apt-get update && apt-get install -y \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libxshmfence1 \
    libx11-xcb1 libxcb-dri3-0 libxext6 libxfixes3 \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    fonts-liberation fonts-noto-color-emoji \
    wget gnupg ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Chromium (not from Playwright — system Chromium with Vulkan support)
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /usr/share/keyrings/google-chrome.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Tell Playwright to use system Chrome
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/google-chrome-stable

# Also install Playwright + its Chromium for fallback
WORKDIR /app
COPY package.json ./
RUN npm install

# Copy the Express server
COPY server.mjs ./

# Copy worker.html
COPY worker.html ./

# Copy splat.js source into /app/src/ (so worker.html can import from /src/index.js)
COPY src/ ./src/

# Create directories for runtime data
RUN mkdir -p /app/frames /app/output

# Expose the processing server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the Express server
CMD ["node", "server.mjs"]
