# TelosView v2 — RunPod Setup Guide

## Overview
This guide covers setting up RunPod GPU instances for Gaussian Splat training.

## Prerequisites
- RunPod account with GPU credits
- API key from RunPod dashboard

## GPU Recommendations
- **Development/Testing**: RTX 3090 (24GB) — ~$0.20/hr
- **Production**: A100 80GB — ~$1.50/hr
- **Batch Processing**: A100 80GB SXM — ~$2.00/hr

## Setup Steps
1. Create a pod with PyTorch template
2. Install nerfstudio or gsplat
3. Upload extracted frames
4. Run training
5. Export .splat file

## Environment Variables
```
RUNPOD_API_KEY=your_api_key
RUNPOD_GPU_TYPE=NVIDIA RTX 3090
```

## Training Command (nerfstudio)
```bash
ns-train nerfacto --data /workspace/frames --output-dir /workspace/output
```

## Export to .splat
```bash
ns-export gaussian-splat --load-config config.yml --output-dir /workspace/splat
```
