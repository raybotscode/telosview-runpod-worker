# TelosView v2 — Build Plan

## Phase 1: API & Frame Extraction ✅
- [x] Express API server with project CRUD
- [x] Video upload with multer
- [x] SQLite database for project state
- [x] Frame extraction via ffmpeg
- [x] SSE progress streaming

## Phase 2: Gaussian Splat Training
- [ ] RunPod integration for GPU processing
- [ ] nerfstudio/gsplat training pipeline
- [ ] .splat file generation
- [ ] Progress tracking & webhooks

## Phase 3: 3D Viewer
- [ ] Three.js-based .splat viewer
- [ ] Camera controls (orbit, pan, zoom)
- [ ] Point cloud visualization
- [ ] Mobile-responsive viewer

## Phase 4: Website & Deployment
- [ ] Landing page
- [ ] Project management UI
- [ ] Upload flow with drag-and-drop
- [ ] Real-time progress dashboard
- [ ] Vercel/Railway deployment

## Phase 5: Docker & Production
- [ ] Docker Compose setup
- [ ] nginx reverse proxy
- [ ] SSL/TLS configuration
- [ ] Monitoring & logging
