import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import projectsRouter from './routes/projects.js';
import authRouter from './routes/auth.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = Number(process.env.PORT ?? 3457);

// Middleware
app.use(cors({
  origin: true, // Allow all origins (restrict in production)
  credentials: true
}));
app.use(express.json());

// Static file serving for extracted frames
const framesDir = path.resolve(__dirname, '..', '..', 'processing', 'frames');
app.use('/frames', express.static(framesDir));

// Serve splat.js source files (needed by worker.html ES module imports)
const splatSrcDir = process.env.SPLAT_SRC_DIR || path.resolve(__dirname, '..', '..', '..', 'splat-test', 'src');
app.use('/src', express.static(splatSrcDir));

// Serve worker.html
const workerHtmlPath = path.resolve(__dirname, '..', '..', 'processing', 'worker.html');
app.get('/worker.html', (_req, res) => {
  res.sendFile(workerHtmlPath);
});

// API routes
app.use('/api/auth', authRouter);
app.use('/api/projects', projectsRouter);

// Health check
app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    version: '2.0.0',
    splatSrcDir,
    workerHtml: workerHtmlPath,
  });
});

app.listen(PORT, () => {
  console.log(`TelosView API server running on http://localhost:${PORT}`);
  console.log(`Frames directory: ${framesDir}`);
  console.log(`Splat.js source: ${splatSrcDir}`);
  console.log(`Worker page: ${workerHtmlPath}`);
});
