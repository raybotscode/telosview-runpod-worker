import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';
import type { Project, ProjectStatus } from './types.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = path.join(__dirname, '..', 'data', 'telosview.db');

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const db: any = new Database(DB_PATH);

// Enable WAL mode for better concurrent access
db.pragma('journal_mode = WAL');

// Create tables
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
  );

  CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'created',
    video_path TEXT,
    frame_count INTEGER DEFAULT 0,
    splat_url TEXT,
    scene_analysis TEXT,
    error TEXT,
    user_id TEXT REFERENCES users(id),
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
  );
`);

// Migration: add scene_analysis column if missing
try {
  db.prepare('SELECT scene_analysis FROM projects LIMIT 1').get();
} catch {
  db.exec('ALTER TABLE projects ADD COLUMN scene_analysis TEXT');
}

// Migration: add user_id column if missing
try {
  db.prepare('SELECT user_id FROM projects LIMIT 1').get();
} catch {
  db.exec('ALTER TABLE projects ADD COLUMN user_id TEXT REFERENCES users(id)');
}

export function createProject(id: string, name: string, description?: string, userId?: string): Project {
  const stmt = db.prepare(
    'INSERT INTO projects (id, name, description, user_id) VALUES (?, ?, ?, ?)'
  );
  stmt.run(id, name, description || null, userId || null);
  return getProject(id)!;
}

export function getProject(id: string): Project | undefined {
  return db.prepare('SELECT * FROM projects WHERE id = ?').get(id) as Project | undefined;
}

export function getAllProjects(userId?: string): Project[] {
  if (userId) {
    return db.prepare('SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC').all(userId) as Project[];
  }
  return db.prepare('SELECT * FROM projects ORDER BY created_at DESC').all() as Project[];
}

export function updateProjectStatus(id: string, status: ProjectStatus, error?: string): void {
  const stmt = db.prepare(
    "UPDATE projects SET status = ?, error = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
  );
  stmt.run(status, error || null, id);
}

export function updateProjectVideoPath(id: string, videoPath: string): void {
  const stmt = db.prepare(
    "UPDATE projects SET video_path = ?, status = 'uploading', updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
  );
  stmt.run(videoPath, id);
}

export function updateProjectFrameCount(id: string, frameCount: number): void {
  const stmt = db.prepare(
    "UPDATE projects SET frame_count = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
  );
  stmt.run(frameCount, id);
}

export function updateProjectSplatUrl(id: string, splatUrl: string): void {
  const stmt = db.prepare(
    "UPDATE projects SET splat_url = ?, status = 'complete', updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
  );
  stmt.run(splatUrl, id);
}

export function deleteProject(id: string): boolean {
  const result = db.prepare('DELETE FROM projects WHERE id = ?').run(id);
  return result.changes > 0;
}

export function getProjectSceneAnalysis(id: string): string | null {
  const row = db.prepare('SELECT scene_analysis FROM projects WHERE id = ?').get(id) as { scene_analysis: string | null } | undefined;
  return row?.scene_analysis ?? null;
}

export function updateProjectSceneAnalysis(id: string, analysis: string): void {
  const stmt = db.prepare(
    "UPDATE projects SET scene_analysis = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?"
  );
  stmt.run(analysis, id);
}

export default db;
