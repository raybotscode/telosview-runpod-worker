/**
 * AI Scene Analysis for TelosView v2
 * Uses Gemini via OpenRouter to understand 3D Gaussian splat scenes.
 */

import fs from 'fs';
import path from 'path';

// ── Types ──

export interface SceneObject {
  name: string;
  area: string;
  description: string;
}

export interface SuggestedHotspot {
  label: string;
  area: string;
  description: string;
  cameraHint: string;
  confidence: number;
}

export interface SceneAnalysis {
  sceneType: string;
  description: string;
  objects: SceneObject[];
  suggestedHotspots: SuggestedHotspot[];
  analyzedAt: string;
}

export interface CameraTarget {
  yaw: number;
  pitch: number;
  distance: number;
}

export interface QAResult {
  answer: string;
  cameraTarget?: CameraTarget;
}

// ── Config ──

const OPENROUTER_API = 'https://openrouter.ai/api/v1/chat/completions';
const MODEL = 'google/gemini-2.5-flash';

function getApiKey(): string {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error('OPENROUTER_API_KEY environment variable is required');
  return key;
}

// ── OpenRouter Integration ──

interface OpenRouterMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | Array<{ type: string; text?: string; image_url?: { url: string } }>;
}

async function callOpenRouter(messages: OpenRouterMessage[]): Promise<string> {
  const response = await fetch(OPENROUTER_API, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${getApiKey()}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://telosview.com',
      'X-Title': 'TelosView',
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      max_tokens: 4096,
      temperature: 0.3,
    }),
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => '');
    throw new Error(`OpenRouter API error ${response.status}: ${errText}`);
  }

  const data = await response.json() as any;
  if (data.error) {
    throw new Error(`OpenRouter error: ${data.error.message || JSON.stringify(data.error)}`);
  }

  return data.choices?.[0]?.message?.content ?? '';
}

// ── Prompts ──

const SCENE_ANALYSIS_PROMPT = `You are analyzing a 3D scene captured from a video. The scene has been reconstructed as a 3D Gaussian splat.

Based on the provided frames, identify:
1. Scene type (indoor/outdoor, room type, environment)
2. Key objects and areas (furniture, doors, windows, equipment, etc.)
3. Spatial layout (where things are relative to each other)
4. Notable features (interesting areas, focal points)
5. Suggested hotspot locations (3-7 areas a user would want to explore)

Return ONLY a JSON object (no markdown fences, no explanation):
{
  "sceneType": "indoor-office",
  "description": "A modern office space with...",
  "objects": [
    { "name": "desk", "area": "center", "description": "Large wooden desk with monitor" },
    { "name": "bookshelf", "area": "left-wall", "description": "Floor-to-ceiling bookshelf" }
  ],
  "suggestedHotspots": [
    { "label": "Main Desk", "area": "center", "description": "Primary workspace area", "cameraHint": "facing desk from entrance", "confidence": 0.9 },
    { "label": "Meeting Area", "area": "right", "description": "Conference table with chairs", "cameraHint": "wide view of meeting space", "confidence": 0.85 }
  ]
}`;

const HOTSPOT_PROMPT = `You are suggesting navigation hotspots for a 3D scene viewer. Based on the provided frames, suggest 3-7 interesting locations a user would want to explore.

For each hotspot, estimate a camera position using spherical coordinates:
- yaw: horizontal angle in degrees (0-360, 0 = front, 90 = right, 180 = behind, 270 = left)
- pitch: vertical angle in degrees (-45 to 45, 0 = level, positive = looking down)
- distance: relative distance (1-5, where 3 is normal viewing distance)

Return ONLY a JSON array (no markdown fences):
[
  { "label": "Main Desk", "area": "center", "description": "Primary workspace area", "cameraHint": "facing desk from entrance", "camera": { "yaw": 45, "pitch": -5, "distance": 3 }, "confidence": 0.9 }
]`;

const QA_PROMPT = `You are answering questions about a 3D scene. The user is viewing a Gaussian splat reconstruction and wants to navigate to or learn about something in the scene.

When answering, provide:
1. A clear answer to the question
2. If the answer involves a location, suggest a camera position (yaw 0-360, pitch -45 to 45, distance 1-5)

Scene context: {context}

Return ONLY a JSON object (no markdown fences):
{
  "answer": "The lab is through the door on the left wall.",
  "cameraTarget": { "yaw": 270, "pitch": 0, "distance": 3 }
}`;

// ── Frame Utilities ──

/**
 * Select N evenly-spaced frames from a directory of extracted frames.
 */
export function selectRepresentativeFrames(framesDir: string, count: number = 5): string[] {
  const allFrames = fs.readdirSync(framesDir)
    .filter(f => /^frame_\d{5}\.jpg$/i.test(f))
    .sort()
    .map(f => path.join(framesDir, f));

  if (allFrames.length === 0) return [];
  if (allFrames.length <= count) return allFrames;

  const step = (allFrames.length - 1) / (count - 1);
  const selected: string[] = [];
  for (let i = 0; i < count; i++) {
    selected.push(allFrames[Math.round(i * step)]);
  }
  return selected;
}

/**
 * Read frame files and encode as base64 data URIs for the API.
 */
function framesToDataUris(framePaths: string[]): string[] {
  return framePaths.map(fp => {
    const buf = fs.readFileSync(fp);
    return `data:image/jpeg;base64,${buf.toString('base64')}`;
  });
}

/**
 * Build a multimodal user message with text + images.
 */
function buildMultimodalMessage(text: string, imageUris: string[]): OpenRouterMessage {
  const content: any[] = [{ type: 'text', text }];
  for (const uri of imageUris) {
    content.push({ type: 'image_url', image_url: { url: uri } });
  }
  return { role: 'user', content };
}

/**
 * Parse JSON from model response, stripping markdown fences if present.
 */
function parseJSON<T>(raw: string): T {
  let cleaned = raw.trim();
  // Strip markdown code fences
  cleaned = cleaned.replace(/^```(?:json)?\s*\n?/i, '').replace(/\n?```\s*$/i, '');
  return JSON.parse(cleaned) as T;
}

// ── Public API ──

/**
 * Analyze a scene from representative frames.
 * Returns structured scene analysis with objects and suggested hotspots.
 */
export async function analyzeScene(
  framesDir: string,
  frameCount: number = 5
): Promise<SceneAnalysis> {
  const framePaths = selectRepresentativeFrames(framesDir, frameCount);
  if (framePaths.length === 0) {
    throw new Error('No frames found in directory');
  }

  console.log(`[scene-analyzer] Analyzing ${framePaths.length} frames from ${framesDir}`);
  const imageUris = framesToDataUris(framePaths);

  const message = buildMultimodalMessage(SCENE_ANALYSIS_PROMPT, imageUris);
  const raw = await callOpenRouter([message]);

  const analysis = parseJSON<Omit<SceneAnalysis, 'analyzedAt'>>(raw);
  return {
    ...analysis,
    analyzedAt: new Date().toISOString(),
  };
}

/**
 * Suggest hotspots with camera positions for a scene.
 */
export async function suggestHotspots(
  framesDir: string,
  frameCount: number = 5
): Promise<SuggestedHotspot[]> {
  const framePaths = selectRepresentativeFrames(framesDir, frameCount);
  if (framePaths.length === 0) {
    throw new Error('No frames found in directory');
  }

  console.log(`[scene-analyzer] Suggesting hotspots from ${framePaths.length} frames`);
  const imageUris = framesToDataUris(framePaths);

  const message = buildMultimodalMessage(HOTSPOT_PROMPT, imageUris);
  const raw = await callOpenRouter([message]);

  return parseJSON<SuggestedHotspot[]>(raw);
}

/**
 * Answer a spatial question about the scene.
 * Returns answer text and optional camera target for navigation.
 */
export async function answerQuestion(
  question: string,
  sceneContext: string,
  framesDir: string,
  frameCount: number = 3
): Promise<QAResult> {
  const framePaths = selectRepresentativeFrames(framesDir, frameCount);
  if (framePaths.length === 0) {
    throw new Error('No frames found in directory');
  }

  console.log(`[scene-analyzer] Answering: "${question}"`);
  const imageUris = framesToDataUris(framePaths);

  const prompt = QA_PROMPT.replace('{context}', sceneContext);
  const message = buildMultimodalMessage(
    `${prompt}\n\nUser question: ${question}`,
    imageUris
  );
  const raw = await callOpenRouter([message]);

  return parseJSON<QAResult>(raw);
}
