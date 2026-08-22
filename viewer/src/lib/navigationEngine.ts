import type { Hotspot } from '../types';

export interface NavigationResult {
  action: 'navigate';
  hotspot: Hotspot;
  confidence: number;
}

export interface AnswerResult {
  action: 'answer';
  text: string;
}

export interface QuestionResult {
  action: 'question';
  text: string;
}

export type IntentResult = NavigationResult | AnswerResult | QuestionResult | null;

// Synonym map for fuzzy matching
const SYNONYMS: Record<string, string[]> = {
  entrance: ['entry', 'door', 'doorway', 'front door', 'main entrance', 'main door', 'enter', 'in'],
  exit: ['way out', 'leave', 'out', 'departure'],
  desk: ['table', 'counter', 'workstation', 'workspace'],
  window: ['windows', 'glass', 'pane'],
  chair: ['seat', 'seating'],
  couch: ['sofa', 'settee'],
  kitchen: ['cooking area', 'cooking space'],
  bathroom: ['restroom', 'washroom', 'toilet', 'lavatory'],
  bedroom: ['sleeping area', 'bed', 'master bedroom'],
  living: ['living room', 'lounge', 'family room', 'sitting room'],
  stairs: ['staircase', 'steps', 'stairway'],
  elevator: ['lift'],
  lobby: ['foyer', 'reception', 'front desk'],
  office: ['workspace', 'work area'],
  garden: ['yard', 'outdoor area', 'patio'],
  garage: ['parking', 'carport'],
};

// Question detection patterns
const QUESTION_PATTERNS = /^(where|what|how|when|who|why|can you|could you|tell me|show me|explain|describe)\b/i;

// Navigation intent patterns
const NAV_PATTERNS = [
  /^(go to|take me to|navigate to|fly to|move to|show me|bring me to|let's go to|let me see|head to|zoom to|focus on)\s+(.+)/i,
  /^(go|fly|move|head)\s+(.+)/i,
  /^(.+)/i, // fallback: treat as potential hotspot name
];

// Stop words to strip from navigation commands
const STOP_WORDS = new Set([
  'the', 'a', 'an', 'to', 'at', 'in', 'on', 'of', 'for', 'is', 'it',
  'please', 'kindly', 'just', 'now', 'right', 'over', 'there',
]);

function normalizeText(text: string): string {
  return text.toLowerCase().trim().replace(/[?!.,;:]+$/, '');
}

function extractHotspotName(transcript: string): string {
  const normalized = normalizeText(transcript);

  for (const pattern of NAV_PATTERNS) {
    const match = normalized.match(pattern);
    if (match) {
      // Get the last capture group (the hotspot name part)
      const name = match[match.length - 1].trim();
      // Strip stop words
      const words = name.split(/\s+/).filter(w => !STOP_WORDS.has(w));
      return words.join(' ') || name;
    }
  }
  return normalized;
}

function exactMatch(name: string, hotspots: Hotspot[]): Hotspot | null {
  const normalized = normalizeText(name);
  return hotspots.find(h => normalizeText(h.label) === normalized) || null;
}

function fuzzyMatch(name: string, hotspots: Hotspot[]): { hotspot: Hotspot; confidence: number } | null {
  const normalized = normalizeText(name);
  const nameWords = normalized.split(/\s+/);

  let bestMatch: { hotspot: Hotspot; confidence: number } | null = null;

  for (const hotspot of hotspots) {
    const label = normalizeText(hotspot.label);
    const labelWords = label.split(/\s+/);

    // 1. Substring match: name contained in label or vice versa
    if (label.includes(normalized) || normalized.includes(label)) {
      const confidence = normalized === label ? 1.0 : 0.85;
      if (!bestMatch || confidence > bestMatch.confidence) {
        bestMatch = { hotspot, confidence };
      }
      continue;
    }

    // 2. Word overlap
    const overlap = nameWords.filter(w => labelWords.some(lw => lw.includes(w) || w.includes(lw)));
    if (overlap.length > 0) {
      const confidence = overlap.length / Math.max(nameWords.length, labelWords.length) * 0.8;
      if (confidence > 0.3 && (!bestMatch || confidence > bestMatch.confidence)) {
        bestMatch = { hotspot, confidence };
      }
    }

    // 3. Check description too
    if (hotspot.description) {
      const desc = normalizeText(hotspot.description);
      if (nameWords.some(w => w.length > 3 && desc.includes(w))) {
        const confidence = 0.5;
        if (!bestMatch || confidence > bestMatch.confidence) {
          bestMatch = { hotspot, confidence };
        }
      }
    }
  }

  return bestMatch;
}

function synonymMatch(name: string, hotspots: Hotspot[]): { hotspot: Hotspot; confidence: number } | null {
  const normalized = normalizeText(name);
  const words = normalized.split(/\s+/);

  for (const word of words) {
    // Check if word is a synonym key
    const synonyms = SYNONYMS[word] || [];
    // Also check if word is a synonym value
    for (const [key, values] of Object.entries(SYNONYMS)) {
      if (values.includes(word)) {
        synonyms.push(key, ...values);
      }
    }

    if (synonyms.length === 0) continue;

    // Try matching synonyms against hotspots
    for (const hotspot of hotspots) {
      const label = normalizeText(hotspot.label);
      const labelWords = label.split(/\s+/);

      for (const syn of synonyms) {
        if (label.includes(syn) || labelWords.some(lw => lw.includes(syn) || syn.includes(lw))) {
          return { hotspot, confidence: 0.7 };
        }
      }
    }
  }

  return null;
}

function isQuestion(text: string): boolean {
  const normalized = normalizeText(text);
  return QUESTION_PATTERNS.test(normalized);
}

export function parseNavigationIntent(
  transcript: string,
  hotspots: Hotspot[]
): IntentResult {
  if (!transcript || hotspots.length === 0) return null;

  const trimmed = transcript.trim();
  if (trimmed.length < 2) return null;

  // Check if it's a question
  if (isQuestion(trimmed)) {
    return {
      action: 'question',
      text: trimmed,
    };
  }

  // Extract the hotspot name from the command
  const hotspotName = extractHotspotName(trimmed);

  // 1. Exact match
  const exact = exactMatch(hotspotName, hotspots);
  if (exact) {
    return { action: 'navigate', hotspot: exact, confidence: 1.0 };
  }

  // 2. Fuzzy match
  const fuzzy = fuzzyMatch(hotspotName, hotspots);
  if (fuzzy && fuzzy.confidence > 0.5) {
    return { action: 'navigate', hotspot: fuzzy.hotspot, confidence: fuzzy.confidence };
  }

  // 3. Synonym match
  const synonym = synonymMatch(hotspotName, hotspots);
  if (synonym) {
    return { action: 'navigate', hotspot: synonym.hotspot, confidence: synonym.confidence };
  }

  // 4. Try the full transcript as a hotspot name (user might just say the name)
  const directExact = exactMatch(trimmed, hotspots);
  if (directExact) {
    return { action: 'navigate', hotspot: directExact, confidence: 0.9 };
  }

  const directFuzzy = fuzzyMatch(trimmed, hotspots);
  if (directFuzzy && directFuzzy.confidence > 0.4) {
    return { action: 'navigate', hotspot: directFuzzy.hotspot, confidence: directFuzzy.confidence };
  }

  // No match found — treat as a question for AI
  return {
    action: 'question',
    text: trimmed,
  };
}

/**
 * Calculate camera position to fly to a hotspot.
 * Places camera at an offset from the hotspot, looking at it.
 */
export function calculateNavigationCamera(
  hotspot: Hotspot,
  currentCameraPos: { x: number; y: number; z: number }
): { position: { x: number; y: number; z: number }; target: { x: number; y: number; z: number } } {
  const hp = hotspot.position;

  // Direction from current camera to hotspot
  const dx = hp.x - currentCameraPos.x;
  const dy = hp.y - currentCameraPos.y;
  const dz = hp.z - currentCameraPos.z;
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

  // Place camera 2 units away from hotspot, slightly above
  const offsetDist = Math.min(2, Math.max(1, dist * 0.5));
  const offsetHeight = 0.5;

  let targetPos: { x: number; y: number; z: number };

  if (dist > 0.01) {
    // Move toward hotspot but stop at offset distance
    const nx = dx / dist;
    const ny = dy / dist;
    const nz = dz / dist;
    targetPos = {
      x: hp.x - nx * offsetDist,
      y: hp.y - ny * offsetDist + offsetHeight,
      z: hp.z - nz * offsetDist,
    };
  } else {
    // Camera is at hotspot, offset forward
    targetPos = {
      x: hp.x,
      y: hp.y + offsetHeight,
      z: hp.z + offsetDist,
    };
  }

  return {
    position: targetPos,
    target: { x: hp.x, y: hp.y, z: hp.z },
  };
}
