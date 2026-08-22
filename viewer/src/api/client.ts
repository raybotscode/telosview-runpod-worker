import type { Project } from '../types';

const API = import.meta.env.VITE_API_URL || '/api';

function getToken(): string | null {
  return localStorage.getItem('token');
}

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options?.headers as Record<string, string> || {}),
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const res = await fetch(url, { ...options, headers });
  if (res.status === 401) {
    localStorage.removeItem('token');
    window.location.href = '/login';
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `Server unreachable (${res.status})` }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json().catch(() => {
    throw new Error(`Invalid server response (${res.status})`);
  });
}

// Auth functions
export async function login(email: string, password: string): Promise<{ token: string; user: any }> {
  const res = await fetch(`${API}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: `Server unreachable (${res.status})` }));
    throw new Error(data.error || 'Login failed');
  }
  return res.json().catch(() => {
    throw new Error(`Invalid server response (${res.status})`);
  });
}

export async function register(email: string, password: string, name: string): Promise<{ token: string; user: any }> {
  const res = await fetch(`${API}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, name }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: `Server unreachable (${res.status})` }));
    throw new Error(data.error || 'Registration failed');
  }
  return res.json().catch(() => {
    throw new Error(`Invalid server response (${res.status})`);
  });
}

export async function getMe(token: string): Promise<{ user: any }> {
  const res = await fetch(`${API}/auth/me`, {
    headers: { 'Authorization': `Bearer ${token}` },
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: `Server unreachable (${res.status})` }));
    throw new Error(data.error || 'Failed to get user');
  }
  const data = await res.json().catch(() => {
    throw new Error(`Invalid server response (${res.status})`);
  });
  return data;
}

// Project functions
export async function createProject(name: string, description?: string): Promise<Project> {
  return fetchJSON<Project>(`${API}/projects`, {
    method: 'POST',
    body: JSON.stringify({ name, description }),
  });
}

export async function getProjects(): Promise<Project[]> {
  return fetchJSON<Project[]>(`${API}/projects`);
}

export async function getProject(id: string): Promise<Project> {
  return fetchJSON<Project>(`${API}/projects/${id}`);
}

export async function uploadVideo(
  id: string,
  file: File,
  onProgress?: (pct: number) => void
): Promise<Project> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}/projects/${id}/upload`);

    const token = getToken();
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error(`Invalid server response (${xhr.status})`));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.error || `Upload failed: ${xhr.status}`));
        } catch {
          reject(new Error(`Upload failed: ${xhr.status}`));
        }
      }
    });

    xhr.addEventListener('error', () => reject(new Error('Upload failed')));

    const form = new FormData();
    form.append('video', file);
    xhr.send(form);
  });
}

export async function processSplat(id: string): Promise<{ message: string }> {
  return fetchJSON<{ message: string }>(`${API}/projects/${id}/process`, {
    method: 'POST',
  });
}

export async function deleteProject(id: string): Promise<void> {
  await fetchJSON<void>(`${API}/projects/${id}`, { method: 'DELETE' });
}

export async function updateProject(id: string, data: Record<string, unknown>): Promise<Project> {
  return fetchJSON<Project>(`${API}/projects/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

export async function askProject(
  id: string,
  question: string
): Promise<{ answer: string; cameraTarget?: { x: number; y: number; z: number } }> {
  return fetchJSON(`${API}/projects/${id}/ask`, {
    method: 'POST',
    body: JSON.stringify({ question }),
  });
}

export function connectSSE(
  id: string,
  onEvent: (data: Record<string, unknown>) => void
): EventSource {
  const es = new EventSource(`${API}/projects/${id}/status`);

  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onEvent(data);
    } catch {
      // ignore parse errors
    }
  };

  es.onerror = () => {
    es.close();
  };

  return es;
}
