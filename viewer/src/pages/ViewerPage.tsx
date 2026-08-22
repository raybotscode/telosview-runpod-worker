import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getProject, updateProject, askProject } from '../api/client';
import type { Project, Hotspot, Tour } from '../types';
import SplatViewer from '../components/SplatViewer';
import type { SplatViewerHandle } from '../components/SplatViewer';
import HotspotRenderer from '../components/HotspotRenderer';
import HotspotPanel from '../components/HotspotPanel';
import HotspotEditor from '../components/HotspotEditor';
import TourControls from '../components/TourControls';
import TourEditor from '../components/TourEditor';
import { useTourPlayback } from '../hooks/useTourPlayback';
import ChatPanel from '../components/ChatPanel';
import type { ChatMessage } from '../components/ChatPanel';
import { parseNavigationIntent, calculateNavigationCamera } from '../lib/navigationEngine';
import * as THREE from 'three';

export default function ViewerPage() {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [splatCount, setSplatCount] = useState<number | null>(null);
  const [showInfo, setShowInfo] = useState(false);
  const [copied, setCopied] = useState(false);

  // Hotspot state
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);
  const [selectedHotspot, setSelectedHotspot] = useState<Hotspot | null>(null);
  const [placementMode, setPlacementMode] = useState(false);
  const [placementPos, setPlacementPos] = useState<{ x: number; y: number; z: number } | null>(null);
  const [editingHotspot, setEditingHotspot] = useState<Hotspot | null>(null);

  // Tour state
  const [tours, setTours] = useState<Tour[]>([]);
  const [activeTour, setActiveTour] = useState<Tour | null>(null);
  const [showTourEditor, setShowTourEditor] = useState(false);

  // Chat / navigation state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatMinimized, setChatMinimized] = useState(true);
  const [isChatProcessing, setIsChatProcessing] = useState(false);
  const [navToast, setNavToast] = useState<string | null>(null);
  const navAnimRef = useRef<number>(0);

  // Viewer handle — stored in state so React tracks it properly
  const viewerRef = useRef<SplatViewerHandle>(null);
  const [viewerHandle, setViewerHandle] = useState<SplatViewerHandle | null>(null);

  // Poll for viewer handle availability after mount
  useEffect(() => {
    const check = () => {
      if (viewerRef.current?.scene && viewerRef.current !== viewerHandle) {
        setViewerHandle(viewerRef.current);
      }
    };
    // Check immediately and on a short interval until ready
    check();
    const interval = setInterval(check, 200);
    return () => clearInterval(interval);
  }, [viewerHandle]);

  // Tour playback
  const [tourState, tourControls] = useTourPlayback(
    activeTour,
    viewerHandle?.camera ?? null,
    viewerHandle?.controls ?? null
  );

  // Load project
  useEffect(() => {
    if (!id) return;
    getProject(id)
      .then((p) => {
        setProject(p);
        setHotspots((p.hotspots as Hotspot[]) || []);
        setTours((p.tours as Tour[]) || []);
      })
      .catch((err: unknown) => {
        setLoadError(err instanceof Error ? err.message : 'Failed to load project');
      })
      .finally(() => setLoading(false));
  }, [id]);

  const handleShare = useCallback(() => {
    navigator.clipboard.writeText(window.location.href).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, []);

  const handleLoad = useCallback((info: { splatCount: number }) => {
    setSplatCount(info.splatCount);
  }, []);

  // Save hotspots to API
  const saveHotspots = useCallback(
    async (updated: Hotspot[]) => {
      setHotspots(updated);
      if (id) {
        try {
          await updateProject(id, { hotspots: JSON.stringify(updated) });
        } catch (err) {
          console.error('Failed to save hotspots:', err);
        }
      }
    },
    [id]
  );

  // Save tours to API
  const saveTours = useCallback(
    async (updated: Tour[]) => {
      setTours(updated);
      if (id) {
        try {
          await updateProject(id, { tours: JSON.stringify(updated) });
        } catch (err) {
          console.error('Failed to save tours:', err);
        }
      }
    },
    [id]
  );

  // Hotspot placement
  const handlePlaceHotspot = useCallback(
    (position: { x: number; y: number; z: number }) => {
      setPlacementPos(position);
      setPlacementMode(false);
    },
    []
  );

  // Hotspot save from editor
  const handleSaveHotspot = useCallback(
    (hotspot: Hotspot) => {
      const exists = hotspots.find((h) => h.id === hotspot.id);
      const updated = exists
        ? hotspots.map((h) => (h.id === hotspot.id ? hotspot : h))
        : [...hotspots, hotspot];
      saveHotspots(updated);
      setPlacementPos(null);
      setEditingHotspot(null);
    },
    [hotspots, saveHotspots]
  );

  // Hotspot delete
  const handleDeleteHotspot = useCallback(
    (hotspotId: string) => {
      const updated = hotspots.filter((h) => h.id !== hotspotId);
      saveHotspots(updated);
      setEditingHotspot(null);
      if (selectedHotspot?.id === hotspotId) setSelectedHotspot(null);
    },
    [hotspots, saveHotspots, selectedHotspot]
  );

  // Tour save
  const handleSaveTour = useCallback(
    (tour: Tour) => {
      saveTours([...tours, tour]);
      setShowTourEditor(false);
    },
    [tours, saveTours]
  );

  // Fly camera to a hotspot with smooth lerp
  const flyToHotspot = useCallback(
    (hotspot: Hotspot) => {
      const camera = viewerHandle?.camera;
      const controls = viewerHandle?.controls;
      if (!camera || !controls) return;

      const nav = calculateNavigationCamera(hotspot, camera.position);
      const startPos = camera.position.clone();
      const startTarget = controls.target.clone();
      const endPos = new THREE.Vector3(nav.position.x, nav.position.y, nav.position.z);
      const endTarget = new THREE.Vector3(nav.target.x, nav.target.y, nav.target.z);

      let t = 0;
      const CAMERA_LERP = 0.09;

      const animate = () => {
        t = Math.min(1, t + CAMERA_LERP);
        const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

        camera.position.lerpVectors(startPos, endPos, ease);
        controls.target.lerpVectors(startTarget, endTarget, ease);

        if (t < 1) {
          navAnimRef.current = requestAnimationFrame(animate);
        } else {
          // Open the hotspot panel when we arrive
          setSelectedHotspot(hotspot);
        }
      };

      cancelAnimationFrame(navAnimRef.current);
      navAnimRef.current = requestAnimationFrame(animate);
    },
    [viewerHandle]
  );

  // Add a chat message
  const addChatMessage = useCallback((msg: Omit<ChatMessage, 'id' | 'timestamp'>) => {
    setChatMessages((prev) => [
      ...prev,
      { ...msg, id: crypto.randomUUID(), timestamp: Date.now() },
    ]);
  }, []);

  // Handle chat message (text or voice)
  const handleChatMessage = useCallback(
    async (text: string) => {
      if (!text.trim()) return;

      // Add user message
      addChatMessage({ role: 'user', text });
      setChatMinimized(false);
      setIsChatProcessing(true);

      // Parse intent
      const intent = parseNavigationIntent(text, hotspots);

      if (intent?.action === 'navigate') {
        // Navigate to hotspot
        const hotspot = intent.hotspot;
        addChatMessage({
          role: 'system',
          text: `Navigating to ${hotspot.label}...`,
        });
        flyToHotspot(hotspot);
        setNavToast(`Navigating to ${hotspot.label}`);
        setTimeout(() => setNavToast(null), 3000);
      } else if (intent?.action === 'question' || intent?.action === 'answer') {
        // Try AI fallback
        if (id) {
          try {
            const response = await askProject(id, text);
            addChatMessage({ role: 'assistant', text: response.answer });
            // If AI returned a camera target, navigate there
            if (response.cameraTarget) {
              const fakeHotspot: Hotspot = {
                id: 'ai-target',
                position: response.cameraTarget,
                label: 'AI suggested location',
              };
              flyToHotspot(fakeHotspot);
            }
          } catch {
            addChatMessage({
              role: 'assistant',
              text: "I couldn't find a matching location. Try saying a hotspot name, like \"go to the desk\".",
            });
          }
        } else {
          addChatMessage({
            role: 'assistant',
            text: "I couldn't match that to a hotspot. Try saying a location name.",
          });
        }
      } else {
        addChatMessage({
          role: 'assistant',
          text: "I'm not sure what you mean. Try \"go to [location]\" or ask a question.",
        });
      }

      setIsChatProcessing(false);
    },
    [hotspots, id, addChatMessage, flyToHotspot]
  );

  // Cleanup nav animation on unmount
  useEffect(() => {
    return () => cancelAnimationFrame(navAnimRef.current);
  }, []);

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-950">
        <div className="animate-spin w-10 h-10 border-4 border-teal-500 border-t-transparent rounded-full" />
      </div>
    );
  }

  if (!project || !project.splat_url) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-950">
        <div className="text-center">
          <p className="text-red-400 mb-4">{loadError || (!project ? 'Project not found' : 'No splat available')}</p>
          <Link to="/" className="text-teal-400 hover:underline">← Back to Dashboard</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen relative bg-slate-950">
      {/* 3D Viewer */}
      <SplatViewer ref={viewerRef} url={project.splat_url} onLoad={handleLoad} />

      {/* Hotspot 3D layer */}
      <HotspotRenderer
        hotspots={hotspots}
        scene={viewerHandle?.scene ?? null}
        camera={viewerHandle?.camera ?? null}
        renderer={viewerHandle?.renderer ?? null}
        container={viewerHandle?.container ?? null}
        placementMode={placementMode}
        onHotspotClick={(h) => { setSelectedHotspot(h); setPlacementMode(false); }}
        onPlaceHotspot={handlePlaceHotspot}
      />

      {/* Top-left: back button */}
      <div className="absolute top-4 left-4 z-10">
        <Link
          to="/"
          className="bg-slate-900/80 backdrop-blur-sm hover:bg-slate-800 text-slate-300 hover:text-white px-3 py-2 rounded-lg text-sm transition-colors border border-slate-700"
        >
          ← Dashboard
        </Link>
      </div>

      {/* Top-right: toolbar + share + info */}
      <div className="absolute top-4 right-4 z-10 flex flex-wrap gap-2 justify-end">
        <button
          onClick={() => { setPlacementMode(!placementMode); setSelectedHotspot(null); }}
          className={`px-3 py-2 rounded-lg text-sm transition-colors border ${
            placementMode
              ? 'bg-teal-600 border-teal-500 text-white'
              : 'bg-slate-900/80 backdrop-blur-sm hover:bg-slate-800 text-slate-300 hover:text-white border-slate-700'
          }`}
        >
          {placementMode ? '✕ Cancel' : '📍 Add Hotspot'}
        </button>
        <button
          onClick={() => setShowTourEditor(true)}
          className="bg-slate-900/80 backdrop-blur-sm hover:bg-slate-800 text-slate-300 hover:text-white px-3 py-2 rounded-lg text-sm transition-colors border border-slate-700"
        >
          🎬 Create Tour
        </button>
        {tours.length > 0 && !activeTour && (
          <select
            onChange={(e) => {
              const t = tours.find((tour) => tour.id === e.target.value);
              if (t) setActiveTour(t);
            }}
            defaultValue=""
            className="bg-slate-900/80 backdrop-blur-sm text-slate-300 px-3 py-2 rounded-lg text-sm border border-slate-700 cursor-pointer"
          >
            <option value="" disabled>▶ Play Tour</option>
            {tours.map((t) => (
              <option key={t.id} value={t.id}>{t.name}</option>
            ))}
          </select>
        )}
        <button
          onClick={handleShare}
          className="bg-slate-900/80 backdrop-blur-sm hover:bg-slate-800 text-slate-300 hover:text-white px-3 py-2 rounded-lg text-sm transition-colors border border-slate-700"
        >
          {copied ? '✓ Copied!' : '🔗 Share'}
        </button>
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="bg-slate-900/80 backdrop-blur-sm hover:bg-slate-800 text-slate-300 hover:text-white px-3 py-2 rounded-lg text-sm transition-colors border border-slate-700"
        >
          ℹ️ Info
        </button>
      </div>

      {/* Info panel */}
      {showInfo && (
        <div className="absolute top-14 right-4 z-10 bg-slate-900/90 backdrop-blur-sm rounded-lg p-4 border border-slate-700 min-w-[200px]">
          <h3 className="text-slate-100 font-medium mb-2">{project.name}</h3>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Status</span>
              <span className="text-green-400">{project.status}</span>
            </div>
            {splatCount !== null && (
              <div className="flex justify-between">
                <span className="text-slate-500">Splats</span>
                <span className="text-slate-200">{splatCount.toLocaleString()}</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-slate-500">Hotspots</span>
              <span className="text-slate-200">{hotspots.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Tours</span>
              <span className="text-slate-200">{tours.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Created</span>
              <span className="text-slate-200">{new Date(project.created_at).toLocaleDateString()}</span>
            </div>
          </div>
        </div>
      )}

      {/* Placement mode hint */}
      {placementMode && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-20 bg-teal-900/90 text-teal-200 px-4 py-2 rounded-lg text-sm border border-teal-700">
          Click anywhere in the scene to place a hotspot
        </div>
      )}

      {/* Hotspot detail panel */}
      <HotspotPanel
        hotspot={selectedHotspot}
        onClose={() => setSelectedHotspot(null)}
      />

      {/* Hotspot editor (placement or edit) */}
      <HotspotEditor
        editingHotspot={editingHotspot}
        placementPosition={placementPos}
        onSave={handleSaveHotspot}
        onCancel={() => { setPlacementPos(null); setEditingHotspot(null); }}
        onDelete={handleDeleteHotspot}
      />

      {/* Tour editor */}
      {showTourEditor && (
        <TourEditor
          hotspots={hotspots}
          onSave={handleSaveTour}
          onCancel={() => setShowTourEditor(false)}
        />
      )}

      {/* Tour playback controls */}
      {activeTour && (
        <TourControls
          tourName={activeTour.name}
          state={tourState}
          controls={tourControls}
          onClose={() => setActiveTour(null)}
        />
      )}

      {/* Bottom hint */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 text-slate-500 text-xs">
        Drag to rotate · Scroll to zoom · Pinch on mobile
      </div>

      {/* Navigation toast */}
      {navToast && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-30 bg-teal-900/90 backdrop-blur-sm text-teal-200 px-4 py-2 rounded-lg text-sm border border-teal-700 animate-pulse">
          ✈️ {navToast}
        </div>
      )}

      {/* Chat panel */}
      <ChatPanel
        messages={chatMessages}
        onSendMessage={handleChatMessage}
        isProcessing={isChatProcessing}
        minimized={chatMinimized}
        onToggleMinimize={() => setChatMinimized(!chatMinimized)}
      />
    </div>
  );
}
