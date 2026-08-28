import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getProject, updateProject, askProject, resolveApiUrl } from '../api/client';
import type { Project, Hotspot, Tour } from '../types';
import SplatViewer from '../components/SplatViewer';
import type { SplatViewerHandle } from '../components/SplatViewer';
import HotspotRenderer from '../components/HotspotRenderer';
import HotspotOverlays from '../components/HotspotOverlays';
import HotspotPanel from '../components/HotspotPanel';
import HotspotEditor from '../components/HotspotEditor';
import TourControls from '../components/TourControls';
import TourEditor from '../components/TourEditor';
import { useTourPlayback } from '../hooks/useTourPlayback';
import ChatPanel from '../components/ChatPanel';
import type { ChatMessage } from '../components/ChatPanel';
import { parseNavigationIntent, calculateNavigationCamera } from '../lib/navigationEngine';
import { formatDate } from '../lib/format';
import * as THREE from 'three';

export default function ViewerPage() {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [splatCount, setSplatCount] = useState<number | null>(null);
  const [showInfo, setShowInfo] = useState(false);
  const [showRotation, setShowRotation] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isPreview, setIsPreview] = useState(false);
  const [splatKey, setSplatKey] = useState(0); // Force SplatViewer remount on splat_url change

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

  // Poll for viewer handle availability after mount (runs once)
  useEffect(() => {
    const check = () => {
      if (viewerRef.current?.scene) {
        setViewerHandle(viewerRef.current);
      }
    };
    check();
    const interval = setInterval(check, 200);
    return () => clearInterval(interval);
  }, []);

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
        setIsPreview(p.status === 'preview');
        // Parse hotspots/tours from JSON string if needed
        const parsedHotspots = typeof p.hotspots === 'string' ? JSON.parse(p.hotspots) : (p.hotspots || []);
        const parsedTours = typeof p.tours === 'string' ? JSON.parse(p.tours) : (p.tours || []);
        setHotspots(parsedHotspots);
        setTours(parsedTours);
      })
      .catch((err: unknown) => {
        setLoadError(err instanceof Error ? err.message : 'Failed to load project');
      })
      .finally(() => setLoading(false));
  }, [id]);

  // Poll for updates when status is 'preview' — the full-quality splat is still processing
  useEffect(() => {
    if (!id || !isPreview) return;
    const pollInterval = setInterval(async () => {
      try {
        const p = await getProject(id);
        if (p.status === 'complete' && p.splat_url !== project?.splat_url) {
          // Full-quality splat is ready — reload
          setProject(p);
          setIsPreview(false);
          setSplatKey((k) => k + 1); // Force SplatViewer remount
        } else if (p.status === 'complete') {
          setIsPreview(false);
          setProject(p);
        }
      } catch {
        // Ignore polling errors
      }
    }, 10000); // Poll every 10 seconds
    return () => clearInterval(pollInterval);
  }, [id, isPreview, project?.splat_url]);

  const handleShare = useCallback(() => {
    const shareUrl = `${window.location.origin}/share/${id}`;
    navigator.clipboard.writeText(shareUrl).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [id]);

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

  // Hotspot placement — convert world space to OpenCV space (negate Y, Z)
  const handlePlaceHotspot = useCallback(
    (position: { x: number; y: number; z: number }) => {
      setPlacementPos({ x: position.x, y: -position.y, z: -position.z });
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
      // Convert OpenCV space → world space (negate Y, Z)
      const endPos = new THREE.Vector3(nav.position.x, -nav.position.y, -nav.position.z);
      const endTarget = new THREE.Vector3(nav.target.x, -nav.target.y, -nav.target.z);

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
      <SplatViewer key={splatKey} ref={viewerRef} url={resolveApiUrl(project.splat_url)} onLoad={handleLoad} />

      {/* Preview quality indicator */}
      {isPreview && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 bg-amber-900/80 backdrop-blur-sm text-amber-200 px-4 py-2 rounded-lg text-sm border border-amber-700 flex items-center gap-2">
          <div className="animate-spin w-4 h-4 border-2 border-amber-400 border-t-transparent rounded-full" />
          <span>Enhancing quality...</span>
        </div>
      )}

      {/* Placement mode raycasting */}
      <HotspotRenderer
        hotspots={hotspots}
        scene={viewerHandle?.scene ?? null}
        camera={viewerHandle?.camera ?? null}
        renderer={viewerHandle?.renderer ?? null}
        container={viewerHandle?.container ?? null}
        placementMode={placementMode}
        onPlaceHotspot={handlePlaceHotspot}
      />

      {/* HTML hotspot overlays (always on top of splat) */}
      <HotspotOverlays
        hotspots={hotspots}
        camera={viewerHandle?.camera ?? null}
        container={viewerHandle?.container ?? null}
        onHotspotClick={(h) => { setSelectedHotspot(h); setPlacementMode(false); }}
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
        <button
          onClick={() => setShowRotation(!showRotation)}
          className={`px-3 py-2 rounded-lg text-sm transition-colors border ${
            showRotation
              ? 'bg-amber-600 border-amber-500 text-white'
              : 'bg-slate-900/80 backdrop-blur-sm hover:bg-slate-800 text-slate-300 hover:text-white border-slate-700'
          }`}
        >
          🔄 Rotate
        </button>
      </div>

      {/* Info panel */}
      {showInfo && (
        <div className="absolute top-14 right-4 z-10 bg-slate-900/90 backdrop-blur-sm rounded-lg p-4 border border-slate-700 min-w-[200px]">
          <h3 className="text-slate-100 font-medium mb-2">{project.name}</h3>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Status</span>
              <span className={isPreview ? 'text-amber-400' : 'text-green-400'}>{isPreview ? 'Enhancing...' : project.status}</span>
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
              <span className="text-slate-200">{formatDate(project.created_at)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Rotation control panel */}
      {showRotation && (
        <div className="absolute top-14 right-4 z-10 bg-slate-900/90 backdrop-blur-sm rounded-lg p-4 border border-slate-700 min-w-[220px]">
          <h3 className="text-slate-100 font-medium mb-3 text-sm">🔄 Rotate Splat</h3>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-slate-500 text-xs w-8">X</span>
              <button onClick={() => {
                const r = viewerRef.current?.getSplatRotation();
                if (r) viewerRef.current?.rotateSplat(r.x + Math.PI / 2, r.y, r.z);
              }} className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded text-xs">+90°</button>
              <button onClick={() => {
                const r = viewerRef.current?.getSplatRotation();
                if (r) viewerRef.current?.rotateSplat(r.x - Math.PI / 2, r.y, r.z);
              }} className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded text-xs">-90°</button>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-slate-500 text-xs w-8">Y</span>
              <button onClick={() => {
                const r = viewerRef.current?.getSplatRotation();
                if (r) viewerRef.current?.rotateSplat(r.x, r.y + Math.PI / 2, r.z);
              }} className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded text-xs">+90°</button>
              <button onClick={() => {
                const r = viewerRef.current?.getSplatRotation();
                if (r) viewerRef.current?.rotateSplat(r.x, r.y - Math.PI / 2, r.z);
              }} className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded text-xs">-90°</button>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-slate-500 text-xs w-8">Z</span>
              <button onClick={() => {
                const r = viewerRef.current?.getSplatRotation();
                if (r) viewerRef.current?.rotateSplat(r.x, r.y, r.z + Math.PI / 2);
              }} className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded text-xs">+90°</button>
              <button onClick={() => {
                const r = viewerRef.current?.getSplatRotation();
                if (r) viewerRef.current?.rotateSplat(r.x, r.y, r.z - Math.PI / 2);
              }} className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded text-xs">-90°</button>
            </div>
            <div className="pt-2 border-t border-slate-700">
              <button onClick={() => {
                viewerRef.current?.rotateSplat(0, 0, 0);
              }} className="w-full bg-teal-800 hover:bg-teal-700 text-teal-200 px-2 py-1 rounded text-xs mb-1">
                Reset (no rotation)
              </button>
              <button onClick={() => {
                viewerRef.current?.rotateSplat(0, 0, 0);
              }} className="w-full bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded text-xs">
                No rotation
              </button>
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
        onEdit={(h) => { setEditingHotspot(h); setSelectedHotspot(null); }}
        onDelete={handleDeleteHotspot}
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
