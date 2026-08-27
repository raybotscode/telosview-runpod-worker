import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { resolveApiUrl } from '../api/client';
import type { Project, Hotspot, Tour } from '../types';
import SplatViewer from '../components/SplatViewer';
import type { SplatViewerHandle } from '../components/SplatViewer';
import HotspotRenderer from '../components/HotspotRenderer';
import HotspotOverlays from '../components/HotspotOverlays';
import HotspotPanel from '../components/HotspotPanel';
import TourControls from '../components/TourControls';
import { useTourPlayback } from '../hooks/useTourPlayback';
import { calculateNavigationCamera } from '../lib/navigationEngine';
import * as THREE from 'three';

const API = import.meta.env.VITE_API_URL || '/api';

export default function SharePage() {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [_splatCount, setSplatCount] = useState<number | null>(null);

  // Hotspot state (view-only)
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);
  const [selectedHotspot, setSelectedHotspot] = useState<Hotspot | null>(null);

  // Tour state
  const [tours, setTours] = useState<Tour[]>([]);
  const [activeTour, setActiveTour] = useState<Tour | null>(null);

  const viewerRef = useRef<SplatViewerHandle>(null);
  const [viewerHandle, setViewerHandle] = useState<SplatViewerHandle | null>(null);
  const navAnimRef = useRef<number>(0);

  // Poll for viewer handle
  useEffect(() => {
    const check = () => {
      if (viewerRef.current?.scene && viewerRef.current !== viewerHandle) {
        setViewerHandle(viewerRef.current);
      }
    };
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

  // Load project via public share endpoint
  useEffect(() => {
    if (!id) return;
    fetch(`${API}/projects/${id}/share`)
      .then(async (res) => {
        if (!res.ok) {
          const err = await res.json().catch(() => ({ error: 'Not found' }));
          throw new Error(err.error || 'Project not found');
        }
        return res.json();
      })
      .then((p: Project) => {
        setProject(p);
        // Parse hotspots/tours from JSON string if needed
        const parsedHotspots = typeof p.hotspots === 'string' ? JSON.parse(p.hotspots) : (p.hotspots || []);
        const parsedTours = typeof p.tours === 'string' ? JSON.parse(p.tours) : (p.tours || []);
        setHotspots(parsedHotspots);
        setTours(parsedTours);
      })
      .catch((err: Error) => {
        setLoadError(err.message);
      })
      .finally(() => setLoading(false));
  }, [id]);

  const handleLoad = useCallback((info: { splatCount: number }) => {
    setSplatCount(info.splatCount);
  }, []);

  // Fly to hotspot (view-only, no editing)
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
          setSelectedHotspot(hotspot);
        }
      };

      cancelAnimationFrame(navAnimRef.current);
      navAnimRef.current = requestAnimationFrame(animate);
    },
    [viewerHandle]
  );

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
          <p className="text-slate-400 text-lg mb-2">{loadError || 'Project not found'}</p>
          <p className="text-slate-600 text-sm">This link may have expired or the project was deleted.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen relative bg-slate-950">
      {/* 3D Viewer */}
      <SplatViewer ref={viewerRef} url={resolveApiUrl(project.splat_url)} onLoad={handleLoad} />

      {/* Placement mode raycasting (view-only, no placement) */}
      <HotspotRenderer
        hotspots={hotspots}
        scene={viewerHandle?.scene ?? null}
        camera={viewerHandle?.camera ?? null}
        renderer={viewerHandle?.renderer ?? null}
        container={viewerHandle?.container ?? null}
        placementMode={false}
      />

      {/* HTML hotspot overlays (always on top of splat) */}
      <HotspotOverlays
        hotspots={hotspots}
        camera={viewerHandle?.camera ?? null}
        container={viewerHandle?.container ?? null}
        onHotspotClick={(h) => { setSelectedHotspot(h); flyToHotspot(h); }}
      />

      {/* Tour selector (if tours exist) */}
      {tours.length > 0 && !activeTour && (
        <div className="absolute top-4 right-4 z-10">
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
        </div>
      )}

      {/* Hotspot detail panel */}
      <HotspotPanel
        hotspot={selectedHotspot}
        onClose={() => setSelectedHotspot(null)}
      />

      {/* Tour playback controls */}
      {activeTour && (
        <TourControls
          tourName={activeTour.name}
          state={tourState}
          controls={tourControls}
          onClose={() => setActiveTour(null)}
        />
      )}

      {/* Interaction hint */}
      <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10 text-slate-500 text-xs">
        Drag to rotate · Scroll to zoom · Pinch on mobile
      </div>

      {/* TelosView branding — bottom center */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10">
        <a
          href="https://telosview.com"
          target="_blank"
          rel="noopener noreferrer"
          className="text-slate-400 hover:text-teal-400 text-sm font-medium tracking-wide transition-colors"
        >
          TelosView
        </a>
      </div>
    </div>
  );
}
