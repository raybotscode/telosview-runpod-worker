import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { Hotspot } from '../types';

interface HotspotOverlaysProps {
  hotspots: Hotspot[];
  camera: THREE.PerspectiveCamera | null;
  container: HTMLDivElement | null;
  onHotspotClick?: (hotspot: Hotspot) => void;
}

interface ProjectedHotspot {
  id: string;
  label: string;
  color: string;
  x: number;
  y: number;
  visible: boolean;
  distance: number;
}

/**
 * Projects 3D hotspot positions to 2D screen coordinates and renders
 * HTML overlays. This bypasses Three.js/Spark.js rendering entirely,
 * avoiding z-depth issues with Gaussian splats.
 */
export default function HotspotOverlays({
  hotspots,
  camera,
  container,
  onHotspotClick,
}: HotspotOverlaysProps) {
  const [projected, setProjected] = useState<ProjectedHotspot[]>([]);
  const frameRef = useRef<number>(0);

  // Project hotspots to screen space every frame
  useEffect(() => {
    if (!camera || !container || hotspots.length === 0) {
      setProjected([]);
      return;
    }

    let disposed = false;
    const vec = new THREE.Vector3();

    const update = () => {
      if (disposed) return;
      frameRef.current = requestAnimationFrame(update);

      const rect = container.getBoundingClientRect();
      const results: ProjectedHotspot[] = [];

      for (const h of hotspots) {
        // Convert OpenCV space → world space (negate Y, Z)
        vec.set(h.position.x, -h.position.y, -h.position.z);
        vec.project(camera);

        // vec is in NDC (-1 to 1). Convert to screen pixels.
        const x = (vec.x * 0.5 + 0.5) * rect.width;
        const y = (-vec.y * 0.5 + 0.5) * rect.height;

        // Hide if behind camera (z > 1 in NDC means behind)
        const visible = vec.z < 1;

        results.push({
          id: h.id,
          label: h.label,
          color: h.color || '#14b8a6',
          x,
          y,
          visible,
          distance: vec.z,
        });
      }

      setProjected(results);
    };

    frameRef.current = requestAnimationFrame(update);
    return () => {
      disposed = true;
      cancelAnimationFrame(frameRef.current);
    };
  }, [hotspots, camera, container]);

  if (!container || hotspots.length === 0) return null;

  return (
    <div className="absolute inset-0 pointer-events-none z-20 overflow-hidden">
      {projected.map((p) => (
        <div
          key={p.id}
          className="absolute pointer-events-auto cursor-pointer"
          style={{
            left: p.x,
            top: p.y,
            transform: 'translate(-50%, -50%)',
            opacity: p.visible ? 1 : 0,
            transition: 'opacity 0.15s',
          }}
          onClick={(e) => {
            e.stopPropagation();
            const hotspot = hotspots.find((h) => h.id === p.id);
            if (hotspot && onHotspotClick) onHotspotClick(hotspot);
          }}
        >
          {/* Pulsing ring */}
          <div
            className="absolute inset-0 rounded-full animate-ping"
            style={{
              backgroundColor: p.color,
              opacity: 0.3,
              width: 28,
              height: 28,
              margin: -6,
            }}
          />
          {/* Dot */}
          <div
            className="rounded-full shadow-lg"
            style={{
              width: 16,
              height: 16,
              backgroundColor: p.color,
              boxShadow: `0 0 12px ${p.color}80`,
            }}
          />
          {/* Label */}
          <div
            className="absolute left-1/2 -translate-x-1/2 whitespace-nowrap px-2 py-0.5 rounded text-xs font-semibold text-slate-200 shadow-lg"
            style={{
              top: -28,
              backgroundColor: 'rgba(15, 23, 42, 0.9)',
              border: `1px solid ${p.color}40`,
            }}
          >
            {p.label}
          </div>
        </div>
      ))}
    </div>
  );
}
