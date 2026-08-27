import { useEffect } from 'react';
import * as THREE from 'three';
import type { Hotspot } from '../types';

interface HotspotRendererProps {
  hotspots: Hotspot[];
  scene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  renderer: THREE.WebGLRenderer | null;
  container: HTMLDivElement | null;
  placementMode?: boolean;
  onPlaceHotspot?: (position: { x: number; y: number; z: number }) => void;
}

/**
 * Handles placement-mode raycasting only.
 * All visual hotspot rendering is done by HotspotOverlays (HTML overlays).
 */
export default function HotspotRenderer({
  hotspots: _hotspots,
  scene,
  camera,
  renderer,
  container,
  placementMode = false,
  onPlaceHotspot,
}: HotspotRendererProps) {
  // Placement mode click handler — raycasts to find 3D position
  useEffect(() => {
    if (!container || !placementMode) return;

    const handleClick = (e: MouseEvent) => {
      if (!onPlaceHotspot || !camera || !renderer || !scene) return;

      const rect = container.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);

      // Try to intersect with scene objects (excluding hotspots)
      const sceneObjects: THREE.Object3D[] = [];
      scene.traverse((child) => {
        if (
          child.userData.type !== 'hotspot' &&
          child.userData.type !== 'ring' &&
          child.userData.type !== 'label' &&
          !(child instanceof THREE.Sprite) &&
          (child instanceof THREE.Mesh)
        ) {
          sceneObjects.push(child);
        }
      });

      const intersects = raycaster.intersectObjects(sceneObjects, true);
      if (intersects.length > 0) {
        const pt = intersects[0].point;
        onPlaceHotspot({ x: pt.x, y: pt.y, z: pt.z });
      } else {
        // Fallback: place at a fixed distance along the ray
        const dir = raycaster.ray.direction.clone().multiplyScalar(3);
        const pos = raycaster.ray.origin.clone().add(dir);
        onPlaceHotspot({ x: pos.x, y: pos.y, z: pos.z });
      }
    };

    container.addEventListener('click', handleClick);
    return () => container.removeEventListener('click', handleClick);
  }, [container, placementMode, camera, renderer, scene, onPlaceHotspot]);

  // Set cursor for placement mode
  useEffect(() => {
    if (!container) return;
    container.style.cursor = placementMode ? 'crosshair' : '';
    return () => { if (container) container.style.cursor = ''; };
  }, [container, placementMode]);

  return null; // No DOM output — all visual rendering is in HotspotOverlays
}
