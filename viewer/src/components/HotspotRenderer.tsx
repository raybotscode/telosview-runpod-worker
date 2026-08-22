import { useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import type { Hotspot } from '../types';

interface HotspotRendererProps {
  hotspots: Hotspot[];
  scene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  renderer: THREE.WebGLRenderer | null;
  container: HTMLDivElement | null;
  placementMode?: boolean;
  onHotspotClick?: (hotspot: Hotspot) => void;
  onPlaceHotspot?: (position: { x: number; y: number; z: number }) => void;
  onHotspotHover?: (hotspot: Hotspot | null) => void;
}

const HOTSPOT_RADIUS = 0.08;
const HOTSPOT_COLOR = 0x14b8a6; // teal-500
const HOTSPOT_HOVER_COLOR = 0x2dd4bf; // teal-400

function createTextSprite(text: string, color: string = '#14b8a6'): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  const fontSize = 48;
  ctx.font = `bold ${fontSize}px sans-serif`;
  const metrics = ctx.measureText(text);
  const textWidth = metrics.width;

  canvas.width = textWidth + 20;
  canvas.height = fontSize + 16;

  // Background
  ctx.fillStyle = 'rgba(15, 23, 42, 0.85)'; // slate-900
  const radius = 8;
  ctx.beginPath();
  ctx.moveTo(radius, 0);
  ctx.lineTo(canvas.width - radius, 0);
  ctx.quadraticCurveTo(canvas.width, 0, canvas.width, radius);
  ctx.lineTo(canvas.width, canvas.height - radius);
  ctx.quadraticCurveTo(canvas.width, canvas.height, canvas.width - radius, canvas.height);
  ctx.lineTo(radius, canvas.height);
  ctx.quadraticCurveTo(0, canvas.height, 0, canvas.height - radius);
  ctx.lineTo(0, radius);
  ctx.quadraticCurveTo(0, 0, radius, 0);
  ctx.closePath();
  ctx.fill();

  // Border
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Text
  ctx.fillStyle = '#e2e8f0'; // slate-200
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 10, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  });

  const sprite = new THREE.Sprite(material);
  const aspect = canvas.width / canvas.height;
  sprite.scale.set(aspect * 0.3, 0.3, 1);
  return sprite;
}

function createHotspotMesh(hotspot: Hotspot): THREE.Group {
  const group = new THREE.Group();
  group.userData = { hotspotId: hotspot.id, type: 'hotspot' };

  // Main sphere
  const color = hotspot.color ? new THREE.Color(hotspot.color).getHex() : HOTSPOT_COLOR;
  const geometry = new THREE.SphereGeometry(HOTSPOT_RADIUS, 16, 16);
  const material = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.5,
    metalness: 0.3,
    roughness: 0.4,
  });
  const sphere = new THREE.Mesh(geometry, material);
  sphere.userData = { hotspotId: hotspot.id, type: 'hotspot' };
  group.add(sphere);

  // Outer ring (pulse effect base)
  const ringGeometry = new THREE.RingGeometry(HOTSPOT_RADIUS * 1.3, HOTSPOT_RADIUS * 1.6, 32);
  const ringMaterial = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.4,
    side: THREE.DoubleSide,
    depthTest: false,
  });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.userData = { type: 'ring' };
  group.add(ring);

  // Label sprite
  const sprite = createTextSprite(hotspot.label, hotspot.color || '#14b8a6');
  sprite.position.y = 0.25;
  sprite.userData = { type: 'label' };
  group.add(sprite);

  group.position.set(hotspot.position.x, hotspot.position.y, hotspot.position.z);

  return group;
}

export default function HotspotRenderer({
  hotspots,
  scene,
  camera,
  renderer,
  container,
  placementMode = false,
  onHotspotClick,
  onPlaceHotspot,
  onHotspotHover,
}: HotspotRendererProps) {
  const groupRef = useRef<THREE.Group | null>(null);
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());
  const hoveredRef = useRef<string | null>(null);

  // Build/rebuild hotspot group when hotspots change
  useEffect(() => {
    if (!scene) return;

    // Remove old group
    if (groupRef.current) {
      scene.remove(groupRef.current);
      groupRef.current.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else {
            child.material.dispose();
          }
        }
        if (child instanceof THREE.Sprite) {
          (child.material as THREE.SpriteMaterial).map?.dispose();
          (child.material as THREE.SpriteMaterial).dispose();
        }
      });
    }

    // Create new group
    const group = new THREE.Group();
    group.name = 'hotspots';
    hotspots.forEach((h) => {
      const mesh = createHotspotMesh(h);
      group.add(mesh);
    });
    scene.add(group);
    groupRef.current = group;

    return () => {
      if (groupRef.current && scene) {
        scene.remove(groupRef.current);
      }
    };
  }, [scene, hotspots]);

  // Animate rings (pulse)
  useEffect(() => {
    if (!scene) return;

    let disposed = false;
    const clock = new THREE.Clock();

    const animate = () => {
      if (disposed || !groupRef.current) return;
      requestAnimationFrame(animate);

      const t = clock.getElapsedTime();

      groupRef.current.children.forEach((child) => {
        if (!(child instanceof THREE.Group)) return;
        const ring = child.children.find((c) => c.userData.type === 'ring') as THREE.Mesh | undefined;
        if (ring && ring.material instanceof THREE.MeshBasicMaterial) {
          const scale = 1 + 0.2 * Math.sin(t * 3);
          ring.scale.set(scale, scale, 1);
          ring.material.opacity = 0.4 - 0.15 * Math.sin(t * 3);
        }

        // Billboard: make labels face camera
        const label = child.children.find((c) => c.userData.type === 'label');
        if (label && camera) {
          label.lookAt(camera.position);
        }
      });
    };
    animate();

    return () => { disposed = true; };
  }, [scene, camera]);

  // Raycaster for click and hover
  const getHotspotFromEvent = useCallback(
    (clientX: number, clientY: number): { hotspot: Hotspot; object: THREE.Object3D } | null => {
      if (!container || !camera || !groupRef.current) return null;

      const rect = container.getBoundingClientRect();
      mouseRef.current.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((clientY - rect.top) / rect.height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);

      // Collect all hotspot spheres
      const targets: THREE.Object3D[] = [];
      groupRef.current.traverse((child) => {
        if (child.userData.type === 'hotspot' && child instanceof THREE.Mesh) {
          targets.push(child);
        }
      });

      const intersects = raycasterRef.current.intersectObjects(targets, false);
      if (intersects.length > 0) {
        const hit = intersects[0].object;
        const hotspotId = hit.userData.hotspotId as string;
        const hotspot = hotspots.find((h) => h.id === hotspotId);
        if (hotspot) return { hotspot, object: hit };
      }
      return null;
    },
    [container, camera, hotspots]
  );

  // Click handler
  useEffect(() => {
    if (!container) return;

    const handleClick = (e: MouseEvent) => {
      if (placementMode && onPlaceHotspot) {
        // Place mode: raycast to scene depth
        if (!camera || !renderer) return;
        const rect = container.getBoundingClientRect();
        const mouse = new THREE.Vector2(
          ((e.clientX - rect.left) / rect.width) * 2 - 1,
          -((e.clientY - rect.top) / rect.height) * 2 + 1
        );
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, camera);

        // Try to intersect with scene objects (excluding hotspots)
        const sceneObjects: THREE.Object3D[] = [];
        scene?.traverse((child) => {
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
        return;
      }

      // Normal mode: check hotspot click
      const result = getHotspotFromEvent(e.clientX, e.clientY);
      if (result && onHotspotClick) {
        onHotspotClick(result.hotspot);
      }
    };

    container.addEventListener('click', handleClick);
    return () => container.removeEventListener('click', handleClick);
  }, [container, placementMode, camera, renderer, scene, getHotspotFromEvent, onHotspotClick, onPlaceHotspot]);

  // Hover handler
  useEffect(() => {
    if (!container) return;

    const handleMove = (e: MouseEvent) => {
      const result = getHotspotFromEvent(e.clientX, e.clientY);

      if (result) {
        const id = result.hotspot.id;
        if (hoveredRef.current !== id) {
          // Unhighlight previous
          if (hoveredRef.current) {
            setHotspotHighlight(hoveredRef.current, false);
          }
          hoveredRef.current = id;
          setHotspotHighlight(id, true);
          container.style.cursor = 'pointer';
          onHotspotHover?.(result.hotspot);
        }
      } else if (hoveredRef.current) {
        setHotspotHighlight(hoveredRef.current, false);
        hoveredRef.current = null;
        container.style.cursor = placementMode ? 'crosshair' : '';
        onHotspotHover?.(null);
      }
    };

    container.addEventListener('mousemove', handleMove);
    return () => container.removeEventListener('mousemove', handleMove);
  }, [container, placementMode, getHotspotFromEvent, onHotspotHover]);

  // Set cursor for placement mode
  useEffect(() => {
    if (!container) return;
    container.style.cursor = placementMode ? 'crosshair' : '';
    return () => { if (container) container.style.cursor = ''; };
  }, [container, placementMode]);

  function setHotspotHighlight(hotspotId: string, highlight: boolean) {
    if (!groupRef.current) return;
    groupRef.current.children.forEach((child) => {
      if (!(child instanceof THREE.Group)) return;
      if (child.userData.hotspotId !== hotspotId) return;

      const sphere = child.children.find((c) => c.userData.type === 'hotspot') as THREE.Mesh | undefined;
      if (sphere && sphere.material instanceof THREE.MeshStandardMaterial) {
        sphere.material.emissiveIntensity = highlight ? 1.0 : 0.5;
        if (highlight) {
          sphere.material.color.set(HOTSPOT_HOVER_COLOR);
          sphere.material.emissive.set(HOTSPOT_HOVER_COLOR);
        } else {
          sphere.material.color.set(sphere.material.emissive.getHex());
        }
      }

      // Scale up on hover
      const scale = highlight ? 1.3 : 1.0;
      child.scale.set(scale, scale, scale);
    });
  }

  return null; // This component only manages Three.js objects, no DOM
}
