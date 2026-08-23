import { useRef, useEffect, useState, useImperativeHandle, forwardRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

export interface SplatViewerHandle {
  scene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  renderer: THREE.WebGLRenderer | null;
  controls: OrbitControls | null;
  container: HTMLDivElement | null;
}

interface SplatViewerProps {
  url: string;
  onLoad?: (info: { splatCount: number }) => void;
}

const SplatViewer = forwardRef<SplatViewerHandle, SplatViewerProps>(
  ({ url, onLoad }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Store Three.js objects in refs so the imperative handle can expose them
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);

    useImperativeHandle(ref, () => ({
      get scene() { return sceneRef.current; },
      get camera() { return cameraRef.current; },
      get renderer() { return rendererRef.current; },
      get controls() { return controlsRef.current; },
      get container() { return containerRef.current; },
    }));

    useEffect(() => {
      const container = containerRef.current;
      if (!container) return;

      // Renderer
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setClearColor(0x0f172a);
      container.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      // Scene
      const scene = new THREE.Scene();
      sceneRef.current = scene;

      // Camera
      const camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
      );
      camera.position.set(0, 1, 3);
      cameraRef.current = camera;

      // Controls
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.target.set(0, 0, 0);
      controlsRef.current = controls;

      // Spark renderer
      const spark = new SparkRenderer({ renderer });
      scene.add(spark);

      // Load splat
      let splatMesh: SplatMesh | null = null;
      let disposed = false;

      const loadSplat = async () => {
        try {
          setLoading(true);
          setError(null);
          splatMesh = new SplatMesh({ url });
          await splatMesh.initialized;
          if (disposed) return;
          // splat.js trains in OpenCV convention (x right, y down, z forward —
          // see splat-test/src/sfm/geometry.js). Three.js/Spark use y up, z back,
          // so without this 180° X rotation the scene renders upside down.
          splatMesh.rotation.x = Math.PI;
          spark.add(splatMesh);
          setLoading(false);
          onLoad?.({ splatCount: splatMesh.numSplats || 0 });
        } catch (err) {
          if (!disposed) {
            setError(err instanceof Error ? err.message : 'Failed to load splat');
            setLoading(false);
          }
        }
      };

      loadSplat();

      // Animation loop
      const animate = () => {
        if (disposed) return;
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };
      animate();

      // Resize handler
      const onResize = () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };
      window.addEventListener('resize', onResize);

      // Cleanup
      return () => {
        disposed = true;
        window.removeEventListener('resize', onResize);
        controls.dispose();
        renderer.dispose();
        sceneRef.current = null;
        cameraRef.current = null;
        rendererRef.current = null;
        controlsRef.current = null;
        if (container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
      };
    }, [url, onLoad]);

    return (
      <div className="relative w-full h-full">
        <div ref={containerRef} className="w-full h-full" />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80">
            <div className="text-center">
              <div className="animate-spin w-10 h-10 border-4 border-teal-500 border-t-transparent rounded-full mx-auto mb-3" />
              <p className="text-slate-400 text-sm">Loading splat...</p>
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80">
            <div className="text-center text-red-400">
              <p className="text-lg mb-2">⚠️</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        )}
      </div>
    );
  }
);

SplatViewer.displayName = 'SplatViewer';

export default SplatViewer;
