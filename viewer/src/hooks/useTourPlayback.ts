import { useState, useRef, useCallback, useEffect } from 'react';
import * as THREE from 'three';
import type { Tour, TourStep } from '../types';

const CAMERA_LERP = 0.09;

export interface TourPlaybackState {
  currentStep: number;
  isPlaying: boolean;
  totalSteps: number;
  currentStepData: TourStep | null;
  progress: number; // 0-1 within current step
}

export interface TourPlaybackControls {
  play: () => void;
  pause: () => void;
  toggle: () => void;
  next: () => void;
  prev: () => void;
  goToStep: (index: number) => void;
}

export function useTourPlayback(
  tour: Tour | null,
  camera: THREE.PerspectiveCamera | null,
  controls: { target: THREE.Vector3 } | null
): [TourPlaybackState, TourPlaybackControls] {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  const animFrameRef = useRef<number>(0);
  const stepTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const targetPosRef = useRef(new THREE.Vector3());
  const targetLookRef = useRef(new THREE.Vector3());
  const startPosRef = useRef(new THREE.Vector3());
  const startLookRef = useRef(new THREE.Vector3());
  const lerpTRef = useRef(1); // 1 = arrived

  const totalSteps = tour?.steps.length ?? 0;
  const currentStepData = tour?.steps[currentStep] ?? null;

  // Animate camera toward target
  useEffect(() => {
    if (!camera || !controls) return;
    let disposed = false;

    const animate = () => {
      if (disposed) return;
      animFrameRef.current = requestAnimationFrame(animate);

      if (lerpTRef.current >= 1) return;

      // Ease-in-out
      lerpTRef.current = Math.min(1, lerpTRef.current + CAMERA_LERP);
      const t = easeInOut(lerpTRef.current);

      camera.position.lerpVectors(startPosRef.current, targetPosRef.current, t);

      const lookTarget = new THREE.Vector3().lerpVectors(
        startLookRef.current,
        targetLookRef.current,
        t
      );
      controls.target.copy(lookTarget);

      setProgress(lerpTRef.current);
    };
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [camera, controls]);

  const goToStepIndex = useCallback(
    (index: number) => {
      if (!tour || !camera || !controls) return;
      const clamped = Math.max(0, Math.min(index, totalSteps - 1));
      const step = tour.steps[clamped];
      if (!step) return;

      setCurrentStep(clamped);
      setProgress(0);

      // Capture current camera state as start
      startPosRef.current.copy(camera.position);
      startLookRef.current.copy(controls.target);

      // Set target
      targetPosRef.current.set(step.camera.position.x, step.camera.position.y, step.camera.position.z);
      targetLookRef.current.set(step.camera.target.x, step.camera.target.y, step.camera.target.z);

      // Start lerping
      lerpTRef.current = 0;
    },
    [tour, camera, controls, totalSteps]
  );

  // Auto-advance when playing
  useEffect(() => {
    if (!isPlaying || !tour) return;

    const step = tour.steps[currentStep];
    const duration = step?.duration ?? 3;

    // Clear any existing timer
    if (stepTimerRef.current) clearTimeout(stepTimerRef.current);

    stepTimerRef.current = setTimeout(() => {
      if (currentStep < totalSteps - 1) {
        goToStepIndex(currentStep + 1);
      } else {
        setIsPlaying(false);
      }
    }, duration * 1000);

    return () => {
      if (stepTimerRef.current) clearTimeout(stepTimerRef.current);
    };
  }, [isPlaying, currentStep, tour, totalSteps, goToStepIndex]);

  const play = useCallback(() => {
    if (!tour) return;
    setIsPlaying(true);
    // If at the end, restart
    if (currentStep >= totalSteps - 1) {
      goToStepIndex(0);
    }
  }, [tour, currentStep, totalSteps, goToStepIndex]);

  const pause = useCallback(() => {
    setIsPlaying(false);
    if (stepTimerRef.current) clearTimeout(stepTimerRef.current);
  }, []);

  const toggle = useCallback(() => {
    if (isPlaying) pause();
    else play();
  }, [isPlaying, play, pause]);

  const next = useCallback(() => {
    if (currentStep < totalSteps - 1) {
      goToStepIndex(currentStep + 1);
    }
  }, [currentStep, totalSteps, goToStepIndex]);

  const prev = useCallback(() => {
    if (currentStep > 0) {
      goToStepIndex(currentStep - 1);
    }
  }, [currentStep, goToStepIndex]);

  const goToStep = useCallback(
    (index: number) => {
      goToStepIndex(index);
    },
    [goToStepIndex]
  );

  // Initialize camera to first step when tour changes
  useEffect(() => {
    if (tour && tour.steps.length > 0) {
      setCurrentStep(0);
      setIsPlaying(false);
      goToStepIndex(0);
    }
  }, [tour?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const state: TourPlaybackState = {
    currentStep,
    isPlaying,
    totalSteps,
    currentStepData,
    progress,
  };

  const tourControls: TourPlaybackControls = {
    play,
    pause,
    toggle,
    next,
    prev,
    goToStep,
  };

  return [state, tourControls];
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}
