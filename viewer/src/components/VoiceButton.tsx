import { useState, useRef, useCallback } from 'react';

interface VoiceButtonProps {
  isListening: boolean;
  isSupported: boolean;
  onStartListening: () => void;
  onStopListening: () => void;
  onToggleListening?: () => void;
  disabled?: boolean;
  className?: string;
}

export default function VoiceButton({
  isListening,
  isSupported,
  onStartListening,
  onStopListening,
  onToggleListening,
  disabled = false,
  className = '',
}: VoiceButtonProps) {
  const [isPressed, setIsPressed] = useState(false);
  const [showTooltip, setShowTooltip] = useState(true);
  const longPressTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isLongPressRef = useRef(false);

  // Hide tooltip after first interaction
  const hideTooltip = useCallback(() => {
    if (showTooltip) setShowTooltip(false);
  }, [showTooltip]);

  // Mouse/touch handlers for push-to-talk
  const handlePointerDown = useCallback(() => {
    if (!isSupported || disabled) return;
    hideTooltip();
    setIsPressed(true);
    isLongPressRef.current = false;

    // Start after a short delay to distinguish click from hold
    longPressTimerRef.current = setTimeout(() => {
      isLongPressRef.current = true;
      if (!isListening) {
        onStartListening();
      }
    }, 200);
  }, [isSupported, disabled, isListening, onStartListening, hideTooltip]);

  const handlePointerUp = useCallback(() => {
    if (!isSupported || disabled) return;
    setIsPressed(false);

    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }

    if (isLongPressRef.current) {
      // Was a long press — stop listening
      if (isListening) {
        onStopListening();
      }
    } else {
      // Was a click — toggle
      onToggleListening?.();
    }
    isLongPressRef.current = false;
  }, [isSupported, disabled, isListening, onStopListening, onToggleListening]);

  const handlePointerLeave = useCallback(() => {
    if (isPressed) {
      setIsPressed(false);
      if (longPressTimerRef.current) {
        clearTimeout(longPressTimerRef.current);
        longPressTimerRef.current = null;
      }
      if (isLongPressRef.current && isListening) {
        onStopListening();
      }
      isLongPressRef.current = false;
    }
  }, [isPressed, isListening, onStopListening]);

  if (!isSupported) {
    return (
      <div className={`relative group ${className}`}>
        <button
          disabled
          className="w-12 h-12 rounded-full bg-slate-700/50 flex items-center justify-center cursor-not-allowed opacity-50"
          title="Voice input not supported in this browser"
        >
          <MicIcon />
        </button>
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-slate-800 text-slate-400 text-xs rounded-lg whitespace-nowrap pointer-events-none">
          Use Chrome or Edge for voice
        </div>
      </div>
    );
  }

  return (
    <div className={`relative group ${className}`}>
      {/* Pulsing ring when listening */}
      {isListening && (
        <div className="absolute inset-0 rounded-full animate-ping bg-red-500/40 pointer-events-none" />
      )}
      {isListening && (
        <div className="absolute inset-[-4px] rounded-full border-2 border-red-500 animate-pulse pointer-events-none" />
      )}

      <button
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerLeave}
        onContextMenu={(e) => e.preventDefault()}
        disabled={disabled}
        className={`
          w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200
          ${isListening
            ? 'bg-red-600 hover:bg-red-700 text-white shadow-lg shadow-red-500/30'
            : isPressed
              ? 'bg-teal-600 text-white scale-95'
              : 'bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-600'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          select-none touch-none
        `}
        aria-label={isListening ? 'Stop listening' : 'Start voice input'}
      >
        {isListening ? (
          <ListeningIcon />
        ) : (
          <MicIcon />
        )}
      </button>

      {/* Tooltip */}
      {showTooltip && !isListening && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-slate-800 text-slate-300 text-xs rounded-lg whitespace-nowrap pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
          Hold to speak or click to toggle
        </div>
      )}

      {/* Listening indicator */}
      {isListening && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-red-900/90 text-red-200 text-xs rounded-lg whitespace-nowrap animate-pulse">
          Listening...
        </div>
      )}
    </div>
  );
}

function MicIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-14 0m14 0a7 7 0 00-14 0m14 0v1a7 7 0 01-14 0v-1m7 8v4m-4 0h8" />
    </svg>
  );
}

function ListeningIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-14 0m14 0a7 7 0 00-14 0m14 0v1a7 7 0 01-14 0v-1m7 8v4m-4 0h8" />
      <circle cx="12" cy="11" r="3" fill="currentColor" opacity={0.5} />
    </svg>
  );
}
