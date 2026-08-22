import type { TourPlaybackState, TourPlaybackControls } from '../hooks/useTourPlayback';

interface TourControlsProps {
  tourName: string;
  state: TourPlaybackState;
  controls: TourPlaybackControls;
  onClose: () => void;
}

export default function TourControls({
  tourName,
  state,
  controls,
  onClose,
}: TourControlsProps) {
  const { currentStep, isPlaying, totalSteps, currentStepData, progress } = state;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-30 bg-slate-900/95 backdrop-blur-md border-t border-slate-700">
      {/* Progress bar */}
      <div className="h-1 bg-slate-800">
        <div
          className="h-full bg-teal-500 transition-all duration-300"
          style={{ width: `${((currentStep + progress) / totalSteps) * 100}%` }}
        />
      </div>

      <div className="flex items-center gap-4 px-4 py-3 max-w-4xl mx-auto">
        {/* Tour name */}
        <div className="hidden sm:block min-w-0">
          <p className="text-xs text-slate-500 uppercase tracking-wider">Tour</p>
          <p className="text-sm text-slate-200 font-medium truncate">{tourName}</p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={controls.prev}
            disabled={currentStep === 0}
            className="p-2 text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            aria-label="Previous step"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>

          <button
            onClick={controls.toggle}
            className="p-3 bg-teal-600 hover:bg-teal-500 text-white rounded-full transition-colors"
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>

          <button
            onClick={controls.next}
            disabled={currentStep >= totalSteps - 1}
            className="p-2 text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            aria-label="Next step"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>

        {/* Step label */}
        <div className="flex-1 min-w-0 text-center">
          <p className="text-sm text-slate-200 truncate">
            {currentStepData?.label || `Step ${currentStep + 1}`}
          </p>
          <p className="text-xs text-slate-500">
            {currentStep + 1} / {totalSteps}
          </p>
        </div>

        {/* Progress dots */}
        <div className="hidden sm:flex items-center gap-1.5">
          {Array.from({ length: totalSteps }, (_, i) => (
            <button
              key={i}
              onClick={() => controls.goToStep(i)}
              className={`w-2 h-2 rounded-full transition-all ${
                i === currentStep
                  ? 'bg-teal-400 scale-125'
                  : i < currentStep
                    ? 'bg-teal-600'
                    : 'bg-slate-600 hover:bg-slate-500'
              }`}
              aria-label={`Go to step ${i + 1}`}
            />
          ))}
        </div>

        {/* Close */}
        <button
          onClick={onClose}
          className="p-2 text-slate-400 hover:text-white transition-colors"
          aria-label="Close tour"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}
