import { useState } from 'react';
import type { Tour, TourStep, Hotspot } from '../types';

interface TourEditorProps {
  hotspots: Hotspot[];
  onSave: (tour: Tour) => void;
  onCancel: () => void;
}

export default function TourEditor({ hotspots, onSave, onCancel }: TourEditorProps) {
  const [name, setName] = useState('');
  const [steps, setSteps] = useState<TourStep[]>([]);

  const addStep = () => {
    setSteps([
      ...steps,
      {
        id: crypto.randomUUID(),
        label: `Step ${steps.length + 1}`,
        camera: { position: { x: 0, y: 1, z: 3 }, target: { x: 0, y: 0, z: 0 } },
        duration: 3,
      },
    ]);
  };

  const updateStep = (index: number, patch: Partial<TourStep>) => {
    const next = [...steps];
    next[index] = { ...next[index], ...patch };
    setSteps(next);
  };

  const removeStep = (index: number) => {
    setSteps(steps.filter((_, i) => i !== index));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || steps.length === 0) return;
    onSave({ id: crypto.randomUUID(), name: name.trim(), steps });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50" onClick={onCancel} />
      <form
        onSubmit={handleSubmit}
        className="relative bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-lg max-h-[85vh] overflow-y-auto"
      >
        <div className="p-5 border-b border-slate-700">
          <h3 className="text-lg font-semibold text-slate-100">Create Tour</h3>
        </div>
        <div className="p-5 space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Tour Name *</label>
            <input
              type="text" value={name} onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Welcome Walkthrough"
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-teal-500"
              required
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-300">Steps</label>
              <button type="button" onClick={addStep}
                className="text-teal-400 hover:text-teal-300 text-sm">+ Add Step</button>
            </div>
            {steps.map((step, i) => (
              <StepRow
                key={step.id}
                step={step}
                index={i}
                hotspots={hotspots}
                onChange={(patch) => updateStep(i, patch)}
                onRemove={() => removeStep(i)}
              />
            ))}
            {steps.length === 0 && (
              <p className="text-slate-500 text-sm text-center py-4">No steps yet. Add one to begin.</p>
            )}
          </div>
        </div>

        <div className="p-5 border-t border-slate-700 flex justify-end gap-2">
          <button type="button" onClick={onCancel}
            className="px-4 py-2 text-sm text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-600 transition-colors">
            Cancel
          </button>
          <button type="submit" disabled={!name.trim() || steps.length === 0}
            className="px-4 py-2 text-sm text-white bg-teal-600 hover:bg-teal-500 rounded-lg transition-colors disabled:opacity-40">
            Create Tour
          </button>
        </div>
      </form>
    </div>
  );
}

function StepRow({ step, index, hotspots, onChange, onRemove }: {
  step: TourStep; index: number; hotspots: Hotspot[];
  onChange: (p: Partial<TourStep>) => void; onRemove: () => void;
}) {
  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-3 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500 w-6">#{index + 1}</span>
        <input
          type="text" value={step.label}
          onChange={(e) => onChange({ label: e.target.value })}
          className="flex-1 bg-transparent text-sm text-slate-200 focus:outline-none"
        />
        <input
          type="number" value={step.duration ?? 3} min={1} max={60}
          onChange={(e) => onChange({ duration: Number(e.target.value) })}
          className="w-16 bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs text-slate-300 text-center"
          title="Duration (seconds)"
        />
        <span className="text-xs text-slate-500">s</span>
        <button type="button" onClick={onRemove}
          className="text-slate-500 hover:text-red-400 text-xs">✕</button>
      </div>
      {hotspots.length > 0 && (
        <select
          value={step.hotspotId || ''}
          onChange={(e) => onChange({ hotspotId: e.target.value || undefined })}
          className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs text-slate-300"
        >
          <option value="">No linked hotspot</option>
          {hotspots.map((h) => (
            <option key={h.id} value={h.id}>{h.label}</option>
          ))}
        </select>
      )}
    </div>
  );
}
