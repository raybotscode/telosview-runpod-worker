import { useState } from 'react';
import type { Hotspot } from '../types';

interface HotspotEditorProps {
  editingHotspot: Hotspot | null;
  placementPosition: { x: number; y: number; z: number } | null;
  onSave: (hotspot: Hotspot) => void;
  onCancel: () => void;
  onDelete?: (id: string) => void;
}

export default function HotspotEditor({
  editingHotspot,
  placementPosition,
  onSave,
  onCancel,
  onDelete,
}: HotspotEditorProps) {
  const isNew = !editingHotspot && placementPosition !== null;
  const isEditing = editingHotspot !== null;

  if (!isNew && !isEditing) return null;

  const initial = editingHotspot || {
    id: crypto.randomUUID(),
    position: placementPosition!,
    label: '',
    color: '#14b8a6',
  };

  return (
    <HotspotForm
      key={initial.id}
      initial={initial}
      isNew={isNew}
      onSave={onSave}
      onCancel={onCancel}
      onDelete={onDelete}
    />
  );
}

function HotspotForm({
  initial,
  isNew,
  onSave,
  onCancel,
  onDelete,
}: {
  initial: Hotspot;
  isNew: boolean;
  onSave: (h: Hotspot) => void;
  onCancel: () => void;
  onDelete?: (id: string) => void;
}) {
  const [label, setLabel] = useState(initial.label);
  const [description, setDescription] = useState(initial.description || '');
  const [color, setColor] = useState(initial.color || '#14b8a6');
  const [panelTitle, setPanelTitle] = useState(initial.panelTitle || '');
  const [panelContent, setPanelContent] = useState(initial.panelContent || '');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!label.trim()) return;
    onSave({
      ...initial,
      label: label.trim(),
      description: description.trim() || undefined,
      color,
      panelTitle: panelTitle.trim() || undefined,
      panelContent: panelContent.trim() || undefined,
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50" onClick={onCancel} />
      <form
        onSubmit={handleSubmit}
        onPointerDown={(e) => e.stopPropagation()}
        className="relative bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-md max-h-[80vh] overflow-y-auto"
      >
        <div className="p-5 border-b border-slate-700">
          <h3 className="text-lg font-semibold text-slate-100">
            {isNew ? 'Add Hotspot' : 'Edit Hotspot'}
          </h3>
        </div>

        <div className="p-5 space-y-4">
          <Field label="Label *" value={label} onChange={setLabel} placeholder="e.g. Main Entrance" required autoFocus />
          <Field label="Description" value={description} onChange={setDescription} placeholder="Short description" />
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Color</label>
            <div className="flex items-center gap-3">
              <input type="color" value={color} onChange={(e) => setColor(e.target.value)}
                className="w-10 h-10 rounded cursor-pointer bg-transparent border border-slate-600" />
              <span className="text-slate-400 text-sm">{color}</span>
            </div>
          </div>
          <Field label="Panel Title" value={panelTitle} onChange={setPanelTitle} placeholder="Title shown in detail panel" />
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Panel Content</label>
            <textarea
              value={panelContent}
              onChange={(e) => setPanelContent(e.target.value)}
              placeholder="HTML or markdown content for the detail panel"
              rows={4}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-teal-500 resize-none"
            />
          </div>
        </div>

        <div className="p-5 border-t border-slate-700 flex items-center justify-between gap-3">
          <div>
            {!isNew && onDelete && (
              showDeleteConfirm ? (
                <div className="flex items-center gap-2">
                  <span className="text-red-400 text-sm">Delete?</span>
                  <button type="button" onClick={() => onDelete(initial.id)}
                    className="text-red-400 hover:text-red-300 text-sm font-medium px-2 py-1 rounded hover:bg-red-900/30">
                    Yes
                  </button>
                  <button type="button" onClick={() => setShowDeleteConfirm(false)}
                    className="text-slate-400 hover:text-slate-300 text-sm px-2 py-1 rounded hover:bg-slate-700/50">
                    No
                  </button>
                </div>
              ) : (
                <button type="button" onClick={() => setShowDeleteConfirm(true)}
                  className="text-red-400 hover:text-red-300 text-sm transition-colors">
                  Delete
                </button>
              )
            )}
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={onCancel}
              className="px-4 py-2 text-sm text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-600 transition-colors">
              Cancel
            </button>
            <button type="submit" disabled={!label.trim()}
              className="px-4 py-2 text-sm text-white bg-teal-600 hover:bg-teal-500 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
              {isNew ? 'Add' : 'Save'}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

function Field({ label, value, onChange, ...props }: {
  label: string; value: string; onChange: (v: string) => void;
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange'>) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-300 mb-1">{label}</label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-teal-500"
        {...props}
      />
    </div>
  );
}
