import type { Hotspot } from '../types';

interface HotspotPanelProps {
  hotspot: Hotspot | null;
  onClose: () => void;
  onEdit?: (hotspot: Hotspot) => void;
  onDelete?: (id: string) => void;
}

function getVideoEmbedUrl(url: string): string | null {
  // YouTube
  const ytMatch = url.match(
    /(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/
  );
  if (ytMatch) return `https://www.youtube.com/embed/${ytMatch[1]}`;

  // Vimeo
  const vimeoMatch = url.match(/vimeo\.com\/(\d+)/);
  if (vimeoMatch) return `https://player.vimeo.com/video/${vimeoMatch[1]}`;

  return null;
}

export default function HotspotPanel({ hotspot, onClose, onEdit, onDelete }: HotspotPanelProps) {
  if (!hotspot) return null;

  const embedUrl = hotspot.videoEmbed ? getVideoEmbedUrl(hotspot.videoEmbed) : null;

  return (
    <>
      {/* Backdrop for mobile */}
      <div
        className="fixed inset-0 bg-black/30 z-40 md:hidden"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={`
          fixed z-50 bg-slate-900/95 backdrop-blur-md border border-slate-700
          shadow-2xl shadow-black/50 overflow-y-auto
          transition-transform duration-300 ease-out
          /* Mobile: slide up from bottom */
          bottom-0 left-0 right-0 max-h-[70vh] rounded-t-2xl
          translate-y-0
          /* Desktop: slide in from right */
          md:bottom-auto md:left-auto md:right-0 md:top-0 md:max-h-none
          md:w-[400px] md:rounded-none md:rounded-l-2xl
          md:border-r-0 md:border-t-0 md:border-b-0
        `}
      >
        {/* Header */}
        <div className="sticky top-0 bg-slate-900/95 backdrop-blur-md border-b border-slate-700 px-5 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: hotspot.color || '#14b8a6' }}
            />
            <h2 className="text-lg font-semibold text-slate-100 truncate">
              {hotspot.panelTitle || hotspot.label}
            </h2>
          </div>
          <div className="flex items-center gap-1">
            {onEdit && (
              <button
                onClick={() => onEdit(hotspot)}
                className="text-slate-400 hover:text-teal-400 transition-colors p-1.5 rounded hover:bg-slate-700/50"
                aria-label="Edit hotspot"
                title="Edit"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </button>
            )}
            {onDelete && (
              <button
                onClick={() => onDelete(hotspot.id)}
                className="text-slate-400 hover:text-red-400 transition-colors p-1.5 rounded hover:bg-slate-700/50"
                aria-label="Delete hotspot"
                title="Delete"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            )}
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-white transition-colors p-1 rounded hover:bg-slate-700/50"
              aria-label="Close panel"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-5 space-y-5">
          {/* Description */}
          {hotspot.description && (
            <p className="text-slate-300 text-sm leading-relaxed">
              {hotspot.description}
            </p>
          )}

          {/* Panel content (HTML/markdown) */}
          {hotspot.panelContent && (
            <div
              className="prose prose-invert prose-sm max-w-none text-slate-300
                [&_h1]:text-slate-100 [&_h2]:text-slate-100 [&_h3]:text-slate-100
                [&_a]:text-teal-400 [&_a]:no-underline hover:[&_a]:underline
                [&_img]:rounded-lg [&_img]:border [&_img]:border-slate-700"
              dangerouslySetInnerHTML={{ __html: hotspot.panelContent }}
            />
          )}

          {/* Images */}
          {hotspot.images && hotspot.images.length > 0 && (
            <div className="space-y-3">
              {hotspot.images.map((src, i) => (
                <img
                  key={i}
                  src={src}
                  alt={`${hotspot.label} image ${i + 1}`}
                  className="w-full rounded-lg border border-slate-700 object-cover"
                  loading="lazy"
                />
              ))}
            </div>
          )}

          {/* Video embed */}
          {embedUrl && (
            <div className="relative w-full pt-[56.25%] rounded-lg overflow-hidden border border-slate-700">
              <iframe
                src={embedUrl}
                className="absolute inset-0 w-full h-full"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                title={hotspot.label}
              />
            </div>
          )}

          {/* Linked scene */}
          {hotspot.linkedScene && (
            <a
              href={`/view/${hotspot.linkedScene}`}
              className="flex items-center gap-2 text-teal-400 hover:text-teal-300 text-sm transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              View linked scene →
            </a>
          )}
        </div>
      </div>
    </>
  );
}
