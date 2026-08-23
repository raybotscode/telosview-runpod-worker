import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { getProject, uploadVideo, processSplat, connectSSE } from '../api/client';
import type { Project, ProcessingProgress, ExtractionProgress } from '../types';
import ProgressBar from '../components/ProgressBar';
import { formatDateTime } from '../lib/format';

export default function UploadPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const [extraction, setExtraction] = useState<ExtractionProgress | null>(null);
  const [processing, setProcessing] = useState<ProcessingProgress | null>(null);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const loadProject = useCallback(async () => {
    if (!id) return;
    setLoadError(null);
    try {
      const p = await getProject(id);
      setProject(p);
      return p;
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Failed to load project');
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    loadProject();
    return () => { esRef.current?.close(); };
  }, [loadProject]);

  // Connect SSE for extracting/processing states
  useEffect(() => {
    if (!id || !project) return;
    if (project.status === 'extracting') {
      esRef.current?.close();
      esRef.current = connectSSE(id, (data) => {
        if (data.type === 'extraction') {
          setExtraction(data as unknown as ExtractionProgress);
        }
        if (data.type === 'extraction_complete') {
          setExtraction(null);
          loadProject();
        }
      });
    } else if (project.status === 'processing') {
      esRef.current?.close();
      esRef.current = connectSSE(id, (data) => {
        if (data.type === 'progress') {
          setProcessing(data as unknown as ProcessingProgress);
        }
        if (data.type === 'complete') {
          setProcessing(null);
          loadProject();
        }
        if (data.type === 'error') {
          setProcessingError(String(data.message || 'Processing failed'));
          loadProject();
        }
      });
    }
    return () => { esRef.current?.close(); };
  }, [id, project?.status, loadProject]);

  const handleUpload = async (file: File) => {
    if (!id) return;
    setUploading(true);
    setUploadPct(0);
    try {
      const updated = await uploadVideo(id, file, setUploadPct);
      setProject(updated);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  const handleProcess = async () => {
    if (!id) return;
    setProcessingError(null);
    try {
      await processSplat(id);
      const p = await loadProject();
      if (p) setProject(p);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start processing');
    }
  };

  if (loading) {
    return <div className="text-center py-20 text-slate-500">Loading...</div>;
  }

  if (!project) {
    return (
      <div className="text-center py-20">
        <p className="text-red-400">{loadError || 'Project not found'}</p>
        <Link to="/" className="text-teal-400 hover:underline mt-4 inline-block">← Back to Dashboard</Link>
      </div>
    );
  }

  return (
    <div className="py-6 max-w-2xl mx-auto">
      <Link to="/" className="text-teal-400 hover:text-teal-300 text-sm mb-4 inline-block">
        ← Dashboard
      </Link>

      <h1 className="text-2xl font-bold text-slate-100 mb-1">{project.name}</h1>
      <p className="text-slate-500 text-sm mb-8">
        Created {formatDateTime(project.created_at)}
      </p>

      {/* Status: created — show upload zone */}
      {project.status === 'created' && !uploading && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer ${
            dragOver ? 'border-teal-400 bg-teal-500/10' : 'border-slate-600 hover:border-slate-500'
          }`}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <div className="text-4xl mb-3">📹</div>
          <p className="text-slate-300 mb-1">Drop a video file here</p>
          <p className="text-slate-500 text-sm">or click to browse</p>
          <input
            id="file-input"
            type="file"
            accept="video/*"
            onChange={handleFileInput}
            className="hidden"
          />
        </div>
      )}

      {/* Uploading */}
      {uploading && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-slate-200 font-medium mb-3">Uploading video...</h3>
          <ProgressBar percent={uploadPct} label="Upload" sublabel={`${uploadPct}%`} />
        </div>
      )}

      {/* Extracting frames */}
      {project.status === 'extracting' && extraction && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-slate-200 font-medium mb-3">Extracting frames...</h3>
          <ProgressBar
            percent={extraction.percent}
            label={`Frame ${extraction.frame} / ${extraction.total_frames}`}
            sublabel={`${Math.round(extraction.percent)}%`}
            color="blue"
          />
        </div>
      )}

      {/* Extracted — ready to process */}
      {project.status === 'extracted' && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 text-center">
          <div className="text-3xl mb-3">✅</div>
          <h3 className="text-slate-200 font-medium mb-2">Frames extracted</h3>
          <p className="text-slate-400 text-sm mb-4">Ready to create 3D Gaussian splat</p>
          <button
            onClick={handleProcess}
            className="bg-teal-600 hover:bg-teal-500 text-white px-6 py-2.5 rounded-lg font-medium transition-colors"
          >
            Process Splat
          </button>
        </div>
      )}

      {/* Processing */}
      {project.status === 'processing' && processing && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-slate-200 font-medium mb-4">Processing splat...</h3>
          <ProgressBar
            percent={processing.percent}
            label={processing.stage}
            sublabel={`Iteration ${processing.iteration}`}
            color="purple"
          />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
            <Stat label="Splats" value={processing.splats.toLocaleString()} />
            <Stat label="Iter/s" value={processing.iter_per_sec.toFixed(1)} />
            <Stat label="PSNR" value={processing.psnr.toFixed(2)} />
            <Stat label="Progress" value={`${Math.round(processing.percent)}%`} />
          </div>
        </div>
      )}

      {/* Processing without live data */}
      {project.status === 'processing' && !processing && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 text-center">
          <div className="animate-spin w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-3" />
          <p className="text-slate-300">Processing in progress...</p>
          <p className="text-slate-500 text-sm mt-1">Waiting for updates</p>
        </div>
      )}

      {/* Complete */}
      {project.status === 'complete' && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 text-center">
          <div className="text-3xl mb-3">🎉</div>
          <h3 className="text-slate-200 font-medium mb-2">Splat ready!</h3>
          {project.splat_count && (
            <p className="text-slate-400 text-sm mb-4">{project.splat_count.toLocaleString()} splats</p>
          )}
          <button
            onClick={() => navigate(`/view/${project.id}`)}
            className="bg-teal-600 hover:bg-teal-500 text-white px-6 py-2.5 rounded-lg font-medium transition-colors"
          >
            View Splat →
          </button>
        </div>
      )}

      {/* Error */}
      {project.status === 'error' && (
        <div className="bg-red-900/30 rounded-xl p-6 border border-red-800 text-center">
          <div className="text-3xl mb-3">❌</div>
          <h3 className="text-red-300 font-medium mb-2">Processing failed</h3>
          <p className="text-red-400 text-sm">{project.error || processingError || 'Unknown error'}</p>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-900 rounded-lg p-3 text-center">
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className="text-sm font-mono text-slate-200">{value}</div>
    </div>
  );
}
