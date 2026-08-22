import { Link } from 'react-router-dom';
import type { Project, ProjectStatus } from '../types';

interface ProjectCardProps {
  project: Project;
  onDelete: (id: string) => void;
}

const statusColors: Record<ProjectStatus, string> = {
  created: 'bg-slate-600 text-slate-200',
  extracting: 'bg-amber-600 text-amber-100',
  extracted: 'bg-blue-600 text-blue-100',
  processing: 'bg-purple-600 text-purple-100',
  complete: 'bg-green-600 text-green-100',
  error: 'bg-red-600 text-red-100',
};

export default function ProjectCard({ project, onDelete }: ProjectCardProps) {
  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (confirm(`Delete "${project.name}"? This cannot be undone.`)) {
      onDelete(project.id);
    }
  };

  const linkTo = project.status === 'complete' ? `/view/${project.id}` : `/project/${project.id}`;

  return (
    <Link
      to={linkTo}
      className="block bg-slate-800 rounded-lg border border-slate-700 hover:border-teal-500/50 transition-all hover:shadow-lg hover:shadow-teal-500/10 overflow-hidden group"
    >
      <div className="aspect-video bg-slate-900 flex items-center justify-center relative overflow-hidden">
        {project.thumbnail_url ? (
          <img
            src={project.thumbnail_url}
            alt={project.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="text-slate-600 text-4xl">🎬</div>
        )}
        <button
          onClick={handleDelete}
          className="absolute top-2 right-2 bg-red-600/80 hover:bg-red-500 text-white rounded-full w-7 h-7 flex items-center justify-center text-sm opacity-0 group-hover:opacity-100 transition-opacity"
          title="Delete project"
        >
          ✕
        </button>
      </div>
      <div className="p-3">
        <h3 className="font-medium text-slate-100 truncate">{project.name}</h3>
        <div className="flex items-center justify-between mt-2">
          <span className={`text-xs px-2 py-0.5 rounded-full ${statusColors[project.status]}`}>
            {project.status}
          </span>
          <span className="text-xs text-slate-500">
            {new Date(project.created_at).toLocaleDateString()}
          </span>
        </div>
      </div>
    </Link>
  );
}
