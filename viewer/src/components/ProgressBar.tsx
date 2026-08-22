interface ProgressBarProps {
  percent: number;
  label?: string;
  sublabel?: string;
  color?: string;
}

export default function ProgressBar({ percent, label, sublabel, color = 'teal' }: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, percent));
  const colorClasses: Record<string, string> = {
    teal: 'bg-teal-500',
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    amber: 'bg-amber-500',
  };

  return (
    <div className="w-full">
      {(label || sublabel) && (
        <div className="flex justify-between mb-1 text-sm">
          <span className="text-slate-300">{label}</span>
          <span className="text-slate-400">{sublabel}</span>
        </div>
      )}
      <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${colorClasses[color] || colorClasses.teal}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
      <div className="text-right text-xs text-slate-500 mt-1">{Math.round(clamped)}%</div>
    </div>
  );
}
