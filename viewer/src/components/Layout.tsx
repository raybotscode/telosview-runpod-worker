import { Outlet, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function Layout() {
  const { user, logout } = useAuth();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <nav className="bg-slate-900 border-b border-slate-800 px-4 py-3">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <Link to="/" className="text-xl font-bold text-teal-400 hover:text-teal-300">
            TelosView
          </Link>
          <div className="flex items-center gap-4">
            {user && (
              <span className="text-sm text-slate-400">{user.name || user.email}</span>
            )}
            <button
              onClick={logout}
              className="text-sm text-slate-500 hover:text-slate-300 transition-colors"
            >
              Logout
            </button>
            <span className="text-xs text-slate-600">v2</span>
          </div>
        </div>
      </nav>
      <main className="max-w-6xl mx-auto p-4">
        <Outlet />
      </main>
    </div>
  );
}
