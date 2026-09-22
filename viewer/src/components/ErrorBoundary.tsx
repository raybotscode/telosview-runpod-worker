import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('Unhandled application error:', error, info);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
          <div className="text-center max-w-lg">
            <p className="text-red-400 mb-2 text-lg font-medium">Something went wrong.</p>
            <p className="text-red-300/80 text-sm mb-4 font-mono break-all">
              {this.state.error?.message || String(this.state.error)}
            </p>
            <button
              onClick={() => { this.setState({ hasError: false, error: null }); }}
              className="text-teal-400 hover:underline mr-4"
            >
              Try again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="text-slate-400 hover:underline"
            >
              Reload the page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
