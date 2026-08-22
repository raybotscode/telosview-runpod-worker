import { useState, useRef, useEffect, useCallback } from 'react';
import { useVoiceInput } from '../hooks/useVoiceInput';
import VoiceButton from './VoiceButton';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  text: string;
  timestamp: number;
}

interface ChatPanelProps {
  onSendMessage: (text: string) => void;
  messages: ChatMessage[];
  isProcessing?: boolean;
  minimized?: boolean;
  onToggleMinimize?: () => void;
}

export default function ChatPanel({
  onSendMessage,
  messages,
  isProcessing = false,
  minimized = false,
  onToggleMinimize,
}: ChatPanelProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const voice = useVoiceInput();

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // When voice transcript arrives, send it
  useEffect(() => {
    if (voice.transcript && voice.transcript.length > 1) {
      onSendMessage(voice.transcript);
    }
  }, [voice.transcript, onSendMessage]);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text) return;
    onSendMessage(text);
    setInput('');
    inputRef.current?.focus();
  }, [input, onSendMessage]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  if (minimized) {
    return (
      <button
        onClick={onToggleMinimize}
        className="fixed bottom-4 right-4 z-30 w-12 h-12 rounded-full bg-slate-800 hover:bg-slate-700 border border-slate-600 text-slate-300 hover:text-white flex items-center justify-center transition-colors shadow-lg"
        title="Open chat"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
        {messages.length > 0 && (
          <span className="absolute -top-1 -right-1 w-5 h-5 bg-teal-500 text-white text-[10px] rounded-full flex items-center justify-center font-bold">
            {messages.length > 9 ? '9+' : messages.length}
          </span>
        )}
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-30 w-[360px] max-w-[calc(100vw-2rem)] bg-slate-900/95 backdrop-blur-md border border-slate-700 rounded-xl shadow-2xl shadow-black/50 flex flex-col max-h-[500px]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <span className="text-sm font-medium text-slate-200">Navigate</span>
          {voice.isListening && (
            <span className="text-xs text-red-400 animate-pulse">● Listening</span>
          )}
        </div>
        <button
          onClick={onToggleMinimize}
          className="text-slate-400 hover:text-white transition-colors p-1 rounded hover:bg-slate-700/50"
          title="Minimize"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-[120px] max-h-[300px]">
        {messages.length === 0 && (
          <div className="text-center text-slate-500 text-sm py-4">
            <p className="mb-1">Say or type where you want to go</p>
            <p className="text-xs text-slate-600">"Take me to the entrance" or "Where is the desk?"</p>
          </div>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] px-3 py-2 rounded-xl text-sm ${
                msg.role === 'user'
                  ? 'bg-teal-600/80 text-teal-50 rounded-br-sm'
                  : msg.role === 'system'
                    ? 'bg-slate-700/50 text-slate-400 italic text-xs'
                    : 'bg-slate-800 text-slate-200 rounded-bl-sm'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        {isProcessing && (
          <div className="flex justify-start">
            <div className="bg-slate-800 text-slate-400 px-3 py-2 rounded-xl rounded-bl-sm text-sm">
              <span className="inline-flex gap-1">
                <span className="animate-bounce" style={{ animationDelay: '0ms' }}>●</span>
                <span className="animate-bounce" style={{ animationDelay: '150ms' }}>●</span>
                <span className="animate-bounce" style={{ animationDelay: '300ms' }}>●</span>
              </span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Voice error */}
      {voice.error && (
        <div className="px-4 py-1.5 bg-red-900/30 border-t border-red-800/50">
          <p className="text-xs text-red-300">{voice.error}</p>
        </div>
      )}

      {/* Interim transcript */}
      {voice.interimTranscript && (
        <div className="px-4 py-1.5 bg-slate-800/50 border-t border-slate-700/50">
          <p className="text-xs text-slate-400 italic">{voice.interimTranscript}...</p>
        </div>
      )}

      {/* Input area */}
      <div className="flex items-center gap-2 px-3 py-3 border-t border-slate-700">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a command..."
          className="flex-1 bg-slate-800 text-slate-200 text-sm px-3 py-2 rounded-lg border border-slate-600 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500/50 placeholder-slate-500"
        />
        <VoiceButton
          isListening={voice.isListening}
          isSupported={voice.isSupported}
          onStartListening={voice.startListening}
          onStopListening={voice.stopListening}
          onToggleListening={voice.toggleListening}
          disabled={isProcessing}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isProcessing}
          className="w-10 h-10 rounded-lg bg-teal-600 hover:bg-teal-500 disabled:opacity-40 disabled:cursor-not-allowed text-white flex items-center justify-center transition-colors"
          title="Send"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </div>
    </div>
  );
}
