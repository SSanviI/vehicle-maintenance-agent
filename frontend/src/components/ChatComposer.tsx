import React, { useState, useRef, useEffect } from 'react';
import { ArrowUp, Loader2 } from 'lucide-react';

interface ChatComposerProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
}

export const ChatComposer: React.FC<ChatComposerProps> = ({ onSendMessage, isLoading }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 220)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim());
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  return (
    <div className="p-4 sm:p-6 bg-gradient-to-t from-[#050505] via-[#050505]/95 to-transparent sticky bottom-0 z-10">
      <div className="max-w-[960px] mx-auto relative">
        <div className="relative flex items-center bg-[#151515] hover:bg-[#181818] border border-[#2c2c2c] focus-within:border-slate-400 focus-within:ring-2 focus-within:ring-slate-400/20 rounded-2xl shadow-2xl transition-all min-h-[56px]">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe what's happening with your vehicle..."
            disabled={isLoading}
            className="w-full bg-transparent text-slate-100 text-[15px] sm:text-[16px] placeholder:text-slate-500 px-5 py-4 pr-14 resize-none outline-none max-h-52 overflow-y-auto leading-relaxed"
          />

          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
            className={`absolute right-3.5 p-2.5 rounded-xl transition-all duration-200 ${
              input.trim() && !isLoading
                ? 'bg-slate-100 text-slate-950 hover:bg-white shadow-md cursor-pointer scale-100'
                : 'bg-[#252525] text-slate-600 cursor-not-allowed scale-95'
            }`}
            title="Send Message"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin text-slate-950" />
            ) : (
              <ArrowUp className="w-4 h-4 font-bold" />
            )}
          </button>
        </div>

        <div className="mt-2 text-center text-xs text-slate-600 font-sans tracking-wide">
          AutoCare AI • Intelligent Vehicle Maintenance Assistant
        </div>
      </div>
    </div>
  );
};
