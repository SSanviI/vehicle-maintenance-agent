import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Bot, User } from 'lucide-react';
import type { Message } from '../types';
import { PredictionCard } from './PredictionCard';

interface ChatMessageProps {
  message: Message;
  onQuickAction?: (message: string) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onQuickAction }) => {
  const isUser = message.role === 'user';
  const showThinkingIndicator = message.isStreaming && !message.content;

  return (
    <div
      className={`py-6 sm:py-7 px-4 sm:px-6 md:px-8 transition-colors ${
        isUser ? 'bg-[#050505]' : 'bg-[#050505]'
      }`}
    >
      <div className="max-w-[960px] mx-auto flex gap-4 sm:gap-5">
        {/* Avatar */}
        <div className="flex-shrink-0 pt-0.5">
          {isUser ? (
            <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-[#222222] border border-[#333333] flex items-center justify-center text-slate-200 shadow-md">
              <User className="w-4 h-4 sm:w-5 sm:h-5" />
            </div>
          ) : (
            <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-tr from-cyan-600/30 via-slate-800 to-blue-600/30 border border-cyan-500/30 flex items-center justify-center text-cyan-300 shadow-md">
              <Bot className="w-4.5 h-4.5 sm:w-5 sm:h-5" />
            </div>
          )}
        </div>

        {/* Message Body & Result Card */}
        <div className="flex-1 min-w-0 space-y-2.5 max-w-[850px]">
          {/* Header row: Sender name & Live indicator */}
          <div className="flex items-center gap-2.5">
            <span className="text-sm font-semibold tracking-wide text-slate-200">
              {isUser ? 'You' : 'AutoCare AI'}
            </span>

            {message.isStreaming && (
              <span className="flex items-center gap-1.5 text-xs text-cyan-400 font-medium">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </span>
            )}
          </div>

          {/* Thinking placeholder if no content yet */}
          {showThinkingIndicator && (
            <div className="flex items-center gap-2 text-sm text-slate-400 py-1">
              <span>AutoCare AI is thinking</span>
              <span className="inline-flex gap-1 items-center">
                <span className="typing-dot text-cyan-400" />
                <span className="typing-dot text-cyan-400" />
                <span className="typing-dot text-cyan-400" />
              </span>
            </div>
          )}

          {/* Markdown Content */}
          {message.content && (
            <div className="text-[15px] sm:text-[16px] text-slate-100 leading-[1.65] sm:leading-[1.7] space-y-3 prose prose-invert max-w-none prose-p:leading-relaxed prose-p:my-1.5 prose-strong:text-white prose-strong:font-semibold prose-pre:bg-[#111111] prose-pre:border prose-pre:border-[#282828] prose-pre:rounded-xl">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}

          {/* Embedded Maintenance Risk Result Card */}
          {message.prediction && (
            <PredictionCard
              prediction={message.prediction}
              explanation={message.explanation}
              onQuickAction={onQuickAction}
            />
          )}
        </div>
      </div>
    </div>
  );
};
