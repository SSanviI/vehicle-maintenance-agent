import React from 'react';
import { Menu, Plus, Shield } from 'lucide-react';

interface HeaderProps {
  title: string;
  onOpenAccount: () => void;
  onToggleSidebar: () => void;
  onNewChat: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  onOpenAccount,
  onToggleSidebar,
  onNewChat,
}) => {
  return (
    <header className="sticky top-0 z-20 bg-[#080808]/90 backdrop-blur-md border-b border-[#1c1c1c] px-4 sm:px-6 py-3.5 flex items-center justify-between">
      <div className="flex items-center gap-3 min-w-0">
        <button
          onClick={onToggleSidebar}
          className="p-2 text-slate-400 hover:text-slate-200 rounded-xl hover:bg-[#181818] border border-[#262626] sm:hidden cursor-pointer"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="min-w-0">
          <h2 className="text-sm sm:text-base font-semibold text-slate-100 truncate max-w-xs sm:max-w-md">
            {title}
          </h2>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              AI Maintenance Agent Active
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onOpenAccount}
          title="Account & Emergency Configuration"
          className="p-2 sm:px-3.5 sm:py-2 bg-[#141414] hover:bg-[#1e1e1e] border border-[#262626] rounded-xl text-slate-300 hover:text-white text-xs sm:text-sm font-medium flex items-center gap-2 transition-all cursor-pointer"
        >
          <Shield className="w-4 h-4 text-cyan-400" />
          <span className="hidden sm:inline">Account & Emergency</span>
        </button>

        <button
          onClick={onNewChat}
          title="New Assessment"
          className="p-2 sm:px-3.5 sm:py-2 bg-[#141414] hover:bg-[#1e1e1e] border border-[#262626] rounded-xl text-slate-300 hover:text-white text-xs sm:text-sm font-medium flex items-center gap-1.5 transition-all cursor-pointer"
        >
          <Plus className="w-4 h-4 text-cyan-400" />
          <span className="hidden md:inline">New Assessment</span>
        </button>
      </div>
    </header>
  );
};
