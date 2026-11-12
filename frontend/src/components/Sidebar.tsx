import React from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  Car,
  Shield,
  ChevronLeft,
  User as UserIcon,
  LogOut,
  CheckCircle2,
} from 'lucide-react';
import type { Conversation, AuthUser } from '../types';

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: string) => void;
  isOpen: boolean;
  onToggleSidebar: () => void;
  isAuthenticated: boolean;
  user: AuthUser | null;
  onOpenAccount: () => void;
  onLogout: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  isOpen,
  onToggleSidebar,
  isAuthenticated,
  user,
  onOpenAccount,
  onLogout,
}) => {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-30 sm:hidden"
          onClick={onToggleSidebar}
        />
      )}

      <aside
        className={`fixed sm:static inset-y-0 left-0 z-40 w-64 sm:w-72 bg-[#0c0c0c] border-r border-[#1e1e1e] flex flex-col transition-transform duration-300 ease-in-out select-none ${
          isOpen ? 'translate-x-0' : '-translate-x-full sm:translate-x-0'
        }`}
      >
        {/* Top Header Branding */}
        <div className="p-4 flex items-center justify-between border-b border-[#1c1c1c]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-white shadow-md">
              <Car className="w-4 h-4" />
            </div>
            <div>
              <h1 className="font-bold text-slate-100 text-sm sm:text-base tracking-tight leading-none">
                AutoCare AI
              </h1>
              <span className="text-[11px] text-cyan-400 font-medium">
                Vehicle Maintenance Agent
              </span>
            </div>
          </div>

          <button
            onClick={onToggleSidebar}
            className="sm:hidden p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-[#1a1a1a] transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
        </div>

        {/* New Assessment Action Button */}
        <div className="p-3">
          <button
            onClick={() => {
              onNewChat();
              if (isOpen) onToggleSidebar();
            }}
            className="w-full py-2.5 px-3.5 bg-[#171717] hover:bg-[#202020] border border-[#2a2a2a] hover:border-cyan-500/40 rounded-xl text-slate-100 text-sm font-semibold flex items-center justify-start gap-2.5 transition-all shadow-sm group cursor-pointer"
          >
            <Plus className="w-4 h-4 text-cyan-400 group-hover:scale-110 transition-transform" />
            <span>New Assessment</span>
          </button>
        </div>

        {/* Recent Conversations List */}
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
          <div className="px-3 py-1.5 text-[11px] font-semibold text-slate-400 tracking-wider uppercase">
            Recent Assessments
          </div>

          {conversations.length === 0 ? (
            <div className="px-3 py-6 text-center text-xs text-slate-400 italic">
              No previous assessments yet
            </div>
          ) : (
            conversations.map((conv) => {
              const isActive = conv.id === activeId;
              return (
                <div
                  key={conv.id}
                  onClick={() => {
                    onSelectConversation(conv.id);
                    if (isOpen) onToggleSidebar();
                  }}
                  className={`group relative flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-xs sm:text-sm transition-all cursor-pointer ${
                    isActive
                      ? 'bg-[#1e1e1e] text-white font-medium border border-[#303030]'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#151515]'
                  }`}
                >
                  <MessageSquare
                    className={`w-4 h-4 flex-shrink-0 ${
                      isActive ? 'text-cyan-400' : 'text-slate-400'
                    }`}
                  />
                  <span className="truncate flex-1">{conv.title}</span>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation(conv.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-rose-400 rounded transition-opacity"
                    title="Delete assessment"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Bottom Section: Account & Emergency Controls */}
        <div className="p-3 border-t border-[#1c1c1c] bg-[#090909] space-y-2">
          <button
            onClick={() => {
              onOpenAccount();
              if (isOpen) onToggleSidebar();
            }}
            className="w-full py-2 px-3 bg-[#141414] hover:bg-[#1c1c1c] border border-[#252525] hover:border-[#353535] rounded-xl text-slate-200 text-xs sm:text-sm font-medium flex items-center justify-between transition-all cursor-pointer group"
          >
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-cyan-400 group-hover:text-cyan-300" />
              <span>Account & Emergency</span>
            </div>
            {isAuthenticated && (
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            )}
          </button>

          {/* User Profile Bar */}
          {isAuthenticated && user ? (
            <div className="flex items-center justify-between p-2 rounded-xl bg-[#141414] border border-[#252525] text-xs">
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold text-xs flex-shrink-0">
                  {user.name ? user.name.charAt(0).toUpperCase() : <UserIcon className="w-3.5 h-3.5" />}
                </div>
                <div className="min-w-0 truncate">
                  <div className="font-semibold text-slate-200 truncate leading-tight flex items-center gap-1">
                    <span className="truncate">{user.name || 'Google User'}</span>
                    <CheckCircle2 className="w-3 h-3 text-emerald-400 flex-shrink-0" />
                  </div>
                  <div className="text-[11px] text-slate-400 font-mono truncate">
                    {user.email}
                  </div>
                </div>
              </div>

              <button
                onClick={onLogout}
                title="Sign Out"
                className="p-1.5 text-slate-400 hover:text-rose-400 rounded-lg hover:bg-[#202020] transition-colors ml-1 cursor-pointer flex-shrink-0"
              >
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </div>
          ) : (
            <a
              href="/auth/login"
              className="w-full py-2 px-3 bg-[#141414] hover:bg-[#1c1c1c] border border-[#252525] hover:border-cyan-500/40 rounded-xl text-slate-300 hover:text-white text-xs sm:text-sm font-medium flex items-center justify-center gap-2 transition-all cursor-pointer"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                />
              </svg>
              <span>Sign in with Google</span>
            </a>
          )}
        </div>
      </aside>
    </>
  );
};
