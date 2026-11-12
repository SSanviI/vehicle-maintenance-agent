import React from 'react';
import { LogOut, User as UserIcon, Loader2 } from 'lucide-react';
import type { AuthUser } from '../types';

interface AuthButtonProps {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading?: boolean;
  onLogout: () => void;
}

export const AuthButton: React.FC<AuthButtonProps> = ({
  user,
  isAuthenticated,
  isLoading = false,
  onLogout,
}) => {
  if (isLoading) {
    return (
      <div className="px-3.5 py-1.5 bg-slate-900 border border-slate-800 rounded-xl text-slate-400 text-xs font-medium flex items-center gap-2">
        <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
        <span>Checking authentication...</span>
      </div>
    );
  }

  if (isAuthenticated && user) {
    return (
      <div className="flex items-center gap-2.5 bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5 text-xs">
        <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold">
          {user.name ? user.name.charAt(0).toUpperCase() : <UserIcon className="w-3.5 h-3.5" />}
        </div>
        <div className="hidden md:block text-left truncate max-w-[130px]">
          <div className="font-semibold text-slate-200 truncate">{user.name || 'Google User'}</div>
          <div className="text-[10px] text-slate-400 truncate">{user.email}</div>
        </div>
        <button
          onClick={onLogout}
          title="Sign Out"
          className="p-1 text-slate-400 hover:text-rose-400 rounded hover:bg-slate-800 transition-colors ml-1 cursor-pointer"
        >
          <LogOut className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  return (
    <a
      href="/auth/login"
      className="px-3.5 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-700 hover:border-cyan-500/50 rounded-xl text-slate-200 text-xs font-medium flex items-center gap-2 transition-all cursor-pointer shadow-sm group"
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
  );
};
