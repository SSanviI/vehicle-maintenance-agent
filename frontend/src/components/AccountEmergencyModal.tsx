import React from 'react';
import { X, Shield } from 'lucide-react';
import { EmergencyContactConfig } from './EmergencyContactConfig';
import type { AuthUser } from '../types';

interface AccountEmergencyModalProps {
  isOpen: boolean;
  onClose: () => void;
  isAuthenticated: boolean;
  user: AuthUser | null;
  conversationId: string | null;
  onLogout: () => void;
}

export const AccountEmergencyModal: React.FC<AccountEmergencyModalProps> = ({
  isOpen,
  onClose,
  isAuthenticated,
  user,
  conversationId,
  onLogout,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-[#0f0f0f] border border-[#262626] rounded-3xl max-w-xl w-full p-5 sm:p-6 shadow-2xl space-y-4 relative my-8">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-1.5 text-slate-400 hover:text-white rounded-xl hover:bg-[#202020] transition-colors cursor-pointer"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Title */}
        <div className="flex items-center gap-3 pr-8">
          <div className="w-10 h-10 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-bold text-slate-100 text-lg">Account & Emergency Controls</h3>
            <p className="text-xs sm:text-sm text-slate-400">Manage Google authentication, emergency contact alerts, and notification history.</p>
          </div>
        </div>

        {/* Embedded Configuration Panel */}
        <div className="pt-2">
          <EmergencyContactConfig
            isAuthenticated={isAuthenticated}
            user={user}
            conversationId={conversationId}
            onLogout={onLogout}
          />
        </div>
      </div>
    </div>
  );
};
