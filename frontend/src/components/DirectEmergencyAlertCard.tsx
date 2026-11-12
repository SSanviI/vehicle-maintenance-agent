import React, { useState } from 'react';
import { AlertTriangle, Send, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';

interface DirectEmergencyAlertCardProps {
  conversationId: string | null;
  userEmail?: string;
  getAuthHeaders: () => Record<string, string>;
}

export const DirectEmergencyAlertCard: React.FC<DirectEmergencyAlertCardProps> = ({
  conversationId,
  userEmail,
  getAuthHeaders,
}) => {
  const [recipientEmail, setRecipientEmail] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleSendAlert = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!recipientEmail.trim()) {
      setStatusMessage({ type: 'error', text: 'Please enter a valid recipient email address.' });
      return;
    }

    setIsSending(true);
    setStatusMessage(null);

    try {
      const response = await fetch('/api/emergency/send', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders(),
        },
        body: JSON.stringify({
          recipient_email: recipientEmail.trim(),
          conversation_id: conversationId,
        }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setStatusMessage({
          type: 'success',
          text: `Emergency alert sent successfully to ${data.recipient}!`,
        });
        setRecipientEmail('');
      } else {
        setStatusMessage({
          type: 'error',
          text: data.detail || 'Failed to send emergency alert.',
        });
      }
    } catch (err) {
      setStatusMessage({
        type: 'error',
        text: 'Network error sending emergency alert.',
      });
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="w-full bg-slate-900/90 border border-rose-500/40 rounded-2xl p-5 shadow-xl space-y-4 my-3">
      <div className="flex items-center gap-3 border-b border-slate-800 pb-3">
        <div className="w-9 h-9 rounded-xl bg-rose-500/20 border border-rose-500/40 flex items-center justify-center text-rose-400">
          <AlertTriangle className="w-5 h-5 animate-pulse" />
        </div>
        <div>
          <h3 className="font-bold text-slate-100 text-base tracking-wide flex items-center gap-2">
            🚨 Emergency Alert
          </h3>
          <p className="text-xs text-slate-400">
            Send an emergency notification to someone you trust.
          </p>
        </div>
      </div>

      <form onSubmit={handleSendAlert} className="space-y-3">
        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-1.5">
            Who should receive the emergency alert? (Recipient Email)
          </label>
          <input
            type="email"
            value={recipientEmail}
            onChange={(e) => setRecipientEmail(e.target.value)}
            placeholder="e.g. parent@gmail.com"
            required
            className="w-full bg-slate-950 border border-slate-700 focus:border-rose-500 focus:ring-1 focus:ring-rose-500 rounded-xl px-3.5 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-all"
          />
        </div>

        {userEmail && (
          <p className="text-[11px] text-slate-400">
            Sender account: <span className="text-cyan-400 font-mono">{userEmail}</span>
          </p>
        )}

        {statusMessage && (
          <div
            className={`p-3 rounded-xl text-xs flex items-center gap-2 ${
              statusMessage.type === 'success'
                ? 'bg-emerald-950/60 border border-emerald-500/30 text-emerald-300'
                : 'bg-rose-950/60 border border-rose-500/30 text-rose-300'
            }`}
          >
            {statusMessage.type === 'success' ? (
              <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
            )}
            <span>{statusMessage.text}</span>
          </div>
        )}

        <button
          type="submit"
          disabled={isSending}
          className={`w-full py-2.5 rounded-xl text-xs font-bold text-white flex items-center justify-center gap-2 shadow-lg transition-all cursor-pointer ${
            isSending
              ? 'bg-rose-900 opacity-60 cursor-not-allowed'
              : 'bg-gradient-to-r from-rose-600 to-red-700 hover:from-rose-500 hover:to-red-600 shadow-rose-950/50'
          }`}
        >
          {isSending ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" /> Dispatching Alert...
            </>
          ) : (
            <>
              <Send className="w-4 h-4" /> Send Emergency Alert
            </>
          )}
        </button>
      </form>
    </div>
  );
};
