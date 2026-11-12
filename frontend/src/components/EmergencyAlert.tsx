import React, { useState } from 'react';
import { AlertTriangle, Send, X, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import type { EmergencyAlertData } from '../types';

interface EmergencyAlertProps {
  alert: EmergencyAlertData;
  conversationId: string | null;
  onClose: () => void;
}

export const EmergencyAlert: React.FC<EmergencyAlertProps> = ({
  alert,
  conversationId,
  onClose,
}) => {
  const [isSending, setIsSending] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string; recipient?: string } | null>(null);

  const handleSend = async () => {
    if (isSending) return;
    setIsSending(true);
    setResult(null);

    try {
      const response = await fetch('/api/emergency/send', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
        }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setResult({
          success: true,
          message: `Emergency email dispatched successfully to ${data.recipient}`,
          recipient: data.recipient,
        });
      } else {
        setResult({
          success: false,
          message: data.detail || 'Failed to send emergency alert email.',
        });
      }
    } catch (err) {
      setResult({
        success: false,
        message: 'Network error sending emergency alert.',
      });
    } finally {
      setIsSending(false);
    }
  };

  const severityColors = alert.severity === 'CRITICAL'
    ? 'bg-rose-500/20 text-rose-300 border-rose-500/30'
    : 'bg-amber-500/20 text-amber-300 border-amber-500/30';

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-rose-600/40 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-5 relative animate-in fade-in zoom-in duration-200">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-rose-500/20 border border-rose-500/40 flex items-center justify-center text-rose-400">
              <AlertTriangle className="w-6 h-6 animate-bounce" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-lg tracking-wide flex items-center gap-2">
                🚨 Emergency Detected
              </h3>
              <p className="text-xs text-slate-400">Validated vehicle safety condition requiring attention</p>
            </div>
          </div>

          <span className={`px-3 py-1 text-xs font-bold rounded-full border ${severityColors}`}>
            {alert.severity}
          </span>
        </div>

        {/* Conditions Breakdown */}
        <div className="p-4 bg-rose-950/30 border border-rose-900/50 rounded-xl space-y-3">
          {alert.conditions.length > 0 && (
            <div>
              <span className="text-xs font-semibold text-rose-300 block mb-1">Detected Conditions:</span>
              <ul className="list-disc list-inside text-xs text-slate-300 space-y-1">
                {alert.conditions.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </div>
          )}

          {alert.symptoms.length > 0 && (
            <div>
              <span className="text-xs font-semibold text-amber-300 block mb-1">Reported Symptoms:</span>
              <ul className="list-disc list-inside text-xs text-slate-300 space-y-1">
                {alert.symptoms.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <p className="text-xs text-slate-300 leading-relaxed">
          Would you like to notify your registered emergency contact via email?
          An alert containing your vehicle's condition will be sent through your Google account.
        </p>

        {/* Result Message */}
        {result && (
          <div
            className={`p-3 rounded-xl text-xs flex items-center gap-2 ${
              result.success
                ? 'bg-emerald-950/60 border border-emerald-500/30 text-emerald-300'
                : 'bg-rose-950/60 border border-rose-500/30 text-rose-300'
            }`}
          >
            {result.success ? (
              <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
            )}
            <span>{result.message}</span>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-medium transition-colors cursor-pointer"
          >
            {result?.success ? 'Close' : 'Cancel'}
          </button>

          {!result?.success && (
            <button
              onClick={handleSend}
              disabled={isSending}
              className={`px-5 py-2 rounded-xl text-xs font-bold text-white flex items-center gap-2 shadow-lg transition-all ${
                isSending
                  ? 'bg-rose-900 cursor-not-allowed opacity-60'
                  : 'bg-gradient-to-r from-rose-600 to-red-700 hover:from-rose-500 hover:to-red-600 shadow-rose-950/50 cursor-pointer'
              }`}
            >
              {isSending ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Sending Alert...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" /> Send Emergency Alert
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
