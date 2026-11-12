import React, { useState, useEffect } from 'react';
import { History, CheckCircle, AlertCircle, Clock, X } from 'lucide-react';
import type { EmailLog } from '../types';

interface EmergencyHistoryProps {
  isAuthenticated: boolean;
  isOpen: boolean;
  onClose: () => void;
}

export const EmergencyHistory: React.FC<EmergencyHistoryProps> = ({
  isAuthenticated,
  isOpen,
  onClose,
}) => {
  const [logs, setLogs] = useState<EmailLog[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen && isAuthenticated) {
      fetchHistory();
    }
  }, [isOpen, isAuthenticated]);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/emergency/history', { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs || []);
      }
    } catch (err) {
      console.error('Failed to fetch emergency history:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4 relative max-h-[80vh] flex flex-col">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
            <History className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-bold text-slate-100 text-base">Emergency Audit Log</h3>
            <p className="text-xs text-slate-400">History of dispatched emergency notifications</p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto divide-y divide-slate-800/80">
          {loading ? (
            <div className="py-8 text-center text-xs text-slate-400 italic">Loading audit logs...</div>
          ) : logs.length === 0 ? (
            <div className="py-8 text-center text-xs text-slate-400 italic">No emergency notifications sent yet</div>
          ) : (
            logs.map((log) => (
              <div key={log.id} className="py-3 space-y-1.5 text-xs">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-medium text-slate-200">
                    {log.status === 'sent' ? (
                      <span className="flex items-center gap-1 text-emerald-400">
                        <CheckCircle className="w-3.5 h-3.5" /> Sent
                      </span>
                    ) : log.status === 'failed' ? (
                      <span className="flex items-center gap-1 text-rose-400">
                        <AlertCircle className="w-3.5 h-3.5" /> Failed
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-amber-400">
                        <Clock className="w-3.5 h-3.5" /> Pending
                      </span>
                    )}
                    <span className="text-slate-400">•</span>
                    <span className="font-mono text-slate-300">{log.contact_email}</span>
                  </div>

                  <span className="text-[11px] text-slate-400 font-mono">
                    {new Date(log.created_at).toLocaleString()}
                  </span>
                </div>

                {log.prediction_summary && (
                  <div className="text-slate-400">
                    Summary: <span className="text-slate-300">{log.prediction_summary}</span>
                    {log.prediction !== undefined && log.prediction !== null && (
                      <span className="ml-2 font-mono text-cyan-400">
                        (Score: {(log.prediction * 100).toFixed(1)}%)
                      </span>
                    )}
                  </div>
                )}

                {log.email_id && (
                  <div className="text-[10px] text-slate-400 font-mono">
                    Gmail ID: <span className="text-slate-400">{log.email_id}</span>
                  </div>
                )}

                {log.error_message && (
                  <div className="text-[11px] text-rose-400 font-mono italic">
                    Error: {log.error_message}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
