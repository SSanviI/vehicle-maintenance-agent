import React, { useState, useEffect } from 'react';
import { Shield, Send, Edit2, Trash2, CheckCircle2, AlertCircle, Loader2, History, UserCheck } from 'lucide-react';

interface EmergencyContact {
  id?: string;
  contact_name: string;
  relationship?: string;
  contact_email: string;
}

interface LogItem {
  id: string;
  contact_email: string;
  subject: string;
  status: string;
  sent_at?: string;
  created_at: string;
}

interface EmergencyContactCardProps {
  isAuthenticated: boolean;
  conversationId: string | null;
  userEmail?: string;
  getAuthHeaders: () => Record<string, string>;
}

export const EmergencyContactCard: React.FC<EmergencyContactCardProps> = ({
  isAuthenticated,
  conversationId,
  userEmail,
  getAuthHeaders,
}) => {
  const [contact, setContact] = useState<EmergencyContact | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);

  // Form Inputs
  const [name, setName] = useState('');
  const [relationship, setRelationship] = useState('');
  const [email, setEmail] = useState('');

  // Status & Feedback Messages
  const [formMessage, setFormMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [sendResult, setSendResult] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Audit Logs
  const [historyLogs, setHistoryLogs] = useState<LogItem[]>([]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchContact();
      fetchHistory();
    }
  }, [isAuthenticated]);

  const fetchContact = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/emergency/contact', {
        credentials: 'include',
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        setContact(data);
        setName(data.contact_name || '');
        setRelationship(data.relationship || '');
        setEmail(data.contact_email || '');
        setIsEditing(false);
      } else if (res.status === 404) {
        setContact(null);
        setName('');
        setRelationship('');
        setEmail('');
        setIsEditing(true);
      }
    } catch (err) {
      console.error('Failed to fetch emergency contact:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/emergency/history', {
        credentials: 'include',
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        setHistoryLogs(data.logs || []);
      }
    } catch (err) {
      console.error('Failed to fetch emergency history:', err);
    }
  };

  const handleSaveContact = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim()) {
      setFormMessage({ type: 'error', text: 'Contact Name and Email Address are required.' });
      return;
    }

    setIsLoading(true);
    setFormMessage(null);

    try {
      const res = await fetch('/api/emergency/contact', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders(),
        },
        body: JSON.stringify({
          contact_name: name.trim(),
          relationship: relationship.trim() || undefined,
          contact_email: email.trim(),
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setContact(data);
        setIsEditing(false);
        setFormMessage({ type: 'success', text: '✓ Emergency contact saved successfully.' });
      } else {
        const errData = await res.json();
        setFormMessage({ type: 'error', text: errData.detail || 'Failed to save contact.' });
      }
    } catch (err) {
      setFormMessage({ type: 'error', text: 'Network error saving contact.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteContact = async () => {
    if (!window.confirm('Are you sure you want to delete this emergency contact?')) return;

    setIsLoading(true);
    setFormMessage(null);

    try {
      const res = await fetch('/api/emergency/contact', {
        method: 'DELETE',
        credentials: 'include',
        headers: getAuthHeaders(),
      });

      if (res.ok) {
        setContact(null);
        setName('');
        setRelationship('');
        setEmail('');
        setIsEditing(true);
        setFormMessage({ type: 'success', text: 'Emergency contact deleted.' });
      }
    } catch (err) {
      setFormMessage({ type: 'error', text: 'Failed to delete emergency contact.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendAlert = async () => {
    if (!contact && !email) {
      setSendResult({ type: 'error', text: 'Please save an emergency contact first.' });
      return;
    }

    setIsSending(true);
    setSendResult(null);

    try {
      const res = await fetch('/api/emergency/send', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeaders(),
        },
        body: JSON.stringify({
          recipient_email: contact?.contact_email || email,
          conversation_id: conversationId,
        }),
      });

      const data = await res.json();

      if (res.ok && data.success) {
        setSendResult({
          type: 'success',
          text: `✅ Emergency alert sent successfully. Sent to: ${data.recipient}`,
        });
        fetchHistory();
      } else {
        setSendResult({
          type: 'error',
          text: data.detail || 'Unable to send the emergency alert. Please try again.',
        });
      }
    } catch (err) {
      setSendResult({
        type: 'error',
        text: 'Unable to send the emergency alert. Please check connection and try again.',
      });
    } finally {
      setIsSending(false);
    }
  };

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="w-full max-w-3xl mx-auto my-4 space-y-4">
      {/* 🚨 Emergency Contact Card */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-rose-500/20 border border-rose-500/40 flex items-center justify-center text-rose-400">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-base tracking-wide flex items-center gap-2">
                🚨 Emergency Contact
              </h3>
              <p className="text-xs text-slate-400">
                {contact && !isEditing
                  ? 'Registered emergency contact for notification dispatches'
                  : 'Who should receive emergency alerts?'}
              </p>
            </div>
          </div>

          {contact && !isEditing && (
            <span className="px-2.5 py-1 text-[11px] font-bold rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
              <UserCheck className="w-3 h-3" /> Status: Saved
            </span>
          )}
        </div>

        {formMessage && (
          <div
            className={`p-3 rounded-xl text-xs flex items-center gap-2 ${
              formMessage.type === 'success'
                ? 'bg-emerald-950/60 border border-emerald-500/30 text-emerald-300'
                : 'bg-rose-950/60 border border-rose-500/30 text-rose-300'
            }`}
          >
            {formMessage.type === 'success' ? (
              <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
            )}
            <span>{formMessage.text}</span>
          </div>
        )}

        {/* Contact Form or Saved Details */}
        {!contact || isEditing ? (
          <form onSubmit={handleSaveContact} className="space-y-3 pt-1">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Contact Name *</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. John Doe"
                  required
                  className="w-full bg-slate-950 border border-slate-700 focus:border-rose-500 rounded-xl px-3.5 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-all"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Relationship</label>
                <input
                  type="text"
                  value={relationship}
                  onChange={(e) => setRelationship(e.target.value)}
                  placeholder="e.g. Parent / Spouse / Mechanic"
                  className="w-full bg-slate-950 border border-slate-700 focus:border-rose-500 rounded-xl px-3.5 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-all"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">Email Address *</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="e.g. parent@gmail.com"
                required
                className="w-full bg-slate-950 border border-slate-700 focus:border-rose-500 rounded-xl px-3.5 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-all"
              />
            </div>

            <div className="flex items-center gap-2 pt-2">
              <button
                type="submit"
                disabled={isLoading}
                className="px-5 py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-xl text-xs font-bold transition-all flex items-center gap-2 cursor-pointer shadow-md"
              >
                {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save Contact'}
              </button>

              {contact && (
                <button
                  type="button"
                  onClick={() => setIsEditing(false)}
                  className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-medium transition-colors cursor-pointer"
                >
                  Cancel
                </button>
              )}
            </div>
          </form>
        ) : (
          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="space-y-1">
              <div className="text-sm font-bold text-slate-100 flex items-center gap-2">
                {contact.contact_name}
                {contact.relationship && (
                  <span className="text-xs font-normal text-slate-400 font-mono">({contact.relationship})</span>
                )}
              </div>
              <div className="text-xs text-cyan-400 font-mono">{contact.contact_email}</div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsEditing(true)}
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <Edit2 className="w-3.5 h-3.5 text-cyan-400" /> Edit Contact
              </button>
              <button
                onClick={handleDeleteContact}
                className="px-3 py-1.5 bg-slate-800 hover:bg-rose-950/50 hover:text-rose-300 text-slate-400 rounded-xl text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5 text-rose-400" /> Delete Contact
              </button>
            </div>
          </div>
        )}
      </div>

      {/* 🚨 Emergency Alert Section (Visible when contact exists) */}
      {contact && (
        <div className="bg-slate-900/90 border border-rose-500/40 rounded-2xl p-5 shadow-xl space-y-4">
          <div>
            <h3 className="font-bold text-slate-100 text-base tracking-wide flex items-center gap-2">
              🚨 Emergency Alert
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Send an emergency alert to: <span className="text-rose-400 font-mono font-semibold">{contact.contact_email}</span>
            </p>
            {userEmail && (
              <p className="text-[11px] text-slate-500 mt-0.5">
                From authenticated account: <span className="text-slate-300 font-mono">{userEmail}</span>
              </p>
            )}
          </div>

          {sendResult && (
            <div
              className={`p-3 rounded-xl text-xs flex items-center gap-2 ${
                sendResult.type === 'success'
                  ? 'bg-emerald-950/60 border border-emerald-500/30 text-emerald-300'
                  : 'bg-rose-950/60 border border-rose-500/30 text-rose-300'
              }`}
            >
              {sendResult.type === 'success' ? (
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
              ) : (
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
              )}
              <span>{sendResult.text}</span>
            </div>
          )}

          <button
            onClick={handleSendAlert}
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
        </div>
      )}

      {/* Notification Audit History */}
      {historyLogs.length > 0 && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-300">
            <History className="w-4 h-4 text-cyan-400" />
            <span>Notification History</span>
          </div>

          <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
            {historyLogs.map((log) => (
              <div
                key={log.id}
                className="p-2.5 bg-slate-950/60 border border-slate-800/80 rounded-xl flex items-center justify-between text-xs font-mono"
              >
                <div>
                  <div className="text-slate-200">To: {log.contact_email}</div>
                  <div className="text-[10px] text-slate-400">{log.subject}</div>
                </div>

                <div className="text-right">
                  <span
                    className={`px-2 py-0.5 text-[10px] font-bold rounded-md ${
                      log.status === 'sent'
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-rose-950 text-rose-400 border border-rose-800'
                    }`}
                  >
                    {log.status.toUpperCase()}
                  </span>
                  <div className="text-[10px] text-slate-500 mt-0.5">
                    {new Date(log.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
