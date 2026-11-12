import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, MapPin, Lock, Phone, Mail, Loader2, Trash2, Edit2, Send } from 'lucide-react';
import type { AuthUser } from '../types';

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

interface EmergencyContactConfigProps {
  isAuthenticated: boolean;
  user: AuthUser | null;
  conversationId: string | null;
  onLogout: () => void;
}

export const EmergencyContactConfig: React.FC<EmergencyContactConfigProps> = ({
  isAuthenticated,
  user,
  conversationId,
  onLogout,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  // Contact state
  const [contact, setContact] = useState<EmergencyContact | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [isLoadingContact, setIsLoadingContact] = useState(false);
  const [isSavingContact, setIsSavingContact] = useState(false);

  // Form Inputs
  const [name, setName] = useState('');
  const [relationship, setRelationship] = useState('');
  const [email, setEmail] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [formSuccess, setFormSuccess] = useState<string | null>(null);

  // Alert Send State
  const [isSendingAlert, setIsSendingAlert] = useState(false);
  const [alertFeedback, setAlertFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  // Notification History
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Location State
  const [locationStatus, setLocationStatus] = useState<string>('');

  useEffect(() => {
    if (isAuthenticated) {
      fetchContact();
      fetchHistory();
      checkLocationPermission();
    } else {
      setContact(null);
      setLogs([]);
    }
  }, [isAuthenticated]);

  const checkLocationPermission = () => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const lat = pos.coords.latitude.toFixed(4);
          const lon = pos.coords.longitude.toFixed(4);
          setLocationStatus(`📍 Active (${lat}, ${lon})`);
        },
        () => {
          setLocationStatus('Sign in / enable browser permission for live location.');
        }
      );
    } else {
      setLocationStatus('Geolocation unavailable.');
    }
  };

  const fetchContact = async () => {
    setIsLoadingContact(true);
    try {
      const res = await fetch('/api/emergency/contact', {
        credentials: 'include',
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
      console.error('Error fetching emergency contact:', err);
    } finally {
      setIsLoadingContact(false);
    }
  };

  const fetchHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const res = await fetch('/api/emergency/history', {
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs || []);
      }
    } catch (err) {
      console.error('Error fetching history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const handleSaveContact = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim()) {
      setFormError('Contact Name and Email Address are required.');
      return;
    }

    setIsSavingContact(true);
    setFormError(null);
    setFormSuccess(null);

    try {
      const res = await fetch('/api/emergency/contact', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
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
        setFormSuccess('✓ Contact saved successfully.');
      } else {
        const errData = await res.json();
        setFormError(errData.detail || 'Failed to save contact.');
      }
    } catch (err) {
      setFormError('Network error saving contact.');
    } finally {
      setIsSavingContact(false);
    }
  };

  const handleDeleteContact = async () => {
    if (!window.confirm('Delete emergency contact?')) return;

    setIsLoadingContact(true);
    setFormError(null);

    try {
      const res = await fetch('/api/emergency/contact', {
        method: 'DELETE',
        credentials: 'include',
      });

      if (res.ok) {
        setContact(null);
        setName('');
        setRelationship('');
        setEmail('');
        setIsEditing(true);
        setFormSuccess('Contact deleted.');
      }
    } catch (err) {
      setFormError('Failed to delete contact.');
    } finally {
      setIsLoadingContact(false);
    }
  };

  const handleSendAlert = async () => {
    if (!contact && !email) {
      setAlertFeedback({ type: 'error', message: 'Please save an emergency contact first.' });
      return;
    }

    setIsSendingAlert(true);
    setAlertFeedback(null);

    try {
      const res = await fetch('/api/emergency/send', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          recipient_email: contact?.contact_email || email,
          conversation_id: conversationId,
        }),
      });

      const data = await res.json();

      if (res.ok && data.success) {
        setAlertFeedback({
          type: 'success',
          message: `✅ Emergency alert sent successfully to ${data.recipient}!`,
        });
        fetchHistory();
      } else {
        setAlertFeedback({
          type: 'error',
          message: data.detail || 'Unable to send emergency alert.',
        });
      }
    } catch (err) {
      setAlertFeedback({
        type: 'error',
        message: 'Network error sending emergency alert.',
      });
    } finally {
      setIsSendingAlert(false);
    }
  };

  const handleConnectGoogle = async () => {
    try {
      const res = await fetch('/auth/google', {
        method: 'POST',
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        if (data.authorization_url) {
          window.location.href = data.authorization_url;
          return;
        }
      }
    } catch (err) {
      console.warn('POST /auth/google fallback to GET /auth/login:', err);
    }
    window.location.href = '/auth/login';
  };

  return (
    <div className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3.5 shadow-xl text-slate-100 space-y-3 my-4">
      {/* Container Header with Accordion Toggle */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between cursor-pointer select-none py-0.5"
      >
        <div className="flex items-center gap-2 font-bold text-sm text-slate-100">
          <span>🚨 Emergency Contact Config</span>
        </div>
        <button className="text-slate-400 hover:text-slate-200 p-1">
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </button>
      </div>

      {isExpanded && (
        <div className="space-y-3 pt-1 border-t border-slate-800/80">
          {/* Card 1: 📍 Live Location */}
          <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3 space-y-1.5">
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-200">
              <MapPin className="w-3.5 h-3.5 text-cyan-400" />
              <span>Live Location</span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              {isAuthenticated
                ? locationStatus || 'Location permission enabled.'
                : 'Sign in to enable live location.'}
            </p>
          </div>

          {/* Card 2: 🔒 Google Account */}
          <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-200">
                <Lock className="w-3.5 h-3.5 text-cyan-400" />
                <span>Google Account</span>
              </div>
              {isAuthenticated && (
                <span className="text-[10px] font-bold text-emerald-400 bg-emerald-950/80 border border-emerald-800/60 px-2 py-0.5 rounded">
                  ✓ Connected
                </span>
              )}
            </div>

            {isAuthenticated && user ? (
              <div className="space-y-2 pt-0.5">
                <div className="text-xs">
                  <div className="font-semibold text-slate-200">{user.name || 'Google User'}</div>
                  <div className="text-[10px] text-slate-400 font-mono">{user.email}</div>
                </div>
                <button
                  onClick={onLogout}
                  className="px-3 py-1 bg-slate-900 hover:bg-rose-950/60 hover:text-rose-300 border border-slate-700 rounded text-[11px] font-medium text-slate-300 transition-colors cursor-pointer"
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-[11px] text-slate-400">
                  Sign in to grant Gmail send permission.
                </p>
                <button
                  type="button"
                  onClick={handleConnectGoogle}
                  className="inline-flex items-center gap-2 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-700 hover:border-cyan-500/50 rounded-lg text-xs font-medium text-slate-200 transition-all cursor-pointer shadow-sm group"
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
                  <span>Connect Google Account</span>
                </button>
              </div>
            )}
          </div>

          {/* Card 3: 📞 Emergency Contact */}
          <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3 space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-200">
                <Phone className="w-3.5 h-3.5 text-rose-400" />
                <span>Emergency Contact</span>
              </div>
              {contact && !isEditing && (
                <span className="text-[10px] font-bold text-emerald-400">
                  ✓ Contact saved
                </span>
              )}
            </div>

            {!isAuthenticated ? (
              <p className="text-[11px] text-slate-400">
                Please sign in with Google first.
              </p>
            ) : isLoadingContact ? (
              <div className="flex items-center gap-2 text-xs text-slate-400 py-1">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                <span>Loading contact details...</span>
              </div>
            ) : !contact || isEditing ? (
              <form onSubmit={handleSaveContact} className="space-y-2 pt-0.5">
                <p className="text-[11px] text-slate-400">
                  Add an emergency contact who should receive alerts.
                </p>

                {formError && (
                  <div className="text-[11px] text-rose-400 bg-rose-950/60 p-2 rounded border border-rose-900/60">
                    {formError}
                  </div>
                )}
                {formSuccess && (
                  <div className="text-[11px] text-emerald-400 bg-emerald-950/60 p-2 rounded border border-emerald-900/60">
                    {formSuccess}
                  </div>
                )}

                <div>
                  <label className="block text-[10px] font-semibold text-slate-300 mb-0.5">
                    Contact Name *
                  </label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Contact Name"
                    required
                    className="w-full bg-slate-900 border border-slate-700 focus:border-cyan-500 rounded px-2.5 py-1 text-xs text-slate-100 outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[10px] font-semibold text-slate-300 mb-0.5">
                    Relationship
                  </label>
                  <input
                    type="text"
                    value={relationship}
                    onChange={(e) => setRelationship(e.target.value)}
                    placeholder="Relationship (e.g. Father, Spouse)"
                    className="w-full bg-slate-900 border border-slate-700 focus:border-cyan-500 rounded px-2.5 py-1 text-xs text-slate-100 outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[10px] font-semibold text-slate-300 mb-0.5">
                    Email Address *
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Email Address"
                    required
                    className="w-full bg-slate-900 border border-slate-700 focus:border-cyan-500 rounded px-2.5 py-1 text-xs text-slate-100 outline-none"
                  />
                </div>

                <div className="flex items-center gap-2 pt-1">
                  <button
                    type="submit"
                    disabled={isSavingContact}
                    className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs rounded transition-colors flex items-center gap-1.5 cursor-pointer"
                  >
                    {isSavingContact ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : 'Save Contact'}
                  </button>

                  {contact && (
                    <button
                      type="button"
                      onClick={() => setIsEditing(false)}
                      className="px-3 py-1.5 bg-slate-800 text-slate-300 hover:bg-slate-700 text-xs rounded transition-colors"
                    >
                      Cancel
                    </button>
                  )}
                </div>
              </form>
            ) : (
              <div className="space-y-2 pt-0.5">
                <div className="text-xs space-y-0.5 bg-slate-900/60 p-2.5 rounded border border-slate-800">
                  <div><span className="text-slate-400 font-medium">Name:</span> <span className="font-bold text-slate-200">{contact.contact_name}</span></div>
                  {contact.relationship && (
                    <div><span className="text-slate-400 font-medium">Relationship:</span> <span className="text-slate-300">{contact.relationship}</span></div>
                  )}
                  <div><span className="text-slate-400 font-medium">Email:</span> <span className="text-cyan-400 font-mono">{contact.contact_email}</span></div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsEditing(true)}
                    className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-medium rounded flex items-center gap-1 cursor-pointer transition-colors"
                  >
                    <Edit2 className="w-3 h-3 text-cyan-400" /> Edit
                  </button>
                  <button
                    onClick={handleDeleteContact}
                    className="px-2.5 py-1 bg-slate-800 hover:bg-rose-950/60 text-rose-300 text-[11px] font-medium rounded flex items-center gap-1 cursor-pointer transition-colors"
                  >
                    <Trash2 className="w-3 h-3 text-rose-400" /> Delete
                  </button>
                </div>

                {/* 🚨 Emergency Alert Section Inside the Same Card */}
                <div className="mt-3 pt-2.5 border-t border-slate-800 space-y-2">
                  <div className="text-xs font-bold text-rose-400 flex items-center gap-1.5">
                    <span>🚨 Emergency Alert</span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Send emergency alert to: <span className="text-cyan-400 font-mono">{contact.contact_email}</span>
                  </p>

                  {alertFeedback && (
                    <div
                      className={`text-[11px] p-2 rounded border ${
                        alertFeedback.type === 'success'
                          ? 'bg-emerald-950/80 border-emerald-800 text-emerald-300'
                          : 'bg-rose-950/80 border-rose-800 text-rose-300'
                      }`}
                    >
                      {alertFeedback.message}
                    </div>
                  )}

                  <button
                    onClick={handleSendAlert}
                    disabled={isSendingAlert}
                    className={`w-full py-1.5 px-3 rounded text-xs font-bold text-white flex items-center justify-center gap-2 transition-all cursor-pointer ${
                      isSendingAlert
                        ? 'bg-rose-900 cursor-not-allowed opacity-70'
                        : 'bg-gradient-to-r from-rose-600 to-red-700 hover:from-rose-500 hover:to-red-600 shadow-md'
                    }`}
                  >
                    {isSendingAlert ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" /> Sending Emergency Alert...
                      </>
                    ) : (
                      <>
                        <Send className="w-3.5 h-3.5" /> Send Emergency Alert
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Card 4: 📬 Notification History */}
          <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3 space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-200">
              <Mail className="w-3.5 h-3.5 text-cyan-400" />
              <span>Notification History</span>
            </div>

            {!isAuthenticated ? (
              <p className="text-[11px] text-slate-400">
                Please sign in with Google first.
              </p>
            ) : isLoadingHistory ? (
              <div className="flex items-center gap-2 text-xs text-slate-400 py-1">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                <span>Loading notification logs...</span>
              </div>
            ) : logs.length === 0 ? (
              <p className="text-[11px] text-slate-500 italic">
                No notification history available.
              </p>
            ) : (
              <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                {logs.map((log) => (
                  <div
                    key={log.id}
                    className="p-2 bg-slate-900 border border-slate-800 rounded text-[11px] font-mono space-y-0.5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300 truncate max-w-[150px]">To: {log.contact_email}</span>
                      <span
                        className={`px-1.5 py-0.2 text-[9px] font-bold rounded ${
                          log.status === 'sent'
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                            : 'bg-rose-950 text-rose-400 border border-rose-800'
                        }`}
                      >
                        {log.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-[10px] text-slate-500">
                      Date: {new Date(log.created_at).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
