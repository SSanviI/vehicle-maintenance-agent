import React, { useState, useEffect } from 'react';
import { Shield, Save, Trash2, Edit3, Lock, CheckCircle, AlertCircle, X } from 'lucide-react';
import type { EmergencyContact } from '../types';

interface EmergencyContactPanelProps {
  isAuthenticated: boolean;
  isOpen: boolean;
  onClose: () => void;
}

export const EmergencyContactPanel: React.FC<EmergencyContactPanelProps> = ({
  isAuthenticated,
  isOpen,
  onClose,
}) => {
  const [contact, setContact] = useState<EmergencyContact | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState('');
  const [relationship, setRelationship] = useState('');
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

  useEffect(() => {
    if (isOpen && isAuthenticated) {
      fetchContact();
    }
  }, [isOpen, isAuthenticated]);

  const fetchContact = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/emergency/contact', { credentials: 'include' });
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
      setLoading(false);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim()) return;

    setLoading(true);
    setMessage(null);

    try {
      const res = await fetch('/api/emergency/contact', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
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
        setMessage({ text: 'Emergency contact saved successfully.', type: 'success' });
      } else {
        const errData = await res.json();
        setMessage({ text: errData.detail || 'Failed to save contact', type: 'error' });
      }
    } catch (err) {
      setMessage({ text: 'Network error saving contact', type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm('Delete emergency contact?')) return;

    setLoading(true);
    setMessage(null);

    try {
      const res = await fetch('/api/emergency/contact', { method: 'DELETE', credentials: 'include' });
      if (res.ok) {
        setContact(null);
        setName('');
        setRelationship('');
        setEmail('');
        setIsEditing(true);
        setMessage({ text: 'Emergency contact deleted.', type: 'success' });
      }
    } catch (err) {
      setMessage({ text: 'Failed to delete contact', type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-5 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-rose-500/20 border border-rose-500/30 flex items-center justify-center text-rose-400">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-bold text-slate-100 text-base">Emergency Contact</h3>
            <p className="text-xs text-slate-400">Recipient for vehicle emergency notifications</p>
          </div>
        </div>

        {!isAuthenticated ? (
          <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-800 text-center space-y-3">
            <Lock className="w-8 h-8 text-amber-400 mx-auto" />
            <p className="text-xs text-slate-300">
              You must be signed in with Google to configure emergency contacts and send notifications.
            </p>
            <a
              href="/auth/login"
              className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl text-xs font-semibold text-white hover:opacity-90 transition-opacity"
            >
              Sign in with Google
            </a>
          </div>
        ) : (
          <>
            {message && (
              <div
                className={`p-3 rounded-xl text-xs flex items-center gap-2 ${
                  message.type === 'success'
                    ? 'bg-emerald-950/60 border border-emerald-500/30 text-emerald-300'
                    : 'bg-rose-950/60 border border-rose-500/30 text-rose-300'
                }`}
              >
                {message.type === 'success' ? (
                  <CheckCircle className="w-4 h-4 flex-shrink-0" />
                ) : (
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                )}
                <span>{message.text}</span>
              </div>
            )}

            {contact && !isEditing ? (
              <div className="space-y-4">
                <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-bold text-slate-200 text-sm">{contact.contact_name}</div>
                      {contact.relationship && (
                        <div className="text-xs text-cyan-400 font-medium">{contact.relationship}</div>
                      )}
                    </div>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      Active
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 font-mono pt-1">{contact.contact_email}</div>
                </div>

                <div className="flex gap-2 justify-end">
                  <button
                    onClick={() => setIsEditing(true)}
                    className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer"
                  >
                    <Edit3 className="w-3.5 h-3.5" /> Edit
                  </button>
                  <button
                    onClick={handleDelete}
                    disabled={loading}
                    className="px-3 py-1.5 bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 border border-rose-800/60 rounded-xl text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" /> Delete
                  </button>
                </div>
              </div>
            ) : (
              <form onSubmit={handleSave} className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Contact Name *</label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. Jane Doe"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-100 outline-none focus:border-cyan-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Relationship</label>
                  <input
                    type="text"
                    value={relationship}
                    onChange={(e) => setRelationship(e.target.value)}
                    placeholder="e.g. Spouse, Parent, Mechanic"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-100 outline-none focus:border-cyan-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Contact Email *</label>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="e.g. jane@example.com"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-100 outline-none focus:border-cyan-500"
                  />
                </div>

                <div className="flex gap-2 justify-end pt-2">
                  {contact && (
                    <button
                      type="button"
                      onClick={() => setIsEditing(false)}
                      className="px-3 py-1.5 bg-slate-800 text-slate-300 rounded-xl text-xs font-medium"
                    >
                      Cancel
                    </button>
                  )}
                  <button
                    type="submit"
                    disabled={loading}
                    className="px-4 py-1.5 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl text-xs font-semibold text-white flex items-center gap-1.5 shadow-md cursor-pointer"
                  >
                    <Save className="w-3.5 h-3.5" /> Save Contact
                  </button>
                </div>
              </form>
            )}
          </>
        )}
      </div>
    </div>
  );
};
