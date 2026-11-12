import React, { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { ChatMessage } from './components/ChatMessage';
import { ChatComposer } from './components/ChatComposer';
import { EmergencyContactPanel } from './components/EmergencyContactPanel';
import { EmergencyAlert } from './components/EmergencyAlert';
import { EmergencyHistory } from './components/EmergencyHistory';
import { AccountEmergencyModal } from './components/AccountEmergencyModal';
import type { Message, Conversation, PredictionData, ExplanationData, AuthUser, EmergencyAlertData } from './types';
import { Car, Wrench, AlertTriangle } from 'lucide-react';

export const App: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // Auth State
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [, setIsAuthLoading] = useState(true);

  // Modal States
  const [isAccountOpen, setIsAccountOpen] = useState(false);
  const [isContactOpen, setIsContactOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [activeEmergencyAlert, setActiveEmergencyAlert] = useState<EmergencyAlertData | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Load auth state and conversations on mount
  useEffect(() => {
    checkAuth();
    fetchConversations();
  }, []);

  const checkAuth = async () => {
    setIsAuthLoading(true);
    console.log('[AUTH] Checking auth status...');
    try {
      const res = await fetch('/auth/status', { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        console.log('[AUTH] Status response:', data);
        setIsAuthenticated(Boolean(data.authenticated));
        setAuthUser(
          data.authenticated
            ? {
                id: data.google_user_id || 'google_user',
                name: data.name || data.email,
                email: data.email,
              }
            : null
        );
      } else {
        setIsAuthenticated(false);
        setAuthUser(null);
      }
    } catch (err) {
      console.error('[AUTH] Auth check failed:', err);
      setIsAuthenticated(false);
      setAuthUser(null);
    } finally {
      setIsAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await fetch('/auth/logout', {
        method: 'POST',
        credentials: 'include',
      });
    } catch (err) {
      console.error('[AUTH] Logout failed:', err);
    }
    setIsAuthenticated(false);
    setAuthUser(null);
  };

  const fetchConversations = async () => {
    try {
      const res = await fetch('/api/conversations', {
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        setConversations(data.conversations || []);
      }
    } catch (err) {
      console.error('Failed to fetch conversations:', err);
    }
  };

  const loadConversation = async (id: string) => {
    try {
      setActiveConversationId(id);
      const res = await fetch(`/api/conversations/${id}`, {
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        const loadedMsgs: Message[] = (data.messages || []).map((m: any) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          prediction: m.metadata?.prediction,
          explanation: m.metadata?.explanation,
          created_at: m.created_at,
        }));
        setMessages(loadedMsgs);
      }
    } catch (err) {
      console.error('Failed to load conversation:', err);
    }
  };

  const handleNewChat = () => {
    setActiveConversationId(null);
    setMessages([]);
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      const res = await fetch(`/api/conversations/${id}`, {
        method: 'DELETE',
        credentials: 'include',
      });
      if (res.ok) {
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (activeConversationId === id) {
          handleNewChat();
        }
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  const handleSendMessage = async (userText: string) => {
    const userMsgId = `user-${Date.now()}`;
    const userMsg: Message = {
      id: userMsgId,
      role: 'user',
      content: userText,
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userText,
          conversation_id: activeConversationId,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) return;

      const assistantMsgId = `assistant-${Date.now()}`;
      let assistantText = '';
      let currentPrediction: PredictionData | undefined = undefined;
      let currentExplanation: ExplanationData | undefined = undefined;

      // Add empty streaming assistant message
      setMessages((prev) => [
        ...prev,
        {
          id: assistantMsgId,
          role: 'assistant',
          content: '',
          isStreaming: true,
        },
      ]);

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));

              if (event.type === 'conversation_id' && event.content) {
                setActiveConversationId(event.content);
              } else if (event.type === 'text' && event.content) {
                assistantText += event.content;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, content: assistantText }
                      : msg
                  )
                );
              } else if (event.type === 'prediction' && event.data) {
                currentPrediction = event.data;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, prediction: currentPrediction }
                      : msg
                  )
                );
              } else if (event.type === 'explanation' && event.data) {
                currentExplanation = event.data;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, explanation: currentExplanation }
                      : msg
                  )
                );
              } else if (event.type === 'emergency_alert' && event.data) {
                // Emergency detected by backend emergency engine
                setActiveEmergencyAlert(event.data);
              } else if (event.type === 'error' && event.content) {
                assistantText = event.content;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, content: assistantText }
                      : msg
                  )
                );
              }
            } catch (err) {
              console.error('SSE parse error:', err);
            }
          }
        }
      }

      // Mark streaming as done
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMsgId ? { ...msg, isStreaming: false } : msg
        )
      );

      // Refresh list
      fetchConversations();

    } catch (err) {
      console.error('Send message error:', err);
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: "I'm having trouble connecting right now. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const activeTitle =
    conversations.find((c) => c.id === activeConversationId)?.title ||
    'New Maintenance Assessment';

  return (
    <div className="flex h-screen bg-[#050505] text-[#f5f5f5] overflow-hidden font-sans">
      {/* ChatGPT Persistent Left Sidebar */}
      <Sidebar
        conversations={conversations}
        activeId={activeConversationId}
        onSelectConversation={loadConversation}
        onNewChat={handleNewChat}
        onDeleteConversation={handleDeleteConversation}
        isOpen={isSidebarOpen}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        isAuthenticated={isAuthenticated}
        user={authUser}
        onOpenAccount={() => setIsAccountOpen(true)}
        onLogout={handleLogout}
      />

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative bg-[#050505]">
        <Header
          title={activeTitle}
          onOpenAccount={() => setIsAccountOpen(true)}
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          onNewChat={handleNewChat}
        />

        {/* Message Stream Workspace */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center p-6 sm:p-8 text-center max-w-3xl mx-auto space-y-8 my-auto">
              <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-gradient-to-tr from-cyan-950/80 via-slate-900 to-blue-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shadow-2xl">
                <Car className="w-8 h-8 sm:w-10 sm:h-10" />
              </div>

              <div className="space-y-3 max-w-xl">
                <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 tracking-tight">
                  AutoCare AI Maintenance Assistant
                </h2>
                <p className="text-sm sm:text-base text-slate-400 leading-relaxed">
                  Start an intelligent vehicle maintenance risk assessment. Share your vehicle's age, symptoms, or observed issues to begin.
                </p>
              </div>

              {/* Sample Assessment Prompt Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full text-left pt-2 max-w-2xl">
                <button
                  onClick={() => handleSendMessage('My car is 5 years old and I have noticed 3 problems recently.')}
                  className="p-4 sm:p-5 bg-[#121212] hover:bg-[#181818] border border-[#262626] hover:border-cyan-500/50 rounded-2xl transition-all space-y-2 cursor-pointer group shadow-md"
                >
                  <div className="font-semibold text-slate-200 group-hover:text-cyan-300 flex items-center gap-2 text-sm sm:text-base">
                    <Wrench className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                    <span>5-Year Vehicle Checkup</span>
                  </div>
                  <div className="text-slate-400 text-xs sm:text-sm leading-relaxed">
                    "My car is 5 years old and I have noticed 3 problems recently."
                  </div>
                </button>

                <button
                  onClick={() => handleSendMessage('My brakes are worn out and battery seems weak.')}
                  className="p-4 sm:p-5 bg-[#121212] hover:bg-[#181818] border border-[#262626] hover:border-rose-500/50 rounded-2xl transition-all space-y-2 cursor-pointer group shadow-md"
                >
                  <div className="font-semibold text-slate-200 group-hover:text-rose-300 flex items-center gap-2 text-sm sm:text-base">
                    <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0" />
                    <span>Worn Brakes & Battery</span>
                  </div>
                  <div className="text-slate-400 text-xs sm:text-sm leading-relaxed">
                    "My brakes are worn out and battery seems weak."
                  </div>
                </button>
              </div>
            </div>
          ) : (
            <div className="divide-y divide-[#181818]">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} onQuickAction={handleSendMessage} />
              ))}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* ChatGPT Style Input Composer */}
        <ChatComposer onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>

      {/* Account & Emergency Controls Modal */}
      <AccountEmergencyModal
        isOpen={isAccountOpen}
        onClose={() => setIsAccountOpen(false)}
        isAuthenticated={isAuthenticated}
        user={authUser}
        conversationId={activeConversationId}
        onLogout={handleLogout}
      />

      {/* Legacy Modal Support for Emergency Contact */}
      <EmergencyContactPanel
        isAuthenticated={isAuthenticated}
        isOpen={isContactOpen}
        onClose={() => setIsContactOpen(false)}
      />

      {/* Emergency History Modal */}
      <EmergencyHistory
        isAuthenticated={isAuthenticated}
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
      />

      {/* Emergency Alert Modal */}
      {activeEmergencyAlert && (
        <EmergencyAlert
          alert={activeEmergencyAlert}
          conversationId={activeConversationId}
          onClose={() => setActiveEmergencyAlert(null)}
        />
      )}
    </div>
  );
};

export default App;
