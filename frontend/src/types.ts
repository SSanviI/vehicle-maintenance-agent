export interface PredictionData {
  prediction_id?: string;
  prediction: string; // "maintenance_required" | "no_maintenance_needed"
  prediction_class: number;
  maintenance_probability?: number;
  no_maintenance_probability?: number;
  model_score: number;
  score_semantics?: string;
  risk_level: string;
  features_used: string[];
  missing_features: string[];
  feature_sources: Record<string, string>;
  confidence_note?: string;
  previous_score?: number;
  known_features_count?: number;
  total_features?: number;
  // Present on newer API payloads. The UI still falls back to the canonical
  // assessment event so older persisted conversations remain readable.
  assessment_complete?: boolean;
  relevant_factor_count?: number;
}

export interface ExplanationFactor {
  feature?: string;
  display_name: string;
  value?: string | number;
  contribution: number;
  shap_value?: number;
  direction: "higher_risk" | "lower_risk";
}

export interface ExplanationData {
  available: boolean;
  risk?: string;
  probability?: number;
  shap_base_value?: number;
  reason?: string;
  top_factors: ExplanationFactor[];
}


/**
 * Display-safe subset of the canonical assessment state supplied by the API.
 * `known_features` is only used to render values that the user has already
 * confirmed; it is never inferred from the chat text.
 */
export interface AssessmentData {
  assessment_complete?: boolean;
  relevant_factor_count?: number;
  total_known?: number;
  assessment_mode?: string;
  latest_prediction_is_current?: boolean;
  known_features?: Record<string, unknown>;
}

export interface FeatureValueState {
  value: any;
  status: "known" | "unknown" | "estimated" | "unavailable" | "uncertain";
  source: "user" | "database" | "ml_estimator" | "tool";
}

export interface FeatureState {
  known_features: Record<string, any>;
  unknown_features: string[];
  declined_features: string[];
  total_known: number;
  total_features: number;
  latest_prediction?: {
    prediction: string;
    model_score: number;
    risk_level: string;
  };
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  prediction?: PredictionData;
  explanation?: ExplanationData;
  assessment?: AssessmentData;
  created_at?: string;
  isStreaming?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface AuthUser {
  id: string;
  name: string;
  email: string;
}

export interface EmergencyContact {
  id?: string;
  contact_name: string;
  relationship?: string;
  contact_email: string;
}

export interface EmergencyAlertData {
  is_emergency: boolean;
  severity: string;
  matched_policies: string[];
  conditions: string[];
  symptoms: string[];
}

export interface EmailLog {
  id: string;
  contact_email: string;
  subject: string;
  status: "pending" | "sent" | "failed";
  error_message?: string;
  prediction_summary?: string;
  prediction?: number;
  email_id?: string;
  sent_at?: string;
  created_at: string;
}
