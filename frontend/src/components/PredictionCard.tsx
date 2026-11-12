import React, { useState } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from 'lucide-react';
import type { PredictionData, ExplanationData } from '../types';

interface PredictionCardProps {
  prediction: PredictionData;
  explanation?: ExplanationData;
  onQuickAction?: (message: string) => void;
}

const formatFeatureName = (feature: string): string => {
  return feature
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
};

export const PredictionCard: React.FC<PredictionCardProps> = ({
  prediction,
  explanation,
  onQuickAction,
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);
  const [showTechDetails, setShowTechDetails] = useState(false);

  const rawRisk = (prediction.risk_level || (prediction.prediction_class === 1 ? 'HIGH' : 'LOW')).toUpperCase();
  const isHigh = rawRisk === 'HIGH' || prediction.prediction_class === 1;
  const isMedium = rawRisk === 'MEDIUM';

  const numericScore = typeof prediction.model_score === 'number'
    ? prediction.model_score
    : (prediction.maintenance_probability ?? 0);
  const scorePct = (numericScore * 100).toFixed(1);

  const isComplete = prediction.assessment_complete === true || Boolean(prediction.relevant_factor_count);
  const factorCount =
    prediction.relevant_factor_count ??
    prediction.known_features_count ??
    prediction.features_used?.length ??
    0;

  const confirmedFeatures = prediction.features_used || [];

  const theme = isHigh
    ? {
        bg: 'bg-[#151313] border-rose-900/50 shadow-rose-950/30',
        badge: 'bg-rose-950/80 text-rose-300 border-rose-800/80',
        scoreColor: 'text-rose-400',
        bar: 'bg-gradient-to-r from-rose-600 to-rose-400',
        track: 'bg-[#221819] border-rose-950',
        icon: <ShieldAlert className="w-5 h-5 text-rose-400 flex-shrink-0" />,
        riskLabel: 'HIGH',
      }
    : isMedium
    ? {
        bg: 'bg-[#151412] border-amber-900/50 shadow-amber-950/30',
        badge: 'bg-amber-950/80 text-amber-300 border-amber-800/80',
        scoreColor: 'text-amber-400',
        bar: 'bg-gradient-to-r from-amber-600 to-amber-400',
        track: 'bg-[#221f18] border-amber-950',
        icon: <ShieldAlert className="w-5 h-5 text-amber-400 flex-shrink-0" />,
        riskLabel: 'MEDIUM',
      }
    : {
        bg: 'bg-[#121513] border-emerald-900/50 shadow-emerald-950/30',
        badge: 'bg-emerald-950/80 text-emerald-300 border-emerald-800/80',
        scoreColor: 'text-emerald-400',
        bar: 'bg-gradient-to-r from-emerald-600 to-emerald-400',
        track: 'bg-[#18221c] border-emerald-950',
        icon: <ShieldCheck className="w-5 h-5 text-emerald-400 flex-shrink-0" />,
        riskLabel: 'LOW',
      };

  return (
    <div
      className={`my-4 w-full max-w-[720px] p-5 sm:p-6 rounded-2xl border ${theme.bg} shadow-2xl transition-all duration-200 text-slate-200 space-y-4`}
    >
      {/* 1. Header: Risk Label + Badge */}
      <div className="flex items-center justify-between gap-3 pb-1 border-b border-white/[0.06]">
        <div className="flex items-center gap-2.5">
          {theme.icon}
          <span className="font-semibold text-slate-100 text-base sm:text-lg tracking-tight">
            Maintenance Risk
          </span>
        </div>

        <span
          className={`px-3 py-1 text-xs sm:text-sm font-bold tracking-wider rounded-lg border ${theme.badge} shadow-sm`}
        >
          {theme.riskLabel}
        </span>
      </div>

      {/* 2. Primary Risk Value & Visual Bar */}
      <div className="space-y-2 py-1">
        <div className="flex items-baseline justify-between">
          <span className="text-xs sm:text-sm font-medium text-slate-400">
            Estimated maintenance risk
          </span>
          <span
            className={`text-3xl sm:text-4xl font-extrabold font-mono tracking-tight ${theme.scoreColor}`}
          >
            {scorePct}%
          </span>
        </div>

        {/* High Contrast Progress Bar */}
        <div
          className={`w-full ${theme.track} rounded-full h-3 overflow-hidden p-0.5 border`}
        >
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${theme.bar}`}
            style={{
              width: `${Math.max(5, Math.min(100, parseFloat(scorePct)))}%`,
            }}
          />
        </div>
      </div>

      {/* 3. Assessment Status & Relevant Factors */}
      <div className="pt-2 border-t border-white/[0.06] flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs sm:text-sm">
        <div className="flex items-center gap-2">
          {isComplete ? (
            <div className="flex items-center gap-2 text-emerald-400 font-medium">
              <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
              <span>Assessment complete</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-slate-300 font-medium">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-500"></span>
              </span>
              <span>Assessment in progress</span>
            </div>
          )}
        </div>

        <div className="text-slate-400 font-medium">
          {factorCount > 0 ? (
            <span>
              {factorCount} relevant factor{factorCount === 1 ? '' : 's'}{' '}
              {isComplete ? 'considered' : 'confirmed'}
            </span>
          ) : (
            <span>Gathering a few more relevant details...</span>
          )}
        </div>
      </div>

      {/* Confirmed Features (User Facing & Non-developer look) */}
      {confirmedFeatures.length > 0 && (
        <div className="pt-1">
          <div className="flex flex-wrap items-center gap-2 pt-1">
            {confirmedFeatures.slice(0, 6).map((feat) => (
              <span
                key={feat}
                className="px-3 py-1.5 rounded-lg bg-[#191919] text-slate-300 border border-[#2b2b2b] text-xs sm:text-[13px] font-medium transition-colors hover:border-slate-600"
              >
                {formatFeatureName(feat)}
              </span>
            ))}

            {confirmedFeatures.length > 6 && (
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="px-2.5 py-1.5 rounded-lg bg-[#191919] text-slate-400 hover:text-slate-200 border border-[#2b2b2b] text-xs font-medium flex items-center gap-1 transition-colors cursor-pointer"
              >
                <span>+{confirmedFeatures.length - 6} more</span>
                {showDetails ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>
            )}
          </div>

          {/* Expanded extra confirmed features */}
          {showDetails && confirmedFeatures.length > 6 && (
            <div className="flex flex-wrap items-center gap-2 pt-2 animate-in fade-in duration-200">
              {confirmedFeatures.slice(6).map((feat) => (
                <span
                  key={feat}
                  className="px-3 py-1.5 rounded-lg bg-[#191919] text-slate-300 border border-[#2b2b2b] text-xs sm:text-[13px] font-medium"
                >
                  {formatFeatureName(feat)}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 4. Model Explanation Section (If present from local model) */}
      {explanation?.available && explanation.top_factors && explanation.top_factors.length > 0 && (
        <div className="pt-3 border-t border-white/[0.06] space-y-2.5">
          <button
            onClick={() => setShowExplanation(!showExplanation)}
            className="flex items-center justify-between w-full py-1 text-xs sm:text-sm font-semibold text-slate-300 hover:text-slate-100 tracking-wide uppercase cursor-pointer transition-colors"
          >
            <div className="flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>Why this result?</span>
            </div>
            <div className="flex items-center gap-1 text-slate-400 text-xs font-normal normal-case">
              <span>{showExplanation ? 'Hide factors' : 'Show top factors'}</span>
              {showExplanation ? (
                <ChevronUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )}
            </div>
          </button>

          {showExplanation && (
            <div className="space-y-2 pt-1 animate-in fade-in duration-200">
              <div className="text-xs text-slate-400 font-medium pb-0.5">
                Strongest model contributors
              </div>
              {explanation.top_factors.map((factor) => {
                const pushesHigher = factor.direction === 'higher_risk';
                return (
                  <div
                    key={`${factor.display_name}-${factor.direction}`}
                    className="flex items-center justify-between gap-3 p-2.5 rounded-xl bg-[#171717] border border-[#262626] text-xs sm:text-sm"
                  >
                    <span className="font-medium text-slate-200">
                      {formatFeatureName(factor.display_name)}
                      {factor.value !== undefined && factor.value !== '' && factor.value !== null ? (
                        <span className="text-slate-400 font-normal">
                          {' '}
                          — {String(factor.value)}
                        </span>
                      ) : null}
                    </span>
                    <span
                      className={`px-2.5 py-0.5 rounded-md text-xs font-semibold border ${
                        pushesHigher
                          ? 'bg-rose-950/70 text-rose-300 border-rose-800/60'
                          : 'bg-emerald-950/70 text-emerald-300 border-emerald-800/60'
                      }`}
                    >
                      {pushesHigher ? 'Higher-risk contribution' : 'Lower-risk contribution'}
                    </span>
                  </div>
                );
              })}

              <div className="pt-1 flex items-center justify-between text-xs text-slate-500">
                <p className="italic">
                  The model combines several confirmed vehicle factors rather than relying on a single condition.
                </p>
                <button
                  onClick={() => setShowTechDetails(!showTechDetails)}
                  className="text-[11px] text-slate-400 hover:text-slate-300 underline ml-2 whitespace-nowrap cursor-pointer"
                >
                  {showTechDetails ? 'Hide technical details' : 'View technical details'}
                </button>
              </div>

              {showTechDetails && (
                <div className="mt-2 p-2.5 rounded-xl bg-[#111111] border border-[#222222] font-mono text-[11px] text-slate-400 space-y-1">
                  <div>Base model probability: {((explanation.shap_base_value ?? 0.5) * 100).toFixed(2)}%</div>
                  {explanation.top_factors.map((f) => (
                    <div key={f.display_name} className="flex justify-between">
                      <span>{f.display_name}:</span>
                      <span className={f.contribution > 0 ? 'text-rose-400' : 'text-emerald-400'}>
                        {f.contribution > 0 ? '+' : ''}{(f.contribution * 100).toFixed(2)}% SHAP impact
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}


      {/* 5. Quick Actions for Complete Assessment */}
      {isComplete && onQuickAction && (
        <div className="flex flex-wrap gap-2.5 pt-3 border-t border-white/[0.06]">
          <button
            onClick={() => onQuickAction('show me the summary')}
            className="inline-flex items-center gap-2 rounded-xl bg-slate-100 px-4 py-2 text-xs sm:text-sm font-semibold text-slate-950 transition hover:bg-white hover:shadow-md cursor-pointer active:scale-98"
          >
            <span>View assessment summary</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => onQuickAction('what is the main factor?')}
            className="rounded-xl border border-slate-700/80 bg-[#191919] hover:bg-[#222] px-4 py-2 text-xs sm:text-sm font-medium text-slate-200 transition hover:border-slate-500 cursor-pointer active:scale-98"
          >
            Why this result?
          </button>
        </div>
      )}
    </div>
  );
};
