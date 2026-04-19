import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

const RISK_CONFIG = {
  low:    { color: '#10b981', bg: 'bg-emerald-400/10', border: 'border-emerald-400/30', text: 'text-emerald-300', Icon: CheckCircle },
  medium: { color: '#f59e0b', bg: 'bg-amber-400/10',   border: 'border-amber-400/30',   text: 'text-amber-300',   Icon: AlertTriangle },
  high:   { color: '#ef4444', bg: 'bg-rose-400/10',     border: 'border-rose-400/30',     text: 'text-rose-300',    Icon: XCircle },
};

/**
 * RiskTimeline — horizontal timeline showing forecast risk at each horizon.
 *
 * Props:
 *   forecast  – array from twin.forecast, each with horizon_months, risk_level,
 *               predicted_updrs3, predicted_moca, predicted_hy, uncertainty
 *   className – optional extra CSS
 */
export default function RiskTimeline({ forecast = [], className = '' }) {
  if (!forecast.length) return null;

  return (
    <div className={className}>
      <div className="relative flex items-center justify-between">
        {/* Connector line */}
        <div className="absolute left-[10%] right-[10%] top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-white/8" />

        {forecast.map((point, index) => {
          const risk = point.risk_level?.toLowerCase() || 'low';
          const config = RISK_CONFIG[risk] || RISK_CONFIG.low;
          const { Icon } = config;

          return (
            <motion.div
              key={point.horizon_months}
              className="relative z-10 flex flex-col items-center gap-2"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.15, duration: 0.4 }}
            >
              {/* Node */}
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-full border-2 ${config.border} ${config.bg}`}
                style={{ boxShadow: `0 0 16px ${config.color}25` }}
              >
                <Icon size={20} className={config.text} />
              </div>

              {/* Horizon label */}
              <div className="text-center">
                <div className="text-sm font-bold text-white">
                  {point.horizon_months}mo
                </div>
                <div className={`text-xs font-semibold uppercase tracking-wider ${config.text}`}>
                  {risk}
                </div>
              </div>

              {/* Metrics */}
              <div className={`mt-1 rounded-xl border ${config.border} ${config.bg} px-3 py-2 text-center`}>
                <div className="text-xs text-slate-400">
                  UPDRS-III{' '}
                  <span className="font-semibold text-white">
                    {point.predicted_updrs3 ?? '—'}
                  </span>
                  {point.uncertainty?.updrs3_pm != null && (
                    <span className="text-slate-500"> ±{point.uncertainty.updrs3_pm}</span>
                  )}
                </div>
                <div className="text-xs text-slate-400">
                  MoCA{' '}
                  <span className="font-semibold text-white">
                    {point.predicted_moca ?? '—'}
                  </span>
                  {point.uncertainty?.moca_pm != null && (
                    <span className="text-slate-500"> ±{point.uncertainty.moca_pm}</span>
                  )}
                </div>
                <div className="text-xs text-slate-400">
                  H&Y{' '}
                  <span className="font-semibold text-white">
                    {point.predicted_hy ?? '—'}
                  </span>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
