import { motion } from 'framer-motion';
import { ArrowDown, ArrowUp, Minus } from 'lucide-react';
import BurdenGauge from './BurdenGauge';

function DeltaIndicator({ label, baseline, simulated }) {
  if (baseline == null || simulated == null) return null;

  const delta = simulated - baseline;
  const absDelta = Math.abs(delta);
  const formatted = absDelta < 0.01 ? '0.00' : absDelta.toFixed(2);

  let icon, colorClass;
  if (Math.abs(delta) < 0.005) {
    icon = <Minus size={14} />;
    colorClass = 'text-slate-400';
  } else if (delta > 0) {
    icon = <ArrowUp size={14} />;
    colorClass = 'text-rose-400';
  } else {
    icon = <ArrowDown size={14} />;
    colorClass = 'text-emerald-400';
  }

  return (
    <div className="flex items-center justify-between rounded-xl border border-white/8 bg-black/20 px-4 py-3">
      <span className="text-sm text-slate-400">{label}</span>
      <div className="flex items-center gap-3 text-right">
        <span className="text-xs text-slate-500">
          {typeof baseline === 'number' ? baseline.toFixed(2) : '—'}
        </span>
        <span className="text-slate-600">→</span>
        <span className="text-sm font-semibold text-white">
          {typeof simulated === 'number' ? simulated.toFixed(2) : '—'}
        </span>
        <span className={`flex items-center gap-0.5 text-xs font-semibold ${colorClass}`}>
          {icon}
          {formatted}
        </span>
      </div>
    </div>
  );
}

/**
 * TwinComparison — side-by-side view of baseline vs simulation burden indices.
 *
 * Props:
 *   baselineState   – twin.current_state
 *   simulatedState  – simulation.state (or null)
 */
export default function TwinComparison({ baselineState, simulatedState }) {
  if (!baselineState) return null;

  const hasSimulation = simulatedState != null;

  const burdens = [
    { key: 'motor_burden_index',     label: 'Motor',     color: '#38bdf8' },
    { key: 'cognitive_burden_index', label: 'Cognitive', color: '#a78bfa' },
    { key: 'non_motor_burden_index', label: 'Non-Motor', color: '#34d399' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Gauge row */}
      <div className="mb-6 grid grid-cols-3 gap-4">
        {burdens.map(({ key, label, color }) => (
          <div key={key} className="flex flex-col items-center gap-1">
            <BurdenGauge
              value={hasSimulation ? simulatedState[key] : baselineState[key]}
              label={label}
              color={color}
              size={130}
            />
            {hasSimulation && baselineState[key] != null && (
              <div className="mt-1 text-center text-xs text-slate-500">
                baseline: {(baselineState[key] * 100).toFixed(0)}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Delta metrics */}
      {hasSimulation && (
        <motion.div
          className="space-y-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
            Metric Deltas (Baseline → Simulation)
          </div>
          {burdens.map(({ key, label }) => (
            <DeltaIndicator
              key={key}
              label={label}
              baseline={baselineState[key]}
              simulated={simulatedState[key]}
            />
          ))}
          <DeltaIndicator
            label="Confidence"
            baseline={baselineState.confidence}
            simulated={simulatedState.confidence}
          />
        </motion.div>
      )}
    </motion.div>
  );
}
