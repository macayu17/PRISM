import { motion } from 'framer-motion';

/**
 * BurdenGauge — animated SVG radial progress arc.
 *
 * Props:
 *   value   – 0-1 float (burden index)
 *   label   – 'Motor', 'Cognitive', 'Non-Motor'
 *   color   – tailwind hex e.g. '#38bdf8'
 *   size    – px diameter (default 140)
 */
export default function BurdenGauge({
  value = 0,
  label = '',
  color = '#38bdf8',
  size = 140,
}) {
  const clamped = Math.max(0, Math.min(1, value ?? 0));
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - clamped);
  const cx = size / 2;
  const cy = size / 2;

  const severityLabel =
    clamped >= 0.7 ? 'High' : clamped >= 0.4 ? 'Moderate' : 'Low';
  const severityColor =
    clamped >= 0.7
      ? 'text-rose-400'
      : clamped >= 0.4
        ? 'text-amber-400'
        : 'text-emerald-400';

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Track */}
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={10}
        />
        {/* Animated arc */}
        <motion.circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={10}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{ filter: `drop-shadow(0 0 8px ${color}40)` }}
        />
        {/* Center value */}
        <text
          x={cx}
          y={cy - 4}
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-white text-2xl font-bold"
          style={{ fontSize: 26, fontFamily: 'Rajdhani, sans-serif', fontWeight: 700 }}
        >
          {(clamped * 100).toFixed(0)}
        </text>
        <text
          x={cx}
          y={cy + 16}
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-slate-400"
          style={{ fontSize: 11, fontFamily: 'Outfit, sans-serif' }}
        >
          / 100
        </text>
      </svg>

      <div className="text-center">
        <div className="text-sm font-semibold text-white">{label}</div>
        <div className={`text-xs font-medium ${severityColor}`}>{severityLabel}</div>
      </div>
    </div>
  );
}
