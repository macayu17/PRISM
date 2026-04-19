import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AlertCircle,
  ArrowRight,
  Brain,
  CalendarDays,
  ChevronDown,
  ChevronUp,
  Gauge,
  GitBranch,
  Loader,
  Orbit,
  Plus,
  Sparkles,
  Stethoscope,
  TrendingUp,
  Zap,
} from 'lucide-react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import BrainScene from '../components/BrainScene';
import BurdenGauge from '../components/BurdenGauge';
import RiskTimeline from '../components/RiskTimeline';
import TwinComparison from '../components/TwinComparison';
import { addTwinSnapshot, getTwin, listTwins, simulateTwin } from '../api/client';
import {
  alertClass,
  badgeClass,
  buttonPrimary,
  buttonSecondary,
  glassPanel,
  innerPanel,
  inputField,
  labelText,
  pageShellWide,
  sectionTitle,
} from '../lib/ui';

/* ------------------------------------------------------------------ */
/*  Scenario slider fields                                             */
/* ------------------------------------------------------------------ */
const scenarioFields = [
  { name: 'sym_tremor', label: 'Tremor', min: 0, max: 4, step: 1 },
  { name: 'sym_rigid', label: 'Rigidity', min: 0, max: 4, step: 1 },
  { name: 'sym_brady', label: 'Bradykinesia', min: 0, max: 4, step: 1 },
  { name: 'sym_posins', label: 'Postural Instability', min: 0, max: 4, step: 1 },
  { name: 'moca', label: 'MoCA', min: 0, max: 30, step: 1 },
  { name: 'ess', label: 'ESS', min: 0, max: 24, step: 1 },
  { name: 'gds', label: 'GDS', min: 0, max: 15, step: 1 },
  { name: 'stai', label: 'STAI', min: 20, max: 80, step: 1 },
  { name: 'LEDD', label: 'LEDD', min: 0, max: 2000, step: 5 },
];

/* ------------------------------------------------------------------ */
/*  Snapshot required fields                                           */
/* ------------------------------------------------------------------ */
const snapshotRequiredFields = [
  { name: 'age', label: 'Age', type: 'number', min: 18, max: 100 },
  { name: 'SEX', label: 'Sex', type: 'select', options: [{ v: '', l: 'Select' }, { v: '0', l: 'Female' }, { v: '1', l: 'Male' }] },
  { name: 'EDUCYRS', label: 'Education Yrs', type: 'number', min: 0, max: 25 },
  { name: 'BMI', label: 'BMI', type: 'number', min: 15, max: 50, step: 0.1 },
  { name: 'sym_tremor', label: 'Tremor (0-4)', type: 'number', min: 0, max: 4 },
  { name: 'sym_rigid', label: 'Rigidity (0-4)', type: 'number', min: 0, max: 4 },
  { name: 'sym_brady', label: 'Bradykinesia (0-4)', type: 'number', min: 0, max: 4 },
  { name: 'sym_posins', label: 'Postural Inst. (0-4)', type: 'number', min: 0, max: 4 },
  { name: 'moca', label: 'MoCA (0-30)', type: 'number', min: 0, max: 30 },
  { name: 'visit_date', label: 'Visit Date', type: 'date' },
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */
function latestSnapshot(twin) {
  if (!twin?.snapshots?.length) return null;
  return twin.snapshots[twin.snapshots.length - 1];
}

function buildScenarioDefaults(snapshot) {
  const rawInputs = snapshot?.raw_inputs || {};
  const defaults = {};
  for (const field of scenarioFields) {
    const rawValue = rawInputs[field.name];
    defaults[field.name] = rawValue ?? '';
  }
  return defaults;
}

function buildSnapshotDefaults(snapshot) {
  const rawInputs = snapshot?.raw_inputs || {};
  const defaults = {};
  for (const field of snapshotRequiredFields) {
    if (field.name === 'visit_date') {
      defaults[field.name] = new Date().toISOString().split('T')[0];
    } else {
      defaults[field.name] = rawInputs[field.name] ?? '';
    }
  }
  return defaults;
}

function formatTwinDate(value) {
  if (!value) return 'Unknown';
  return value.replace('T', ' ').replace('Z', '');
}

/* ------------------------------------------------------------------ */
/*  Snapshot Diff                                                       */
/* ------------------------------------------------------------------ */
function SnapshotDiff({ snapshots }) {
  if (!snapshots || snapshots.length < 2) return null;

  const first = snapshots[0];
  const last = snapshots[snapshots.length - 1];

  const metrics = [
    { key: 'motor.updrs3_score', label: 'UPDRS-III', path: ['motor', 'updrs3_score'] },
    { key: 'motor.hy', label: 'H&Y Stage', path: ['motor', 'hy'] },
    { key: 'cognition.moca', label: 'MoCA', path: ['cognition', 'moca'] },
    { key: 'non_motor.gds', label: 'Depression (GDS)', path: ['non_motor', 'gds'] },
    { key: 'non_motor.ess', label: 'Sleepiness (ESS)', path: ['non_motor', 'ess'] },
    { key: 'ledd', label: 'LEDD', path: ['ledd'] },
  ];

  function getValue(snapshot, path) {
    let val = snapshot;
    for (const key of path) {
      val = val?.[key];
      if (val == null) return null;
    }
    return typeof val === 'number' ? val : null;
  }

  return (
    <div className="space-y-2">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
        First → Latest Snapshot Changes
      </div>
      {metrics.map(({ key, label, path }) => {
        const v1 = getValue(first, path);
        const v2 = getValue(last, path);
        if (v1 == null && v2 == null) return null;

        const delta = v1 != null && v2 != null ? v2 - v1 : null;
        const increasing = delta != null && delta > 0;
        const decreasing = delta != null && delta < 0;

        return (
          <div
            key={key}
            className="flex items-center justify-between rounded-xl border border-white/8 bg-black/20 px-4 py-2.5"
          >
            <span className="text-sm text-slate-400">{label}</span>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-500">
                {v1 != null ? v1.toFixed(1) : '—'}
              </span>
              <ArrowRight size={12} className="text-slate-600" />
              <span className="text-sm font-semibold text-white">
                {v2 != null ? v2.toFixed(1) : '—'}
              </span>
              {delta != null && Math.abs(delta) >= 0.05 && (
                <span
                  className={`flex items-center gap-0.5 text-xs font-semibold ${
                    increasing ? 'text-rose-400' : decreasing ? 'text-emerald-400' : 'text-slate-400'
                  }`}
                >
                  {increasing ? (
                    <ChevronUp size={12} />
                  ) : decreasing ? (
                    <ChevronDown size={12} />
                  ) : null}
                  {Math.abs(delta).toFixed(1)}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */
export default function TwinPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const queryTwinId = new URLSearchParams(location.search).get('id');

  const [twins, setTwins] = useState([]);
  const [selectedTwinId, setSelectedTwinId] = useState(queryTwinId || '');
  const [twin, setTwin] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [scenarioName, setScenarioName] = useState('What-if scenario');
  const [scenarioForm, setScenarioForm] = useState({});
  const [loadingList, setLoadingList] = useState(true);
  const [loadingTwin, setLoadingTwin] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState('');

  // Add-snapshot panel state
  const [showAddSnapshot, setShowAddSnapshot] = useState(false);
  const [snapshotForm, setSnapshotForm] = useState({});
  const [addingSnapshot, setAddingSnapshot] = useState(false);

  // ---- Load twins list ----
  useEffect(() => {
    let active = true;
    async function load() {
      setLoadingList(true);
      try {
        const data = await listTwins();
        if (!active) return;
        const nextTwins = Array.isArray(data?.twins) ? data.twins : [];
        setTwins(nextTwins);
        if (!selectedTwinId && nextTwins.length > 0) {
          setSelectedTwinId(queryTwinId || nextTwins[0].twin_id);
        }
      } catch (err) {
        if (active) {
          setError(err.message || 'Failed to load digital twins');
        }
      } finally {
        if (active) {
          setLoadingList(false);
        }
      }
    }

    load();
    return () => {
      active = false;
    };
  }, [queryTwinId, selectedTwinId]);

  // ---- Load selected twin ----
  useEffect(() => {
    if (!selectedTwinId) {
      setTwin(null);
      return;
    }

    let active = true;
    async function loadTwinDetail() {
      setLoadingTwin(true);
      setError('');
      try {
        const data = await getTwin(selectedTwinId);
        if (!active) return;
        const nextTwin = data?.twin || null;
        setTwin(nextTwin);
        setSimulation(null);
        setScenarioForm(buildScenarioDefaults(latestSnapshot(nextTwin)));
        setSnapshotForm(buildSnapshotDefaults(latestSnapshot(nextTwin)));
        navigate(`/twin?id=${selectedTwinId}`, { replace: true });
      } catch (err) {
        if (active) {
          setError(err.message || 'Failed to load selected twin');
        }
      } finally {
        if (active) {
          setLoadingTwin(false);
        }
      }
    }

    loadTwinDetail();
    return () => {
      active = false;
    };
  }, [navigate, selectedTwinId]);

  const selectedSnapshot = latestSnapshot(twin);
  const simulatedSnapshot = simulation?.simulated_snapshot || null;

  // ---- Forecast chart data with uncertainty bands ----
  const forecastChartData = useMemo(() => {
    const baseForecast = Array.isArray(twin?.forecast) ? twin.forecast : [];
    const simulationForecast = Array.isArray(simulation?.forecast) ? simulation.forecast : [];

    return baseForecast.map((point, index) => {
      const simPoint = simulationForecast[index];
      return {
        horizon: `${point.horizon_months} months`,
        baseline_updrs3: point.predicted_updrs3,
        baseline_moca: point.predicted_moca,
        // Uncertainty bands for baseline
        updrs3_upper: point.predicted_updrs3 != null && point.uncertainty?.updrs3_pm != null
          ? point.predicted_updrs3 + point.uncertainty.updrs3_pm
          : null,
        updrs3_lower: point.predicted_updrs3 != null && point.uncertainty?.updrs3_pm != null
          ? Math.max(0, point.predicted_updrs3 - point.uncertainty.updrs3_pm)
          : null,
        moca_upper: point.predicted_moca != null && point.uncertainty?.moca_pm != null
          ? Math.min(30, point.predicted_moca + point.uncertainty.moca_pm)
          : null,
        moca_lower: point.predicted_moca != null && point.uncertainty?.moca_pm != null
          ? Math.max(0, point.predicted_moca - point.uncertainty.moca_pm)
          : null,
        // Simulation lines
        simulated_updrs3: simPoint?.predicted_updrs3 ?? null,
        simulated_moca: simPoint?.predicted_moca ?? null,
      };
    });
  }, [simulation, twin]);

  // ---- Run simulation ----
  async function handleRunSimulation() {
    if (!selectedTwinId) return;

    const overrides = {};
    for (const [key, value] of Object.entries(scenarioForm)) {
      if (value !== '' && value !== null && value !== undefined) {
        overrides[key] = Number.isNaN(Number(value)) ? value : parseFloat(value);
      }
    }

    setSimulating(true);
    setError('');
    try {
      const data = await simulateTwin(selectedTwinId, overrides, scenarioName);
      setSimulation(data?.simulation || null);
    } catch (err) {
      setError(err.message || 'Simulation failed');
    } finally {
      setSimulating(false);
    }
  }

  // ---- Add snapshot ----
  async function handleAddSnapshot() {
    if (!selectedTwinId) return;

    const patientData = {};
    for (const [key, value] of Object.entries(snapshotForm)) {
      if (value !== '' && value !== null && value !== undefined) {
        patientData[key] = Number.isNaN(Number(value)) ? value : parseFloat(value);
      }
    }

    setAddingSnapshot(true);
    setError('');
    try {
      const data = await addTwinSnapshot(selectedTwinId, patientData);
      if (data?.twin) {
        setTwin(data.twin);
        setSimulation(null);
        setScenarioForm(buildScenarioDefaults(latestSnapshot(data.twin)));
        setSnapshotForm(buildSnapshotDefaults(latestSnapshot(data.twin)));
        setShowAddSnapshot(false);
        // Refresh the twins list
        const refreshed = await listTwins();
        setTwins(Array.isArray(refreshed?.twins) ? refreshed.twins : []);
      }
    } catch (err) {
      setError(err.message || 'Failed to add snapshot');
    } finally {
      setAddingSnapshot(false);
    }
  }

  return (
    <div className={pageShellWide}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        {/* Header */}
        <div className="mb-8 text-center">
          <span className={sectionTitle}>Digital Twin</span>
          <h2>Parkinson&apos;s Patient Twin</h2>
          <p className="mx-auto max-w-3xl text-base text-slate-400">
            A longitudinal clinical twin built from assessment data. Track progression,
            compare snapshots, simulate treatment scenarios, and forecast disease trajectory.
          </p>
        </div>

        {/* Error alert */}
        {error && (
          <div className={alertClass('danger')}>
            <AlertCircle size={18} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Main grid: sidebar + content */}
        <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">

          {/* -------- Sidebar: Twin list -------- */}
          <div className={`${glassPanel} bg-black/25`}>
            <div className="mb-4 flex items-center justify-between gap-3">
              <h4 className="text-lg">Saved Twins</h4>
              <span className={badgeClass('info')}>{twins.length}</span>
            </div>

            {loadingList ? (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <Loader size={16} className="animate-spin" />
                Loading twins...
              </div>
            ) : twins.length === 0 ? (
              <div className={`${innerPanel} text-sm text-slate-400`}>
                No twins yet. Run an assessment and create one from the results panel.
              </div>
            ) : (
              <div className="space-y-3">
                {twins.map((item) => (
                  <button
                    key={item.twin_id}
                    type="button"
                    onClick={() => setSelectedTwinId(item.twin_id)}
                    className={`w-full rounded-[1.25rem] border p-4 text-left transition ${
                      item.twin_id === selectedTwinId
                        ? 'border-sky-400/40 bg-sky-400/10'
                        : 'border-white/10 bg-black/30 hover:border-white/20 hover:bg-black/40'
                    }`}
                  >
                    <div className="mb-2 flex items-start justify-between gap-3">
                      <div>
                        <div className="text-sm font-semibold text-white">{item.patient_label}</div>
                        <div className="text-xs text-slate-500">{item.twin_id}</div>
                      </div>
                      <span className={badgeClass(item.confidence >= 0.6 ? 'success' : 'warning')}>
                        {(item.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-sm text-slate-400">{item.current_cohort_estimate}</div>
                    <div className="mt-2 text-xs text-slate-500">
                      {item.snapshot_count} snapshots • Updated {formatTwinDate(item.updated_at)}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* -------- Main content -------- */}
          <div className="space-y-6">
            {loadingTwin ? (
              <div className={`${glassPanel} bg-black/25 text-sm text-slate-400`}>
                <div className="flex items-center gap-2">
                  <Loader size={16} className="animate-spin" />
                  Loading twin dashboard...
                </div>
              </div>
            ) : !twin ? (
              <div className={`${glassPanel} bg-black/25 text-sm text-slate-400`}>
                Select a digital twin to inspect its state, forecast, and simulations.
              </div>
            ) : (
              <>
                {/* ---- KPI Row ---- */}
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  <motion.div
                    className={`${innerPanel} bg-black/25`}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0 }}
                  >
                    <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                      <Brain size={16} />
                      Cohort Estimate
                    </div>
                    <div className="text-xl font-semibold text-white">
                      {twin.current_state.current_cohort_estimate}
                    </div>
                    <div className="mt-2 text-xs text-slate-500">
                      Source: {twin.current_state.prediction_source}
                    </div>
                  </motion.div>

                  <motion.div
                    className={`${innerPanel} bg-black/25`}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.05 }}
                  >
                    <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                      <Sparkles size={16} />
                      Confidence
                    </div>
                    <div className="text-xl font-semibold text-white">
                      {(twin.current_state.confidence * 100).toFixed(1)}%
                    </div>
                  </motion.div>

                  <motion.div
                    className={`${innerPanel} bg-black/25`}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                  >
                    <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                      <Gauge size={16} />
                      Treatment Response
                    </div>
                    <div className="text-xl font-semibold text-white">
                      {twin.current_state.treatment_response_proxy != null
                        ? twin.current_state.treatment_response_proxy.toFixed(1)
                        : 'N/A'}
                    </div>
                    <div className="mt-2 text-xs text-slate-500">
                      UPDRS OFF–ON delta
                    </div>
                  </motion.div>

                  <motion.div
                    className={`${innerPanel} bg-black/25`}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.15 }}
                  >
                    <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                      <GitBranch size={16} />
                      Progression Velocity
                    </div>
                    <div className="text-xl font-semibold text-white">
                      {twin.current_state.progression_velocity != null
                        ? `${twin.current_state.progression_velocity.toFixed(2)} u/yr`
                        : 'Baseline only'}
                    </div>
                  </motion.div>
                </div>

                {/* ---- Burden Gauges + Brain Scene ---- */}
                <div className="grid gap-6 xl:grid-cols-2">
                  {/* Burden Comparison Panel */}
                  <div className={`${glassPanel} bg-black/25`}>
                    <div className="mb-4 flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <TrendingUp size={18} />
                        <h4 className="text-lg">Burden Indices</h4>
                      </div>
                      <span className={badgeClass(simulation ? 'accent' : 'info')}>
                        {simulation ? 'Simulated' : 'Current'}
                      </span>
                    </div>
                    <TwinComparison
                      baselineState={twin.current_state}
                      simulatedState={simulation?.state || null}
                    />
                  </div>

                  {/* Brain Visualization */}
                  <div className={`${glassPanel} bg-black/25`}>
                    <div className="mb-4 flex items-center justify-between gap-3">
                      <h4 className="text-lg">Neural Visualization</h4>
                      <span className={badgeClass(simulation ? 'accent' : 'info')}>
                        {simulation ? simulation.scenario_name : selectedSnapshot?.event_id || 'Current'}
                      </span>
                    </div>
                    <div className="h-[380px] overflow-hidden rounded-[1.5rem] border border-white/10 bg-black/30">
                      <BrainScene
                        symptomData={
                          simulatedSnapshot?.raw_inputs || selectedSnapshot?.raw_inputs || {}
                        }
                      />
                    </div>
                    <div className="mt-3 text-sm text-slate-400">
                      {simulation
                        ? 'Heatmap reflects simulated symptom values'
                        : `Latest visit: ${selectedSnapshot?.visit_date || 'Unknown'} • ${
                            twin.summary?.snapshot_count || twin.snapshots?.length || 0
                          } snapshots`}
                    </div>
                  </div>
                </div>

                {/* ---- Risk Timeline ---- */}
                <div className={`${glassPanel} bg-black/25`}>
                  <div className="mb-6 flex items-center gap-2">
                    <Zap size={18} />
                    <h4 className="text-lg">Risk Forecast Timeline</h4>
                  </div>
                  <RiskTimeline forecast={twin.forecast} />
                </div>

                {/* ---- Forecast Chart with Uncertainty ---- */}
                <div className={`${glassPanel} bg-black/25`}>
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <h4 className="text-lg">Forecast Trajectory</h4>
                    <span className={badgeClass('info')}>Heuristic v1</span>
                  </div>
                  <div className="h-[360px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={forecastChartData}>
                        <defs>
                          <linearGradient id="updrs3Band" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.15} />
                            <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.02} />
                          </linearGradient>
                          <linearGradient id="mocaBand" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.15} />
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0.02} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                        <XAxis dataKey="horizon" stroke="#64748b" tick={{ fontSize: 12 }} />
                        <YAxis stroke="#64748b" tick={{ fontSize: 12 }} />
                        <Tooltip
                          contentStyle={{
                            background: '#0a0a0a',
                            border: '1px solid rgba(255,255,255,0.12)',
                            borderRadius: 16,
                            fontSize: 13,
                          }}
                          labelStyle={{ color: '#94a3b8' }}
                        />
                        <Legend />

                        {/* Uncertainty bands */}
                        <Area
                          type="monotone"
                          dataKey="updrs3_upper"
                          stroke="none"
                          fill="url(#updrs3Band)"
                          name="UPDRS III ± range"
                          legendType="none"
                        />
                        <Area
                          type="monotone"
                          dataKey="updrs3_lower"
                          stroke="none"
                          fill="url(#updrs3Band)"
                          legendType="none"
                          name=" "
                        />

                        {/* Main lines */}
                        <Line
                          type="monotone"
                          dataKey="baseline_updrs3"
                          stroke="#38bdf8"
                          strokeWidth={2.5}
                          dot={{ fill: '#38bdf8', r: 4 }}
                          name="Baseline UPDRS III"
                        />
                        <Line
                          type="monotone"
                          dataKey="baseline_moca"
                          stroke="#10b981"
                          strokeWidth={2.5}
                          dot={{ fill: '#10b981', r: 4 }}
                          name="Baseline MoCA"
                        />

                        {/* Simulation overlay */}
                        {simulation && (
                          <>
                            <Line
                              type="monotone"
                              dataKey="simulated_updrs3"
                              stroke="#f59e0b"
                              strokeWidth={2}
                              strokeDasharray="8 4"
                              dot={{ fill: '#f59e0b', r: 3 }}
                              name="Scenario UPDRS III"
                            />
                            <Line
                              type="monotone"
                              dataKey="simulated_moca"
                              stroke="#f472b6"
                              strokeWidth={2}
                              strokeDasharray="8 4"
                              dot={{ fill: '#f472b6', r: 3 }}
                              name="Scenario MoCA"
                            />
                          </>
                        )}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* ---- Scenario Simulator + Timeline + Evidence ---- */}
                <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
                  {/* Scenario Simulator */}
                  <div className={`${glassPanel} bg-black/25`}>
                    <div className="mb-4 flex items-center gap-2">
                      <Orbit size={18} />
                      <h4 className="text-lg">Scenario Simulator</h4>
                    </div>
                    <div className="mb-4">
                      <label className={labelText}>Scenario Name</label>
                      <input
                        className={inputField}
                        value={scenarioName}
                        onChange={(e) => setScenarioName(e.target.value)}
                        placeholder="What-if scenario"
                      />
                    </div>
                    <div className="grid gap-4 md:grid-cols-2">
                      {scenarioFields.map((field) => (
                        <div key={field.name}>
                          <label className={labelText}>{field.label}</label>
                          <input
                            className={inputField}
                            type="number"
                            min={field.min}
                            max={field.max}
                            step={field.step}
                            value={scenarioForm[field.name] ?? ''}
                            onChange={(e) =>
                              setScenarioForm((prev) => ({
                                ...prev,
                                [field.name]: e.target.value,
                              }))
                            }
                          />
                        </div>
                      ))}
                    </div>
                    <div className="mt-6 flex flex-wrap gap-3">
                      <button
                        type="button"
                        className={buttonPrimary}
                        onClick={handleRunSimulation}
                        disabled={simulating}
                      >
                        {simulating ? (
                          <>
                            <Loader size={16} className="animate-spin" />
                            Simulating...
                          </>
                        ) : (
                          <>
                            <Sparkles size={16} />
                            Run Simulation
                          </>
                        )}
                      </button>
                      <button
                        type="button"
                        className={buttonSecondary}
                        onClick={() => {
                          setSimulation(null);
                          setScenarioForm(buildScenarioDefaults(selectedSnapshot));
                        }}
                      >
                        Reset Scenario
                      </button>
                    </div>
                  </div>

                  {/* Right column: Timeline + Evidence + Snapshot Diff */}
                  <div className="space-y-6">
                    {/* Add Snapshot */}
                    <div className={`${glassPanel} bg-black/25`}>
                      <div className="mb-4 flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <Plus size={18} />
                          <h4 className="text-lg">Add Follow-up Visit</h4>
                        </div>
                        <button
                          type="button"
                          className={buttonSecondary + ' !px-3 !py-1.5 !text-xs'}
                          onClick={() => setShowAddSnapshot(!showAddSnapshot)}
                        >
                          {showAddSnapshot ? 'Hide' : 'Expand'}
                        </button>
                      </div>
                      <AnimatePresence>
                        {showAddSnapshot && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3 }}
                            className="overflow-hidden"
                          >
                            <div className="grid gap-3 md:grid-cols-2">
                              {snapshotRequiredFields.map((field) => (
                                <div key={field.name}>
                                  <label className={labelText}>{field.label}</label>
                                  {field.type === 'select' ? (
                                    <select
                                      className={`${inputField} appearance-none`}
                                      value={snapshotForm[field.name] ?? ''}
                                      onChange={(e) =>
                                        setSnapshotForm((prev) => ({
                                          ...prev,
                                          [field.name]: e.target.value,
                                        }))
                                      }
                                    >
                                      {field.options.map((opt) => (
                                        <option key={opt.v} value={opt.v}>
                                          {opt.l}
                                        </option>
                                      ))}
                                    </select>
                                  ) : (
                                    <input
                                      className={inputField}
                                      type={field.type}
                                      min={field.min}
                                      max={field.max}
                                      step={field.step}
                                      value={snapshotForm[field.name] ?? ''}
                                      onChange={(e) =>
                                        setSnapshotForm((prev) => ({
                                          ...prev,
                                          [field.name]: e.target.value,
                                        }))
                                      }
                                    />
                                  )}
                                </div>
                              ))}
                            </div>
                            <div className="mt-4">
                              <button
                                type="button"
                                className={buttonPrimary}
                                onClick={handleAddSnapshot}
                                disabled={addingSnapshot}
                              >
                                {addingSnapshot ? (
                                  <>
                                    <Loader size={16} className="animate-spin" />
                                    Adding...
                                  </>
                                ) : (
                                  <>
                                    <Stethoscope size={16} />
                                    Add Snapshot
                                  </>
                                )}
                              </button>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Snapshot Timeline */}
                    <div className={`${glassPanel} bg-black/25`}>
                      <div className="mb-4 flex items-center gap-2">
                        <CalendarDays size={18} />
                        <h4 className="text-lg">Snapshot Timeline</h4>
                      </div>
                      <div className="space-y-3">
                        {twin.snapshots.map((snapshot, idx) => (
                          <motion.div
                            key={snapshot.snapshot_id}
                            className={`${innerPanel} bg-black/25`}
                            initial={{ opacity: 0, x: -12 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.05 }}
                          >
                            <div className="mb-1 flex items-center justify-between gap-3">
                              <span className="font-semibold text-white">
                                {snapshot.event_id}
                              </span>
                              <span className={badgeClass('info')}>
                                {snapshot.visit_date}
                              </span>
                            </div>
                            <div className="text-sm text-slate-400">
                              LEDD {snapshot.ledd ?? 'N/A'} • Age{' '}
                              {snapshot.age_at_visit ?? 'N/A'} • Duration{' '}
                              {snapshot.duration_years ?? 'N/A'}
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>

                    {/* Snapshot comparison (if multi-snapshot) */}
                    {twin.snapshots.length >= 2 && (
                      <div className={`${glassPanel} bg-black/25`}>
                        <div className="mb-4 flex items-center gap-2">
                          <TrendingUp size={18} />
                          <h4 className="text-lg">Longitudinal Changes</h4>
                        </div>
                        <SnapshotDiff snapshots={twin.snapshots} />
                      </div>
                    )}

                    {/* Evidence */}
                    <div className={`${glassPanel} bg-black/25`}>
                      <div className="mb-4 flex items-center gap-2">
                        <Sparkles size={18} />
                        <h4 className="text-lg">Evidence</h4>
                      </div>
                      <div className="space-y-3 text-sm text-slate-300">
                        {twin.current_state.evidence?.map((item, idx) => (
                          <motion.div
                            key={item}
                            className={`${innerPanel} bg-black/25`}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: idx * 0.06 }}
                          >
                            {item}
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
