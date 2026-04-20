import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  AlertCircle,
  Brain,
  CalendarDays,
  Gauge,
  GitBranch,
  Loader,
  Orbit,
  PlusCircle,
  Sparkles,
} from 'lucide-react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
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

const visitFields = [
  { name: 'visit_date', label: 'Visit Date', type: 'date' },
  { name: 'EVENT_ID', label: 'Event ID', type: 'text', placeholder: 'e.g. V02 or MANUAL_2' },
  { name: 'YEAR', label: 'Year Index', type: 'number', min: 0, max: 30, step: 0.1 },
  { name: 'age', label: 'Age at Visit', type: 'number', min: 18, max: 120, step: 1 },
  { name: 'duration_yrs', label: 'Disease Duration (yrs)', type: 'number', min: 0, max: 80, step: 0.1 },
  { name: 'LEDD', label: 'LEDD', type: 'number', min: 0, max: 2500, step: 5 },
  { name: 'sym_tremor', label: 'Tremor', type: 'number', min: 0, max: 4, step: 1 },
  { name: 'sym_rigid', label: 'Rigidity', type: 'number', min: 0, max: 4, step: 1 },
  { name: 'sym_brady', label: 'Bradykinesia', type: 'number', min: 0, max: 4, step: 1 },
  { name: 'sym_posins', label: 'Postural Instability', type: 'number', min: 0, max: 4, step: 1 },
  { name: 'moca', label: 'MoCA', type: 'number', min: 0, max: 30, step: 1 },
  { name: 'ess', label: 'ESS', type: 'number', min: 0, max: 24, step: 1 },
  { name: 'gds', label: 'GDS', type: 'number', min: 0, max: 15, step: 1 },
  { name: 'stai', label: 'STAI', type: 'number', min: 20, max: 80, step: 1 },
  { name: 'updrs3_score', label: 'UPDRS III (OFF)', type: 'number', min: 0, max: 140, step: 0.5 },
  { name: 'updrs3_score_on', label: 'UPDRS III (ON)', type: 'number', min: 0, max: 140, step: 0.5 },
];

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

function addMonthsToIso(dateValue, months = 3) {
  if (!dateValue) return null;
  const dt = new Date(dateValue);
  if (Number.isNaN(dt.getTime())) return null;
  dt.setMonth(dt.getMonth() + months);
  return dt.toISOString().slice(0, 10);
}

function toNumberOrNull(value) {
  if (value === null || value === undefined || value === '') return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatStatValue(value, digits = 1) {
  const numeric = toNumberOrNull(value);
  if (numeric === null) {
    if (value === null || value === undefined || value === '') return 'N/A';
    return String(value);
  }
  return numeric.toFixed(digits);
}

function formatStatDelta(value, baseValue, digits = 1) {
  const current = toNumberOrNull(value);
  const base = toNumberOrNull(baseValue);
  if (current === null || base === null) return null;
  const delta = current - base;
  const sign = delta > 0 ? '+' : '';
  return `${sign}${delta.toFixed(digits)}`;
}

function getSnapshotMetric(snapshot, key) {
  if (!snapshot) return null;

  const rawInputs = snapshot.raw_inputs || {};
  if (rawInputs[key] !== undefined && rawInputs[key] !== null && rawInputs[key] !== '') {
    return rawInputs[key];
  }

  const grouped = [snapshot.motor, snapshot.cognition, snapshot.non_motor, snapshot.autonomic];
  for (const section of grouped) {
    if (section && section[key] !== undefined && section[key] !== null && section[key] !== '') {
      return section[key];
    }
  }

  if (snapshot[key] !== undefined && snapshot[key] !== null && snapshot[key] !== '') {
    return snapshot[key];
  }

  return null;
}

function buildVisitDefaults(snapshot, snapshotCount = 1) {
  const rawInputs = snapshot?.raw_inputs || {};
  const defaults = {};
  for (const field of visitFields) {
    defaults[field.name] = rawInputs[field.name] ?? '';
  }

  const existingDate = defaults.visit_date || snapshot?.visit_date;
  defaults.visit_date = addMonthsToIso(existingDate, 3) || new Date().toISOString().slice(0, 10);

  const yearVal = toNumberOrNull(defaults.YEAR);
  if (yearVal !== null) {
    defaults.YEAR = (yearVal + 0.25).toFixed(2);
  }

  const durationVal = toNumberOrNull(defaults.duration_yrs);
  if (durationVal !== null) {
    defaults.duration_yrs = (durationVal + 0.25).toFixed(2);
  }

  const ageVal = toNumberOrNull(defaults.age);
  if (ageVal !== null) {
    defaults.age = (ageVal + 0.25).toFixed(2);
  }

  defaults.EVENT_ID = `MANUAL_${snapshotCount + 1}`;
  return defaults;
}

function formatTwinDate(value) {
  if (!value) return 'Unknown';
  return value.replace('T', ' ').replace('Z', '');
}

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
  const [visitForm, setVisitForm] = useState({});
  const [loadingList, setLoadingList] = useState(true);
  const [loadingTwin, setLoadingTwin] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [addingVisit, setAddingVisit] = useState(false);
  const [error, setError] = useState('');

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
        const nextSnapshot = latestSnapshot(nextTwin);
        setScenarioForm(buildScenarioDefaults(nextSnapshot));
        setVisitForm(buildVisitDefaults(nextSnapshot, nextTwin?.snapshots?.length || 1));
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

  const forecastChartData = useMemo(() => {
    const baseForecast = Array.isArray(twin?.forecast) ? twin.forecast : [];
    const simulationForecast = Array.isArray(simulation?.forecast) ? simulation.forecast : [];
    return baseForecast.map((point, index) => ({
      horizon: `${point.horizon_months}m`,
      baseline_updrs3: point.predicted_updrs3,
      baseline_moca: point.predicted_moca,
      simulated_updrs3: simulationForecast[index]?.predicted_updrs3 ?? null,
      simulated_moca: simulationForecast[index]?.predicted_moca ?? null,
      delta_updrs3:
        simulationForecast[index]?.predicted_updrs3 != null && point.predicted_updrs3 != null
          ? simulationForecast[index].predicted_updrs3 - point.predicted_updrs3
          : null,
      delta_moca:
        simulationForecast[index]?.predicted_moca != null && point.predicted_moca != null
          ? simulationForecast[index].predicted_moca - point.predicted_moca
          : null,
    }));
  }, [simulation, twin]);

  const currentStats = useMemo(
    () => [
      { label: 'Tremor', value: getSnapshotMetric(selectedSnapshot, 'sym_tremor'), digits: 1 },
      { label: 'Rigidity', value: getSnapshotMetric(selectedSnapshot, 'sym_rigid'), digits: 1 },
      { label: 'Bradykinesia', value: getSnapshotMetric(selectedSnapshot, 'sym_brady'), digits: 1 },
      { label: 'Postural', value: getSnapshotMetric(selectedSnapshot, 'sym_posins'), digits: 1 },
      { label: 'MoCA', value: getSnapshotMetric(selectedSnapshot, 'moca'), digits: 1 },
      { label: 'LEDD', value: getSnapshotMetric(selectedSnapshot, 'LEDD') ?? selectedSnapshot?.ledd, digits: 1 },
      { label: 'UPDRS III OFF', value: getSnapshotMetric(selectedSnapshot, 'updrs3_score'), digits: 1 },
      { label: 'UPDRS III ON', value: getSnapshotMetric(selectedSnapshot, 'updrs3_score_on'), digits: 1 },
      { label: 'Motor Burden', value: twin?.current_state?.motor_burden_index, digits: 2 },
      { label: 'Progression Velocity', value: twin?.current_state?.progression_velocity, digits: 2 },
      { label: 'Treatment Effect', value: twin?.current_state?.treatment_effect, digits: 2 },
      { label: 'Cluster', value: twin?.current_state?.cluster_label, digits: 0 },
    ],
    [selectedSnapshot, twin],
  );

  const scenarioStats = useMemo(
    () => [
      {
        label: 'Tremor',
        value: getSnapshotMetric(simulatedSnapshot, 'sym_tremor'),
        base: getSnapshotMetric(selectedSnapshot, 'sym_tremor'),
        digits: 1,
      },
      {
        label: 'Rigidity',
        value: getSnapshotMetric(simulatedSnapshot, 'sym_rigid'),
        base: getSnapshotMetric(selectedSnapshot, 'sym_rigid'),
        digits: 1,
      },
      {
        label: 'Bradykinesia',
        value: getSnapshotMetric(simulatedSnapshot, 'sym_brady'),
        base: getSnapshotMetric(selectedSnapshot, 'sym_brady'),
        digits: 1,
      },
      {
        label: 'Postural',
        value: getSnapshotMetric(simulatedSnapshot, 'sym_posins'),
        base: getSnapshotMetric(selectedSnapshot, 'sym_posins'),
        digits: 1,
      },
      {
        label: 'MoCA',
        value: getSnapshotMetric(simulatedSnapshot, 'moca'),
        base: getSnapshotMetric(selectedSnapshot, 'moca'),
        digits: 1,
      },
      {
        label: 'LEDD',
        value: getSnapshotMetric(simulatedSnapshot, 'LEDD') ?? simulatedSnapshot?.ledd,
        base: getSnapshotMetric(selectedSnapshot, 'LEDD') ?? selectedSnapshot?.ledd,
        digits: 1,
      },
      {
        label: 'UPDRS III OFF',
        value: getSnapshotMetric(simulatedSnapshot, 'updrs3_score'),
        base: getSnapshotMetric(selectedSnapshot, 'updrs3_score'),
        digits: 1,
      },
      {
        label: 'UPDRS III ON',
        value: getSnapshotMetric(simulatedSnapshot, 'updrs3_score_on'),
        base: getSnapshotMetric(selectedSnapshot, 'updrs3_score_on'),
        digits: 1,
      },
      {
        label: 'Motor Burden',
        value: simulation?.state?.motor_burden_index,
        base: twin?.current_state?.motor_burden_index,
        digits: 2,
      },
      {
        label: 'Progression Velocity',
        value: simulation?.state?.progression_velocity,
        base: twin?.current_state?.progression_velocity,
        digits: 2,
      },
      {
        label: 'Treatment Effect',
        value: simulation?.state?.treatment_effect,
        base: twin?.current_state?.treatment_effect,
        digits: 2,
      },
      {
        label: 'Cluster',
        value: simulation?.state?.cluster_label,
        base: twin?.current_state?.cluster_label,
        digits: 0,
      },
    ],
    [simulation, simulatedSnapshot, selectedSnapshot, twin],
  );

  const forecastSummary = useMemo(() => {
    if (!simulation || forecastChartData.length === 0) return null;
    const tail = forecastChartData[forecastChartData.length - 1];
    return {
      horizon: tail.horizon,
      baselineUpdrs3: tail.baseline_updrs3,
      scenarioUpdrs3: tail.simulated_updrs3,
      baselineMoca: tail.baseline_moca,
      scenarioMoca: tail.simulated_moca,
      deltaUpdrs3: tail.delta_updrs3,
      deltaMoca: tail.delta_moca,
    };
  }, [forecastChartData, simulation]);

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

  async function handleAddVisitSnapshot() {
    if (!selectedTwinId || !twin) return;

    const baseInputs = selectedSnapshot?.raw_inputs || {};
    const payload = { ...baseInputs };

    for (const field of visitFields) {
      const value = visitForm[field.name];
      if (value === '' || value === null || value === undefined) {
        continue;
      }

      if (field.type === 'text' || field.type === 'date') {
        payload[field.name] = value;
      } else {
        const numeric = Number(value);
        if (!Number.isNaN(numeric)) {
          payload[field.name] = numeric;
        }
      }
    }

    setAddingVisit(true);
    setError('');
    try {
      const data = await addTwinSnapshot(selectedTwinId, payload);
      const nextTwin = data?.twin || null;
      if (!nextTwin) {
        throw new Error('Snapshot added but no updated twin was returned');
      }

      setTwin(nextTwin);
      setSimulation(null);
      const nextSnapshot = latestSnapshot(nextTwin);
      setScenarioForm(buildScenarioDefaults(nextSnapshot));
      setVisitForm(buildVisitDefaults(nextSnapshot, nextTwin?.snapshots?.length || 1));

      const listData = await listTwins();
      const nextTwins = Array.isArray(listData?.twins) ? listData.twins : [];
      setTwins(nextTwins);
    } catch (err) {
      setError(err.message || 'Failed to add visit snapshot');
    } finally {
      setAddingVisit(false);
    }
  }

  return (
    <div className={pageShellWide}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="mb-8 text-center">
          <span className={sectionTitle}>Digital Twin</span>
          <h2>Parkinson&apos;s Patient Twin</h2>
          <p className="mx-auto max-w-3xl text-base text-slate-400">
            A longitudinal clinical twin built from assessment data, snapshots, and a baseline
            progression forecast. This phase is additive and does not alter the existing diagnosis flow.
          </p>
        </div>

        {error && (
          <div className={alertClass('danger')}>
            <AlertCircle size={18} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
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
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  <div className={`${innerPanel} bg-black/25`}>
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
                  </div>
                  <div className={`${innerPanel} bg-black/25`}>
                    <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                      <Sparkles size={16} />
                      Confidence
                    </div>
                    <div className="text-xl font-semibold text-white">
                      {(twin.current_state.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className={`${innerPanel} bg-black/25`}>
                    <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                      <Gauge size={16} />
                      Motor Burden
                    </div>
                    <div className="text-xl font-semibold text-white">
                      {twin.current_state.motor_burden_index ?? 'N/A'}
                    </div>
                  </div>
                  <div className={`${innerPanel} bg-black/25`}>
                    <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                      <GitBranch size={16} />
                      Progression Velocity
                    </div>
                    <div className="text-xl font-semibold text-white">
                      {twin.current_state.progression_velocity ?? 'Baseline only'}
                    </div>
                  </div>
                </div>

                <div className="grid gap-6 xl:grid-cols-2">
                  <div className={`${glassPanel} bg-black/25`}>
                    <div className="mb-4 flex items-center justify-between gap-3">
                      <h4 className="text-lg">Current Twin State</h4>
                      <span className={badgeClass('info')}>
                        {selectedSnapshot?.event_id || 'Current'}
                      </span>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2">
                      {currentStats.map((item) => (
                        <div key={item.label} className={`${innerPanel} bg-black/25`}>
                          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                            {item.label}
                          </div>
                          <div className="mt-1 text-lg font-semibold text-white">
                            {formatStatValue(item.value, item.digits)}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-3 mb-2 text-xs text-slate-500">
                      Stats view of latest twin snapshot and computed state.
                    </div>
                    <div className="text-sm text-slate-400">
                      Latest visit: {selectedSnapshot?.visit_date || 'Unknown'} • Snapshot count: {twin.summary?.snapshot_count || twin.snapshots?.length || 0}
                    </div>
                  </div>

                  <div className={`${glassPanel} bg-black/25`}>
                    <div className="mb-4 flex items-center justify-between gap-3">
                      <h4 className="text-lg">Scenario Twin</h4>
                      <span className={badgeClass(simulation ? 'accent' : 'warning')}>
                        {simulation ? simulation.scenario_name : 'Not simulated'}
                      </span>
                    </div>
                    {simulation ? (
                      <>
                        <div className="grid gap-3 sm:grid-cols-2">
                          {scenarioStats.map((item) => {
                            const delta = formatStatDelta(item.value, item.base, item.digits);
                            return (
                              <div key={item.label} className={`${innerPanel} bg-black/25`}>
                                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                                  {item.label}
                                </div>
                                <div className="mt-1 text-lg font-semibold text-white">
                                  {formatStatValue(item.value, item.digits)}
                                </div>
                                {delta !== null && (
                                  <div className={`mt-1 text-xs ${delta.startsWith('+') ? 'text-amber-300' : 'text-emerald-300'}`}>
                                    Delta vs current: {delta}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                        <div className="mt-3 text-sm text-slate-400">
                          Scenario snapshot is simulated and not persisted until you add a real visit.
                        </div>
                      </>
                    ) : (
                      <div className={`${innerPanel} bg-black/25 text-sm text-slate-400`}>
                        Run a scenario to compare a hypothetical future state against the current twin using side-by-side stats.
                      </div>
                    )}
                  </div>
                </div>

                <div className={`${glassPanel} bg-black/25`}>
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <h4 className="text-lg">Forecast Trajectory</h4>
                    <span className={badgeClass(simulation ? 'accent' : 'info')}>
                      {simulation ? 'Baseline vs Scenario' : 'Baseline'}
                    </span>
                  </div>
                  <div className="h-[320px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={forecastChartData}>
                        <CartesianGrid stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                        <XAxis dataKey="horizon" stroke="#94a3b8" />
                        <YAxis stroke="#94a3b8" />
                        <Tooltip
                          contentStyle={{
                            background: '#0a0a0a',
                            border: '1px solid rgba(255,255,255,0.12)',
                            borderRadius: 16,
                          }}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="baseline_updrs3"
                          stroke="#60a5fa"
                          strokeWidth={2}
                          strokeDasharray="4 4"
                          dot={false}
                          name="Baseline UPDRS III"
                        />
                        <Line
                          type="monotone"
                          dataKey="baseline_moca"
                          stroke="#34d399"
                          strokeWidth={2}
                          strokeDasharray="4 4"
                          dot={false}
                          name="Baseline MoCA"
                        />
                        {simulation && (
                          <>
                            <Line
                              type="monotone"
                              dataKey="simulated_updrs3"
                              stroke="#fb923c"
                              strokeWidth={4}
                              dot={{ r: 4, fill: '#fb923c', stroke: '#fff', strokeWidth: 1 }}
                              activeDot={{ r: 6 }}
                              name="Scenario UPDRS III"
                            />
                            <Line
                              type="monotone"
                              dataKey="simulated_moca"
                              stroke="#f472b6"
                              strokeWidth={4}
                              dot={{ r: 4, fill: '#f472b6', stroke: '#fff', strokeWidth: 1 }}
                              activeDot={{ r: 6 }}
                              name="Scenario MoCA"
                            />
                          </>
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  {forecastSummary && (
                    <div className="mt-4 grid gap-3 sm:grid-cols-2">
                      <div className={`${innerPanel} bg-black/25`}>
                        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                          {forecastSummary.horizon} UPDRS III
                        </div>
                        <div className="mt-1 text-sm text-slate-300">
                          Baseline {formatStatValue(forecastSummary.baselineUpdrs3, 1)}
                        </div>
                        <div className="text-sm text-slate-300">
                          Scenario {formatStatValue(forecastSummary.scenarioUpdrs3, 1)}
                        </div>
                        <div
                          className={`mt-1 text-xs ${
                            toNumberOrNull(forecastSummary.deltaUpdrs3) === null
                              ? 'text-slate-400'
                              : toNumberOrNull(forecastSummary.deltaUpdrs3) >= 0
                                ? 'text-amber-300'
                                : 'text-emerald-300'
                          }`}
                        >
                          Delta {formatStatDelta(forecastSummary.deltaUpdrs3, 0, 1) || 'N/A'}
                        </div>
                      </div>
                      <div className={`${innerPanel} bg-black/25`}>
                        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                          {forecastSummary.horizon} MoCA
                        </div>
                        <div className="mt-1 text-sm text-slate-300">
                          Baseline {formatStatValue(forecastSummary.baselineMoca, 1)}
                        </div>
                        <div className="text-sm text-slate-300">
                          Scenario {formatStatValue(forecastSummary.scenarioMoca, 1)}
                        </div>
                        <div
                          className={`mt-1 text-xs ${
                            toNumberOrNull(forecastSummary.deltaMoca) === null
                              ? 'text-slate-400'
                              : toNumberOrNull(forecastSummary.deltaMoca) >= 0
                                ? 'text-emerald-300'
                                : 'text-amber-300'
                          }`}
                        >
                          Delta {formatStatDelta(forecastSummary.deltaMoca, 0, 1) || 'N/A'}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
                  <div className={`${glassPanel} bg-black/25`}>
                    <div className="mb-4 flex items-center gap-2">
                      <PlusCircle size={18} />
                      <h4 className="text-lg">Add Visit Snapshot</h4>
                    </div>
                    <p className="mb-4 text-sm text-slate-400">
                      Add follow-up visits to personalize trajectory and progression velocity from real longitudinal history.
                    </p>
                    <div className="grid gap-4 md:grid-cols-2">
                      {visitFields.map((field) => (
                        <div key={field.name}>
                          <label className={labelText}>{field.label}</label>
                          <input
                            className={inputField}
                            type={field.type}
                            min={field.min}
                            max={field.max}
                            step={field.step}
                            placeholder={field.placeholder}
                            value={visitForm[field.name] ?? ''}
                            onChange={(e) =>
                              setVisitForm((prev) => ({ ...prev, [field.name]: e.target.value }))
                            }
                          />
                        </div>
                      ))}
                    </div>
                    <div className="mt-6 flex flex-wrap gap-3">
                      <button
                        type="button"
                        className={buttonPrimary}
                        onClick={handleAddVisitSnapshot}
                        disabled={addingVisit}
                      >
                        {addingVisit ? (
                          <>
                            <Loader size={16} className="animate-spin" />
                            Saving Visit...
                          </>
                        ) : (
                          <>
                            <PlusCircle size={16} />
                            Add Visit to Twin
                          </>
                        )}
                      </button>
                      <button
                        type="button"
                        className={buttonSecondary}
                        onClick={() =>
                          setVisitForm(
                            buildVisitDefaults(selectedSnapshot, twin?.snapshots?.length || 1),
                          )
                        }
                      >
                        Reset Visit Form
                      </button>
                    </div>

                    <div className="my-6 h-px bg-white/10" />

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
                              setScenarioForm((prev) => ({ ...prev, [field.name]: e.target.value }))
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

                  <div className="space-y-6">
                    <div className={`${glassPanel} bg-black/25`}>
                      <div className="mb-4 flex items-center gap-2">
                        <CalendarDays size={18} />
                        <h4 className="text-lg">Snapshot Timeline</h4>
                      </div>
                      <div className="space-y-3">
                        {twin.snapshots.map((snapshot) => (
                          <div key={snapshot.snapshot_id} className={`${innerPanel} bg-black/25`}>
                            <div className="mb-1 flex items-center justify-between gap-3">
                              <span className="font-semibold text-white">{snapshot.event_id}</span>
                              <span className={badgeClass('info')}>{snapshot.visit_date}</span>
                            </div>
                            <div className="text-sm text-slate-400">
                              LEDD {snapshot.ledd ?? 'N/A'} • Age {snapshot.age_at_visit ?? 'N/A'} • Duration {snapshot.duration_years ?? 'N/A'}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className={`${glassPanel} bg-black/25`}>
                      <div className="mb-4 flex items-center gap-2">
                        <Sparkles size={18} />
                        <h4 className="text-lg">Evidence</h4>
                      </div>
                      <div className="space-y-3 text-sm text-slate-300">
                        {twin.current_state.evidence?.map((item) => (
                          <div key={item} className={`${innerPanel} bg-black/25`}>
                            {item}
                          </div>
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
