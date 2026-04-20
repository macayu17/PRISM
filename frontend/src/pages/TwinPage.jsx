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
import BrainScene from '../components/BrainScene';
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

function buildVisitDefaults(snapshot) {
  const rawInputs = snapshot?.raw_inputs || {};
  const defaults = {};
  for (const field of visitFields) {
    defaults[field.name] = rawInputs[field.name] ?? '';
  }
  defaults.visit_date = defaults.visit_date || new Date().toISOString().slice(0, 10);
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
        setVisitForm(buildVisitDefaults(nextSnapshot));
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
    }));
  }, [simulation, twin]);

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
      setVisitForm(buildVisitDefaults(nextSnapshot));

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
                    <div className="mb-4 h-[320px] overflow-hidden rounded-[1.5rem] border border-white/10 bg-black/30">
                      <BrainScene symptomData={selectedSnapshot?.raw_inputs || {}} />
                    </div>
                    <div className="mb-2 text-xs text-slate-500">
                      Visual symptom map only, not diagnostic imaging.
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
                    <div className="mb-4 h-[320px] overflow-hidden rounded-[1.5rem] border border-white/10 bg-black/30">
                      <BrainScene symptomData={simulatedSnapshot?.raw_inputs || selectedSnapshot?.raw_inputs || {}} />
                    </div>
                    <div className="mb-2 text-xs text-slate-500">
                      Same visual map rendered with simulated values.
                    </div>
                    <div className="text-sm text-slate-400">
                      {simulation
                        ? 'Overlay reflects the simulated latest snapshot without saving it.'
                        : 'Run a scenario to compare a hypothetical future state against the current twin.'}
                    </div>
                  </div>
                </div>

                <div className={`${glassPanel} bg-black/25`}>
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <h4 className="text-lg">Forecast Trajectory</h4>
                    <span className={badgeClass('info')}>Heuristic v1</span>
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
                        <Line type="monotone" dataKey="baseline_updrs3" stroke="#38bdf8" strokeWidth={2} name="Baseline UPDRS III" />
                        <Line type="monotone" dataKey="baseline_moca" stroke="#10b981" strokeWidth={2} name="Baseline MoCA" />
                        {simulation && (
                          <>
                            <Line type="monotone" dataKey="simulated_updrs3" stroke="#f59e0b" strokeWidth={2} strokeDasharray="6 4" name="Scenario UPDRS III" />
                            <Line type="monotone" dataKey="simulated_moca" stroke="#f472b6" strokeWidth={2} strokeDasharray="6 4" name="Scenario MoCA" />
                          </>
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
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
                        onClick={() => setVisitForm(buildVisitDefaults(selectedSnapshot))}
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
