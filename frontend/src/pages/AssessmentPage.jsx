import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  Stethoscope,
  User,
  Activity,
  Brain,
  Bed,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  FileText,
  Download,
  Loader,
  Orbit,
} from 'lucide-react';
import { createTwin, predict, generateReport, generateReportPdf } from '../api/client';
import BrainScene from '../components/BrainScene';
import {
  alertClass,
  badgeClass,
  buttonPrimary,
  buttonSecondary,
  buttonSuccess,
  glassPanel,
  inputField,
  labelText,
  pageShellNarrow,
  progressTrack,
  reportShell,
  sectionHeading,
  sectionTitle,
} from '../lib/ui';

const FIELD_SECTIONS = [
  {
    title: 'Patient Demographics',
    icon: <User size={20} />,
    fields: [
      {
        name: 'patient_id',
        label: 'Patient ID',
        type: 'text',
        placeholder: 'Enter patient identifier',
      },
      {
        name: 'age',
        label: 'Age',
        type: 'number',
        min: 18,
        max: 100,
        required: true,
        placeholder: 'Enter age',
      },
      {
        name: 'SEX',
        label: 'Sex',
        type: 'select',
        required: true,
        options: [
          { v: '', l: 'Select' },
          { v: '0', l: 'Female' },
          { v: '1', l: 'Male' },
        ],
      },
      {
        name: 'EDUCYRS',
        label: 'Education Years',
        type: 'number',
        min: 0,
        max: 25,
        required: true,
        placeholder: '0-25',
      },
      {
        name: 'race',
        label: 'Race',
        type: 'select',
        options: [
          { v: '', l: 'Select' },
          { v: '1', l: 'White' },
          { v: '2', l: 'Black/African American' },
          { v: '3', l: 'Asian' },
          { v: '4', l: 'Other' },
        ],
      },
      {
        name: 'BMI',
        label: 'BMI',
        type: 'number',
        min: 15,
        max: 50,
        step: 0.1,
        required: true,
        placeholder: 'Body Mass Index',
      },
      {
        name: 'fampd',
        label: 'Family History of PD',
        type: 'select',
        options: [
          { v: '3', l: 'No family history' },
          { v: '1', l: 'First degree relative' },
          { v: '2', l: 'Other relative' },
        ],
      },
    ],
  },
  {
    title: 'Motor Symptoms (0-4)',
    icon: <Activity size={20} />,
    fields: [
      {
        name: 'sym_tremor',
        label: 'Tremor Severity',
        type: 'select',
        required: true,
        options: [
          { v: '', l: 'Select' },
          { v: '0', l: '0 - None' },
          { v: '1', l: '1 - Mild' },
          { v: '2', l: '2 - Moderate' },
          { v: '3', l: '3 - Severe' },
          { v: '4', l: '4 - Very Severe' },
        ],
      },
      {
        name: 'sym_rigid',
        label: 'Rigidity',
        type: 'select',
        required: true,
        options: [
          { v: '', l: 'Select' },
          { v: '0', l: '0 - None' },
          { v: '1', l: '1 - Mild' },
          { v: '2', l: '2 - Moderate' },
          { v: '3', l: '3 - Severe' },
          { v: '4', l: '4 - Very Severe' },
        ],
      },
      {
        name: 'sym_brady',
        label: 'Bradykinesia',
        type: 'select',
        required: true,
        options: [
          { v: '', l: 'Select' },
          { v: '0', l: '0 - None' },
          { v: '1', l: '1 - Mild' },
          { v: '2', l: '2 - Moderate' },
          { v: '3', l: '3 - Severe' },
          { v: '4', l: '4 - Very Severe' },
        ],
      },
      {
        name: 'sym_posins',
        label: 'Postural Instability',
        type: 'select',
        required: true,
        options: [
          { v: '', l: 'Select' },
          { v: '0', l: '0 - None' },
          { v: '1', l: '1 - Mild' },
          { v: '2', l: '2 - Moderate' },
          { v: '3', l: '3 - Severe' },
          { v: '4', l: '4 - Very Severe' },
        ],
      },
    ],
  },
  {
    title: 'Sleep & Mood',
    icon: <Bed size={20} />,
    fields: [
      {
        name: 'rem',
        label: 'REM Sleep Disorder',
        type: 'select',
        options: [
          { v: '0', l: 'No' },
          { v: '1', l: 'Yes' },
        ],
      },
      {
        name: 'ess',
        label: 'Epworth Sleepiness (0-24)',
        type: 'number',
        min: 0,
        max: 24,
        placeholder: '0-24',
      },
      {
        name: 'gds',
        label: 'Depression Scale (0-15)',
        type: 'number',
        min: 0,
        max: 15,
        placeholder: '0-15',
      },
      {
        name: 'stai',
        label: 'Anxiety Inventory (20-80)',
        type: 'number',
        min: 20,
        max: 80,
        placeholder: '20-80',
      },
    ],
  },
  {
    title: 'Cognitive Assessment',
    icon: <Brain size={20} />,
    fields: [
      {
        name: 'moca',
        label: 'MoCA Score (0-30)',
        type: 'number',
        min: 0,
        max: 30,
        placeholder: '0-30',
      },
      {
        name: 'clockdraw',
        label: 'Clock Drawing (0-4)',
        type: 'number',
        min: 0,
        max: 4,
        placeholder: '0-4',
      },
      {
        name: 'bjlot',
        label: 'Benton Line Orientation (0-30)',
        type: 'number',
        min: 0,
        max: 30,
        placeholder: '0-30',
      },
    ],
  },
];

const CHART_COLORS = ['#10b981', '#ef4444', '#f59e0b', '#3b82f6'];

export default function AssessmentPage() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({});
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [report, setReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [twinLoading, setTwinLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (name, value) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const buildPatientPayload = () => {
    const data = {};
    for (const [k, v] of Object.entries(formData)) {
      if (v !== '') data[k] = isNaN(v) ? v : parseFloat(v);
    }
    return data;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults(null);
    setReport(null);

    try {
      const data = buildPatientPayload();
      const res = await predict(data);
      setResults(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!results) {
      setError('Please run the assessment before generating a report.');
      return;
    }

    setReportLoading(true);
    setError('');

    try {
      const data = buildPatientPayload();
      const res = await generateReport(data, formData.patient_id || null);
      setReport(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setReportLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!results) {
      setError('Please run the assessment before downloading a PDF.');
      return;
    }

    setError('');

    try {
      const data = buildPatientPayload();
      const blob = await generateReportPdf(
        data,
        formData.patient_id || null,
        results,
        report?.report || '',
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pd_assessment_${formData.patient_id || 'patient'}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(`PDF generation failed: ${err.message}`);
    }
  };

  const handleCreateTwin = async () => {
    if (!results) {
      setError('Please run the assessment before creating a digital twin.');
      return;
    }

    setTwinLoading(true);
    setError('');

    try {
      const data = buildPatientPayload();
      const created = await createTwin(data, formData.patient_id || null);
      const twinId = created?.twin_id || created?.twin?.profile?.twin_id;
      if (!twinId) {
        throw new Error('Twin creation succeeded but no twin identifier was returned.');
      }
      navigate(`/twin?id=${encodeURIComponent(twinId)}`);
    } catch (err) {
      setError(err.message || 'Failed to create digital twin');
    } finally {
      setTwinLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({});
    setResults(null);
    setReport(null);
    setError('');
  };

  const chartData = results
    ? Object.entries(results.probabilities).map(([name, prob]) => ({
        name,
        value: +(prob * 100).toFixed(1),
      }))
    : [];

  const confidenceLevel =
    results?.confidence > 0.8
      ? 'High'
      : results?.confidence > 0.6
        ? 'Medium'
        : 'Low';
  const confidenceTextClass =
    results?.confidence > 0.8
      ? 'text-emerald-400'
      : results?.confidence > 0.6
        ? 'text-amber-400'
        : 'text-rose-400';
  const confidenceFillClass =
    results?.confidence > 0.8
      ? 'bg-emerald-500'
      : results?.confidence > 0.6
        ? 'bg-amber-500'
        : 'bg-rose-500';

  return (
    <div className={pageShellNarrow}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="mb-8 text-center">
          <span className={sectionTitle}>Patient Assessment</span>
          <h2>Clinical Data Input</h2>
          <p className="text-base text-slate-400">Enter patient data for AI-powered diagnostic analysis</p>
        </div>

        {error && (
          <div className={alertClass('danger')}>
            <AlertCircle size={18} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {FIELD_SECTIONS.map((section) => (
            <div key={section.title} className={`${glassPanel} mb-6 bg-black/25`}>
              <div className={sectionHeading}>
                {section.icon}
                <h4>{section.title}</h4>
              </div>
              <div className="grid gap-6 md:grid-cols-2">
                {section.fields.map((field) => (
                  <div key={field.name}>
                    <label className={labelText}>
                      {field.label}
                      {field.required && <span className="ml-1 text-rose-300">*</span>}
                    </label>
                    {field.type === 'select' ? (
                      <select
                        className={`${inputField} appearance-none`}
                        value={formData[field.name] || ''}
                        onChange={(e) => handleChange(field.name, e.target.value)}
                        required={field.required}
                      >
                        {field.options.map((option) => (
                          <option key={option.v} value={option.v}>
                            {option.l}
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
                        placeholder={field.placeholder}
                        value={formData[field.name] || ''}
                        onChange={(e) => handleChange(field.name, e.target.value)}
                        required={field.required}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}

          <div className="mt-8 flex flex-col-reverse gap-4 border-t border-white/10 pt-6 sm:flex-row sm:justify-end">
            <button type="button" className={buttonSecondary} onClick={resetForm}>
              <RotateCcw size={16} /> Reset
            </button>
            <button type="submit" className={buttonPrimary} disabled={loading}>
              {loading ? (
                <>
                  <Loader size={16} className="animate-spin" /> Analyzing...
                </>
              ) : (
                <>
                  <Stethoscope size={16} /> Run Assessment
                </>
              )}
            </button>
          </div>
        </form>

        {results && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`${glassPanel} mt-8 bg-black/25`}
          >
            <h3 className="mb-6 flex items-center gap-2">
              <CheckCircle size={22} className="text-emerald-300" />
              Assessment Results
            </h3>

            <div className="mb-8 h-[400px] overflow-hidden rounded-2xl border border-white/10 bg-black/30">
              <BrainScene symptomData={formData} />
            </div>

            <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)]">
              <div>
                <span className={sectionTitle}>Primary Diagnosis</span>
                <h2 className="mt-1 text-3xl">{results.prediction}</h2>
                <p className="text-sm text-slate-400">Based on multimodal ML analysis</p>
              </div>
              <div>
                <span className={sectionTitle}>Confidence</span>
                <div className="mt-1 flex items-baseline gap-3">
                  <h2 className={`text-3xl ${confidenceTextClass}`}>
                    {(results.confidence * 100).toFixed(1)}%
                  </h2>
                  <span className={badgeClass(
                    results.confidence > 0.8 ? 'success' : results.confidence > 0.6 ? 'warning' : 'danger'
                  )}>
                    {confidenceLevel}
                  </span>
                </div>
                <div className="mt-3">
                  <div className={progressTrack}>
                    <div
                      className={`h-full rounded-full ${confidenceFillClass}`}
                      style={{
                        width: `${(results.confidence * 100).toFixed(1)}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8">
              <h4 className="mb-4 text-lg">Probability Distribution</h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData} barSize={50}>
                  <XAxis
                    dataKey="name"
                    tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--bg-secondary)',
                      border: '1px solid rgba(255,255,255,0.12)',
                      borderRadius: 8,
                      color: 'var(--text-primary)',
                    }}
                    formatter={(v) => [`${v}%`, 'Probability']}
                  />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                    {chartData.map((item, i) => (
                      <Cell
                        key={item.name}
                        fill={CHART_COLORS[i % CHART_COLORS.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-6 flex flex-col gap-4 border-t border-white/10 pt-6 sm:flex-row sm:justify-end">
              <button
                className={buttonPrimary}
                onClick={handleGenerateReport}
                disabled={reportLoading || !results}
                type="button"
              >
                {reportLoading ? (
                  <>
                    <Loader size={16} className="animate-spin" /> Generating...
                  </>
                ) : (
                  <>
                    <FileText size={16} /> Generate Report
                  </>
                )}
              </button>
              <button
                className={buttonSuccess}
                onClick={handleDownloadPdf}
                disabled={!results}
                type="button"
              >
                <Download size={16} /> Download PDF
              </button>
              <button
                className={buttonSecondary}
                onClick={handleCreateTwin}
                disabled={!results || twinLoading}
                type="button"
              >
                {twinLoading ? (
                  <>
                    <Loader size={16} className="animate-spin" /> Creating Twin...
                  </>
                ) : (
                  <>
                    <Orbit size={16} /> Create Digital Twin
                  </>
                )}
              </button>
            </div>
          </motion.div>
        )}

        {report && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`${glassPanel} mt-6 bg-black/25`}
          >
            <h4 className="mb-4 flex items-center gap-2 text-lg">
              <FileText size={18} />
              Medical Report
            </h4>
            <pre className={reportShell}>{report.report}</pre>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
