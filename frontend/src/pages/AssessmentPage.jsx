import { useState } from 'react';
import { motion } from 'framer-motion';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import {
    Stethoscope, User, Activity, Brain, Bed, RotateCcw,
    CheckCircle, AlertCircle, FileText, Download, Loader
} from 'lucide-react';
import { predict, generateReport, generateReportPdf } from '../api/client';
import BrainScene from '../components/BrainScene';
import './AssessmentPage.css';

const FIELD_SECTIONS = [
    {
        title: 'Patient Demographics',
        icon: <User size={20} />,
        fields: [
            { name: 'patient_id', label: 'Patient ID', type: 'text', placeholder: 'Enter patient identifier' },
            { name: 'age', label: 'Age', type: 'number', min: 18, max: 100, required: true, placeholder: 'Enter age' },
            { name: 'SEX', label: 'Sex', type: 'select', required: true, options: [{ v: '', l: 'Select' }, { v: '0', l: 'Female' }, { v: '1', l: 'Male' }] },
            { name: 'EDUCYRS', label: 'Education Years', type: 'number', min: 0, max: 25, required: true, placeholder: '0-25' },
            { name: 'BMI', label: 'BMI', type: 'number', min: 15, max: 50, step: 0.1, required: true, placeholder: 'Body Mass Index' },
            { name: 'fampd', label: 'Family History of PD', type: 'select', options: [{ v: '0', l: 'No' }, { v: '1', l: 'Yes' }] },
        ],
    },
    {
        title: 'Motor Symptoms (0-4)',
        icon: <Activity size={20} />,
        fields: [
            { name: 'sym_tremor', label: 'Tremor Severity', type: 'select', required: true, options: [{ v: '', l: 'Select' }, { v: '0', l: '0 - None' }, { v: '1', l: '1 - Mild' }, { v: '2', l: '2 - Moderate' }, { v: '3', l: '3 - Severe' }, { v: '4', l: '4 - Very Severe' }] },
            { name: 'sym_rigid', label: 'Rigidity', type: 'select', required: true, options: [{ v: '', l: 'Select' }, { v: '0', l: '0 - None' }, { v: '1', l: '1 - Mild' }, { v: '2', l: '2 - Moderate' }, { v: '3', l: '3 - Severe' }, { v: '4', l: '4 - Very Severe' }] },
            { name: 'sym_brady', label: 'Bradykinesia', type: 'select', required: true, options: [{ v: '', l: 'Select' }, { v: '0', l: '0 - None' }, { v: '1', l: '1 - Mild' }, { v: '2', l: '2 - Moderate' }, { v: '3', l: '3 - Severe' }, { v: '4', l: '4 - Very Severe' }] },
            { name: 'sym_posins', label: 'Postural Instability', type: 'select', required: true, options: [{ v: '', l: 'Select' }, { v: '0', l: '0 - None' }, { v: '1', l: '1 - Mild' }, { v: '2', l: '2 - Moderate' }, { v: '3', l: '3 - Severe' }, { v: '4', l: '4 - Very Severe' }] },
        ],
    },
    {
        title: 'Sleep & Mood',
        icon: <Bed size={20} />,
        fields: [
            { name: 'rem', label: 'REM Sleep Disorder', type: 'select', options: [{ v: '0', l: 'No' }, { v: '1', l: 'Yes' }] },
            { name: 'ess', label: 'Epworth Sleepiness (0-24)', type: 'number', min: 0, max: 24, placeholder: '0-24' },
            { name: 'gds', label: 'Depression Scale (0-15)', type: 'number', min: 0, max: 15, placeholder: '0-15' },
            { name: 'stai', label: 'Anxiety Inventory (20-80)', type: 'number', min: 20, max: 80, placeholder: '20-80' },
        ],
    },
    {
        title: 'Cognitive Assessment',
        icon: <Brain size={20} />,
        fields: [
            { name: 'moca', label: 'MoCA Score (0-30)', type: 'number', min: 0, max: 30, placeholder: '0-30' },
            { name: 'clockdraw', label: 'Clock Drawing (0-4)', type: 'number', min: 0, max: 4, placeholder: '0-4' },
            { name: 'bjlot', label: 'Benton Line Orientation (0-30)', type: 'number', min: 0, max: 30, placeholder: '0-30' },
        ],
    },
];

const CHART_COLORS = ['#10b981', '#ef4444', '#f59e0b', '#3b82f6'];

export default function AssessmentPage() {
    const [formData, setFormData] = useState({});
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [report, setReport] = useState(null);
    const [reportLoading, setReportLoading] = useState(false);
    const [error, setError] = useState('');

    const handleChange = (name, value) => {
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResults(null);
        setReport(null);

        try {
            const data = {};
            for (const [k, v] of Object.entries(formData)) {
                if (v !== '') data[k] = isNaN(v) ? v : parseFloat(v);
            }
            const res = await predict(data);
            setResults(res);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleGenerateReport = async () => {
        setReportLoading(true);
        try {
            const data = {};
            for (const [k, v] of Object.entries(formData)) {
                if (v !== '') data[k] = isNaN(v) ? v : parseFloat(v);
            }
            const res = await generateReport(data, formData.patient_id || null);
            setReport(res);
        } catch (err) {
            setError(err.message);
        } finally {
            setReportLoading(false);
        }
    };

    const handleDownloadPdf = async () => {
        try {
            const data = {};
            for (const [k, v] of Object.entries(formData)) {
                if (v !== '') data[k] = isNaN(v) ? v : parseFloat(v);
            }
            const blob = await generateReportPdf(data, formData.patient_id || null, results, report?.report);
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `pd_assessment_${formData.patient_id || 'patient'}.pdf`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            setError('PDF generation failed: ' + err.message);
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
            name, value: +(prob * 100).toFixed(1),
        }))
        : [];

    const confidenceLevel = results?.confidence > 0.8 ? 'High' : results?.confidence > 0.6 ? 'Medium' : 'Low';
    const confidenceColor = results?.confidence > 0.8 ? '#10b981' : results?.confidence > 0.6 ? '#f59e0b' : '#ef4444';

    return (
        <div className="container" style={{ maxWidth: 800 }}>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                    <span className="section-title">Patient Assessment</span>
                    <h2>Clinical Data Input</h2>
                    <p>Enter patient data for AI-powered diagnostic analysis</p>
                </div>

                {error && (
                    <div className="alert alert-danger">
                        <AlertCircle size={18} />
                        <span>{error}</span>
                    </div>
                )}

                <form onSubmit={handleSubmit}>
                    {FIELD_SECTIONS.map((section, si) => (
                        <div key={si} className="glass-card-static" style={{ marginBottom: '1.5rem' }}>
                            <div className="section-header">
                                {section.icon}
                                <h4>{section.title}</h4>
                            </div>
                            <div className="form-grid">
                                {section.fields.map((f) => (
                                    <div key={f.name} className="form-group">
                                        <label className="form-label">
                                            {f.label}
                                            {f.required && <span className="required">*</span>}
                                        </label>
                                        {f.type === 'select' ? (
                                            <select
                                                className="form-select"
                                                value={formData[f.name] || ''}
                                                onChange={(e) => handleChange(f.name, e.target.value)}
                                                required={f.required}
                                            >
                                                {f.options.map((o) => (
                                                    <option key={o.v} value={o.v}>{o.l}</option>
                                                ))}
                                            </select>
                                        ) : (
                                            <input
                                                className="form-input"
                                                type={f.type}
                                                min={f.min}
                                                max={f.max}
                                                step={f.step}
                                                placeholder={f.placeholder}
                                                value={formData[f.name] || ''}
                                                onChange={(e) => handleChange(f.name, e.target.value)}
                                                required={f.required}
                                            />
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}

                    <div className="form-actions">
                        <button type="button" className="btn-secondary" onClick={resetForm}>
                            <RotateCcw size={16} /> Reset
                        </button>
                        <button type="submit" className="btn-primary" disabled={loading}>
                            {loading ? <><Loader size={16} className="spin" /> Analyzing...</> : <><Stethoscope size={16} /> Run Assessment</>}
                        </button>
                    </div>
                </form>

                {/* Results */}
                {results && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass-card-static results-section"
                        style={{ marginTop: '2rem' }}
                    >
                        <h3 style={{ marginBottom: '1.5rem' }}>
                            <CheckCircle size={22} style={{ color: 'var(--success)', marginRight: '0.5rem' }} />
                            Assessment Results
                        </h3>

                        {/* Digital Twin Visualization */}
                        <div style={{ height: 400, marginBottom: '2rem', borderRadius: 12, overflow: 'hidden', background: 'rgba(0,0,0,0.1)', border: '1px solid var(--border-color)' }}>
                            <BrainScene symptomData={formData} />
                        </div>

                        <div className="results-grid">
                            <div>
                                <span className="section-title">Primary Diagnosis</span>
                                <h2 style={{ fontSize: '1.5rem', marginTop: '0.25rem' }}>{results.prediction}</h2>
                                <p style={{ fontSize: '0.85rem' }}>Based on multimodal ML analysis</p>
                            </div>
                            <div>
                                <span className="section-title">Confidence</span>
                                <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginTop: '0.25rem' }}>
                                    <h2 style={{ fontSize: '1.5rem', color: confidenceColor }}>
                                        {(results.confidence * 100).toFixed(1)}%
                                    </h2>
                                    <span className={`badge ${results.confidence > 0.8 ? 'badge-success' : results.confidence > 0.6 ? 'badge-warning' : 'badge-danger'}`}>
                                        {confidenceLevel}
                                    </span>
                                </div>
                                <div className="progress-bar" style={{ marginTop: '0.5rem' }}>
                                    <div className="progress-fill" style={{
                                        width: `${(results.confidence * 100).toFixed(1)}%`,
                                        background: confidenceColor,
                                    }} />
                                </div>
                            </div>
                        </div>

                        <div style={{ marginTop: '2rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>Probability Distribution</h4>
                            <ResponsiveContainer width="100%" height={250}>
                                <BarChart data={chartData} barSize={50}>
                                    <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                                    <YAxis domain={[0, 100]} tick={{ fill: 'var(--text-muted)', fontSize: 12 }} tickFormatter={v => `${v}%`} />
                                    <Tooltip
                                        contentStyle={{
                                            background: 'var(--bg-secondary)',
                                            border: '1px solid var(--border-color)',
                                            borderRadius: 8,
                                            color: 'var(--text-primary)',
                                        }}
                                        formatter={(v) => [`${v}%`, 'Probability']}
                                    />
                                    <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                                        {chartData.map((_, i) => (
                                            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="report-actions" style={{ marginTop: '1.5rem' }}>
                            <button className="btn-primary" onClick={handleGenerateReport} disabled={reportLoading}>
                                {reportLoading ? <><Loader size={16} className="spin" /> Generating...</> : <><FileText size={16} /> Generate Report</>}
                            </button>
                            <button className="btn-success" onClick={handleDownloadPdf}>
                                <Download size={16} /> Download PDF
                            </button>
                        </div>
                    </motion.div>
                )}

                {/* Report */}
                {report && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass-card-static"
                        style={{ marginTop: '1.5rem' }}
                    >
                        <h4 style={{ marginBottom: '1rem' }}>
                            <FileText size={18} style={{ marginRight: '0.5rem' }} />
                            Medical Report
                        </h4>
                        <pre className="report-content">{report.report}</pre>
                    </motion.div>
                )}
            </motion.div>
        </div>
    );
}
