const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const DEMO_MODE = !API_BASE;
const STATIC_DEMO_MODE = DEMO_MODE && import.meta.env.PROD;
const BACKEND_UNAVAILABLE_MESSAGE =
  "Live API is unavailable. Using static interview demo data.";
const DEMO_TWINS_KEY = "neuroassess_demo_twins_v1";
const DEMO_DOCS_KEY = "neuroassess_demo_docs_v1";

const diagnosisLabels = [
  "Healthy Control",
  "Parkinson's Disease",
  "SWEDD",
  "Prodromal PD",
];

const demoDocuments = [
  {
    id: "demo-mds-criteria",
    title: "MDS Clinical Diagnostic Criteria",
    type: "guideline",
    metadata: {
      title: "MDS Clinical Diagnostic Criteria",
      authors: "Movement Disorder Society",
      source: "Clinical criteria",
      year: "2015",
      size_bytes: 184000,
    },
    preview:
      "Evidence summary covering bradykinesia, tremor, rigidity, exclusion criteria, and supportive clinical features used for Parkinson's disease assessment.",
    content:
      "MDS clinical criteria organize Parkinson's disease diagnosis around parkinsonism, supportive criteria, red flags, and absolute exclusion criteria. This demo document is used to show the retrieval and report workflow without requiring the local Flask index.",
  },
  {
    id: "demo-updrs",
    title: "MDS-UPDRS Motor Examination Reference",
    type: "paper",
    metadata: {
      title: "MDS-UPDRS Motor Examination Reference",
      authors: "MDS Task Force",
      source: "UPDRS motor assessment",
      year: "2019",
      size_bytes: 226000,
    },
    preview:
      "Motor rating context for tremor, rigidity, bradykinesia, postural instability, treatment state, and longitudinal follow-up scoring.",
    content:
      "The MDS-UPDRS motor examination provides structured scoring for motor burden and longitudinal change. NeuroAssess uses these concepts to explain model output, simulated progression, and digital twin snapshots.",
  },
  {
    id: "demo-ppmi",
    title: "PPMI Cohort Feature Overview",
    type: "paper",
    metadata: {
      title: "PPMI Cohort Feature Overview",
      authors: "PPMI",
      source: "Research cohort",
      year: "2024",
      size_bytes: 142000,
    },
    preview:
      "Cohort-level feature context for demographics, non-motor symptoms, cognition, family history, and diagnostic class labels.",
    content:
      "The PPMI cohort provides structured clinical and research features for Parkinson's disease modelling. This demo summary mirrors the fields used in the assessment form.",
  },
  {
    id: "demo-mds-nice-workup",
    title: "PD Diagnostic Workup Reference",
    type: "guideline",
    metadata: {
      title: "PD Diagnostic Workup Reference",
      authors: "NeuroAssess curated reference",
      source: "MDS and NICE clinical guidance",
      year: "2026",
      size_bytes: 2760,
    },
    preview:
      "Structured diagnostic-workup notes for bradykinesia, tremor, rigidity, red flags, exclusion criteria, and specialist review.",
    content:
      "This curated reference summarizes MDS and NICE diagnostic reasoning for Parkinson's disease. It supports report sections that discuss parkinsonism, bradykinesia, rest tremor, rigidity, postural instability, differential diagnosis, red flags, and limits of single-test diagnosis.",
  },
  {
    id: "demo-non-motor-reference",
    title: "Non-Motor Symptoms Reference",
    type: "paper",
    metadata: {
      title: "Non-Motor Symptoms Reference",
      authors: "NeuroAssess curated reference",
      source: "Parkinson's Foundation and NICE",
      year: "2026",
      size_bytes: 3190,
    },
    preview:
      "Sleep, mood, cognition, autonomic, fatigue, pain, speech, swallowing, and gastrointestinal context for PD reports.",
    content:
      "This reference covers non-motor Parkinson's disease context, including REM sleep behavior disorder, depression, anxiety, apathy, cognitive changes, constipation, dizziness, fatigue, pain, speech and swallowing issues, and how these signals should be interpreted alongside motor findings.",
  },
  {
    id: "demo-rehab-care-reference",
    title: "Rehabilitation and Supportive Care Reference",
    type: "guideline",
    metadata: {
      title: "Rehabilitation and Supportive Care Reference",
      authors: "NeuroAssess curated reference",
      source: "Parkinson's Foundation and NICE",
      year: "2026",
      size_bytes: 2630,
    },
    preview:
      "Exercise, physical therapy, occupational therapy, speech therapy, gait, balance, and fall-risk support guidance.",
    content:
      "This supportive-care reference summarizes rehabilitation concepts used in reports when patients show gait difficulty, postural instability, freezing, speech or swallowing concerns, occupational limitations, or high motor burden.",
  },
];

function buildUrl(path) {
  return `${API_BASE}${path}`;
}

function isJsonResponse(res) {
  return (res.headers.get("Content-Type") || "").includes("application/json");
}

async function parseErrorResponse(res) {
  if (!isJsonResponse(res)) {
    return BACKEND_UNAVAILABLE_MESSAGE;
  }

  try {
    const data = await res.json();
    return data?.error || data?.message || res.statusText || "Request failed";
  } catch {
    return res.statusText || "Request failed";
  }
}

function isDemoFallbackError(err) {
  return (
    err?.message === BACKEND_UNAVAILABLE_MESSAGE ||
    err?.name === "TypeError" ||
    String(err?.message || "").includes("Failed to fetch")
  );
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };

  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(buildUrl(path), {
    ...options,
    headers,
  });

  if (!res.ok) {
    throw new Error(await parseErrorResponse(res));
  }

  if (path.startsWith("/api/") && !isJsonResponse(res)) {
    throw new Error(BACKEND_UNAVAILABLE_MESSAGE);
  }

  return res;
}

async function withDemoFallback(action, fallback) {
  if (STATIC_DEMO_MODE) {
    return fallback();
  }

  try {
    return await action();
  } catch (err) {
    if (DEMO_MODE && isDemoFallbackError(err)) {
      return fallback();
    }
    throw err;
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function toNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function round(value, digits = 2) {
  return Number(value.toFixed(digits));
}

function normalizeProbabilities(raw) {
  const total = raw.reduce((sum, value) => sum + value, 0) || 1;
  return Object.fromEntries(
    diagnosisLabels.map((label, index) => [label, round(raw[index] / total, 3)]),
  );
}

function demoPrediction(patientData = {}) {
  const motorScore =
    toNumber(patientData.sym_tremor) +
    toNumber(patientData.sym_rigid) +
    toNumber(patientData.sym_brady) +
    toNumber(patientData.sym_posins);
  const moca = toNumber(patientData.moca, 28);
  const age = toNumber(patientData.age, 58);
  const rem = toNumber(patientData.rem);
  const family = toNumber(patientData.fampd) === 1 ? 1 : 0;

  const motorBurden = clamp(motorScore / 16, 0, 1);
  const cognitiveBurden = clamp((30 - moca) / 12, 0, 1);
  const risk =
    motorBurden * 0.55 +
    cognitiveBurden * 0.2 +
    rem * 0.08 +
    family * 0.08 +
    (age > 65 ? 0.09 : 0);

  const raw = [
    clamp(1.05 - risk * 1.4, 0.05, 0.9),
    clamp(risk * 1.35 + motorBurden * 0.45, 0.08, 1),
    clamp(0.18 + motorBurden * 0.25 - rem * 0.04, 0.05, 0.5),
    clamp(0.12 + rem * 0.22 + family * 0.15 + cognitiveBurden * 0.2, 0.05, 0.65),
  ];
  const probabilities = normalizeProbabilities(raw);
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return {
    prediction: sorted[0][0],
    confidence: sorted[0][1],
    probabilities,
    demo_mode: true,
  };
}

function demoReport(patientData, patientId) {
  const prediction = demoPrediction(patientData);
  const symptomSummary = [
    `Tremor: ${toNumber(patientData.sym_tremor)}/4`,
    `Rigidity: ${toNumber(patientData.sym_rigid)}/4`,
    `Bradykinesia: ${toNumber(patientData.sym_brady)}/4`,
    `Postural instability: ${toNumber(patientData.sym_posins)}/4`,
    `MoCA: ${patientData.moca ?? "not provided"}`,
  ].join("\n- ");

  return {
    report: `NEUROASSESS DEMO REPORT

Patient: ${patientId || patientData.patient_id || "Demo patient"}
Primary impression: ${prediction.prediction}
Confidence: ${(prediction.confidence * 100).toFixed(1)}%

Key inputs:
- ${symptomSummary}

Interpretation:
This static Vercel demo uses the same frontend workflow as the live Flask-backed app, with deterministic demo scoring for interview use. Connect VITE_API_BASE_URL to the Flask API for production model inference, document retrieval, and PDF generation.

Clinical note:
This report is decision-support output for demonstration and educational review only. It is not a medical diagnosis.`,
    filename: `demo_report_${patientId || "patient"}.txt`,
    demo_mode: true,
  };
}

function escapePdfText(value) {
  return String(value)
    .replace(/[^\x20-\x7E]/g, " ")
    .replace(/[\\()]/g, "\\$&")
    .replace(/\r?\n/g, " ");
}

function wrapPdfLines(value, maxLength = 78, maxLines = 8) {
  const words = String(value || "")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .filter(Boolean);
  const lines = [];
  let current = "";

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > maxLength) {
      if (current) lines.push(current);
      current = word;
    } else {
      current = next;
    }
    if (lines.length >= maxLines) break;
  }

  if (current && lines.length < maxLines) lines.push(current);
  return lines;
}

function pdfRgb(hex) {
  const clean = hex.replace("#", "");
  return [0, 2, 4]
    .map((index) => parseInt(clean.slice(index, index + 2), 16) / 255)
    .map((value) => value.toFixed(3))
    .join(" ");
}

function pdfRect(x, y, width, height, fill) {
  return `${pdfRgb(fill)} rg ${x} ${y} ${width} ${height} re f`;
}

function pdfText(text, x, y, size = 10, font = "F1", fill = "#111827") {
  return `${pdfRgb(fill)} rg BT /${font} ${size} Tf ${x} ${y} Td (${escapePdfText(text)}) Tj ET`;
}

function buildDemoPdfBlob(text, options = {}) {
  const patientData = options.patientData || {};
  const predictionResults = options.predictionResults || demoPrediction(patientData);
  const patientId = options.patientId || patientData.patient_id || "Demo patient";
  const generatedAt = new Date().toLocaleDateString();
  const probabilities = predictionResults?.probabilities || {};
  const reportLines = String(text || "NeuroAssess demo report")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !line.match(/^[-=]{4,}$/))
    .slice(0, 18);

  const commands = [
    pdfRect(0, 0, 612, 792, "#f8fafc"),
    pdfRect(0, 684, 612, 108, "#071312"),
    pdfRect(0, 684, 612, 5, "#5eead4"),
    pdfText("NEUROASSESS", 48, 750, 24, "F2", "#ffffff"),
    pdfText("Parkinson's Disease Assessment Report", 50, 728, 11, "F1", "#a7f3d0"),
    pdfText(`Generated ${generatedAt}`, 452, 752, 9, "F1", "#cbd5e1"),
    pdfText("Research and educational decision-support output", 354, 728, 9, "F1", "#94a3b8"),
    pdfRect(44, 594, 248, 64, "#ffffff"),
    pdfRect(44, 594, 5, 64, "#5eead4"),
    pdfText("PRIMARY IMPRESSION", 64, 635, 8, "F2", "#0f766e"),
    pdfText(predictionResults.prediction || "Demo result", 64, 614, 16, "F2", "#0f172a"),
    pdfRect(320, 594, 248, 64, "#ffffff"),
    pdfRect(320, 594, 5, 64, "#38bdf8"),
    pdfText("CONFIDENCE", 340, 635, 8, "F2", "#0369a1"),
    pdfText(`${((predictionResults.confidence || 0) * 100).toFixed(1)}%`, 340, 614, 18, "F2", "#0f172a"),
    pdfText(`Patient: ${patientId}`, 48, 565, 10, "F2", "#334155"),
    pdfText(`Age: ${patientData.age || "N/A"}    Sex: ${toNumber(patientData.SEX) === 1 ? "Male" : "Female"}`, 48, 548, 9, "F1", "#475569"),
    pdfText("Probability Distribution", 48, 516, 13, "F2", "#0f172a"),
  ];

  let barY = 492;
  const palette = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6"];
  Object.entries(probabilities).slice(0, 4).forEach(([label, prob], index) => {
    const pct = clamp(toNumber(prob) * 100, 0, 100);
    commands.push(pdfText(label, 54, barY + 2, 8, "F1", "#475569"));
    commands.push(pdfRect(184, barY, 200, 8, "#e2e8f0"));
    commands.push(pdfRect(184, barY, Math.max(2, pct * 2), 8, palette[index % palette.length]));
    commands.push(pdfText(`${pct.toFixed(1)}%`, 398, barY + 2, 8, "F2", "#334155"));
    barY -= 22;
  });

  commands.push(pdfText("Clinical Summary", 48, 374, 13, "F2", "#0f172a"));
  let y = 352;
  for (const rawLine of reportLines) {
    const isHeading = rawLine.endsWith(":") || rawLine === rawLine.toUpperCase();
    const lines = wrapPdfLines(rawLine.replace(/^[-•]\s*/, ""), isHeading ? 60 : 86, isHeading ? 1 : 3);
    for (const line of lines) {
      commands.push(pdfText(line, 54, y, isHeading ? 10 : 8.5, isHeading ? "F2" : "F1", isHeading ? "#0f766e" : "#334155"));
      y -= isHeading ? 15 : 12;
      if (y < 96) break;
    }
    if (y < 96) break;
  }

  commands.push(pdfRect(44, 42, 524, 38, "#ecfeff"));
  commands.push(pdfText("Clinical note", 58, 62, 8, "F2", "#0f766e"));
  commands.push(pdfText("Generated for demonstration and educational review only; not a medical diagnosis.", 122, 62, 8, "F1", "#334155"));

  const content = commands.join("\n");
  const objects = [
    "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
    "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
    "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >> endobj",
    "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
    "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> endobj",
    `6 0 obj << /Length ${content.length} >> stream\n${content}\nendstream endobj`,
  ];
  let pdf = "%PDF-1.4\n";
  const offsets = [0];
  for (const object of objects) {
    offsets.push(pdf.length);
    pdf += `${object}\n`;
  }
  const xrefStart = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  pdf += offsets
    .slice(1)
    .map((offset) => `${String(offset).padStart(10, "0")} 00000 n \n`)
    .join("");
  pdf += `trailer << /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`;
  return new Blob([pdf], { type: "application/pdf" });
}

function readJsonStorage(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function writeJsonStorage(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Local storage may be blocked in some interview browsers; the app still works for the current session.
  }
}

function documentCounts(documents) {
  return {
    total: documents.length,
    paper: documents.filter((doc) => doc.type === "paper").length,
    guideline: documents.filter((doc) => doc.type === "guideline").length,
    textbook: documents.filter((doc) => doc.type === "textbook").length,
  };
}

function getDemoDocuments() {
  const userDocs = readJsonStorage(DEMO_DOCS_KEY, []);
  return [...demoDocuments, ...userDocs];
}

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function snapshotFromPatient(patientData = {}, index = 0) {
  return {
    snapshot_id: `demo_snap_${Date.now()}_${index}`,
    event_id: patientData.EVENT_ID || (index === 0 ? "BASELINE" : `VISIT_${index + 1}`),
    visit_date: patientData.visit_date || todayIso(),
    year_index: toNumber(patientData.YEAR, index * 0.5),
    age_at_visit: toNumber(patientData.age_at_visit ?? patientData.age, 62 + index),
    duration_years: toNumber(patientData.duration_yrs, 2 + index * 0.5),
    ledd: toNumber(patientData.LEDD, 450),
    raw_inputs: { ...patientData },
    motor: {
      sym_tremor: toNumber(patientData.sym_tremor, 2),
      sym_rigid: toNumber(patientData.sym_rigid, 2),
      sym_brady: toNumber(patientData.sym_brady, 2),
      sym_posins: toNumber(patientData.sym_posins, 1),
      updrs3_score: toNumber(patientData.updrs3_score, 22 + index * 4),
      updrs3_score_on: toNumber(patientData.updrs3_score_on, 16 + index * 3),
    },
    cognition: {
      moca: toNumber(patientData.moca, 26),
      clockdraw: toNumber(patientData.clockdraw, 3),
      bjlot: toNumber(patientData.bjlot, 24),
    },
    non_motor: {
      ess: toNumber(patientData.ess, 8),
      gds: toNumber(patientData.gds, 5),
      stai: toNumber(patientData.stai, 38),
      rem: toNumber(patientData.rem, 1),
    },
    autonomic: {},
  };
}

function burdenFromSnapshot(snapshot) {
  const motor =
    (toNumber(snapshot.motor.sym_tremor) +
      toNumber(snapshot.motor.sym_rigid) +
      toNumber(snapshot.motor.sym_brady) +
      toNumber(snapshot.motor.sym_posins)) /
    16;
  const cognitive = clamp((30 - toNumber(snapshot.cognition.moca, 28)) / 30, 0, 1);
  const nonMotor =
    (toNumber(snapshot.non_motor.ess) / 24 +
      toNumber(snapshot.non_motor.gds) / 15 +
      clamp((toNumber(snapshot.non_motor.stai, 20) - 20) / 60, 0, 1) +
      toNumber(snapshot.non_motor.rem)) /
    4;
  return {
    motor: round(clamp(motor, 0, 1), 2),
    cognitive: round(cognitive, 2),
    nonMotor: round(clamp(nonMotor, 0, 1), 2),
  };
}

function progressionVelocity(snapshots) {
  if (snapshots.length < 2) return null;
  const first = burdenFromSnapshot(snapshots[0]);
  const last = burdenFromSnapshot(snapshots[snapshots.length - 1]);
  const firstYear = toNumber(snapshots[0].year_index, 0);
  const lastYear = toNumber(snapshots[snapshots.length - 1].year_index, firstYear + 1);
  const deltaYears = Math.max(0.25, lastYear - firstYear);
  return round(((last.motor + last.cognitive + last.nonMotor) - (first.motor + first.cognitive + first.nonMotor)) / 3 / deltaYears, 2);
}

function forecastFromSnapshot(snapshot, state) {
  const currentUpdrs = toNumber(snapshot.motor.updrs3_score, 20);
  const currentMoca = toNumber(snapshot.cognition.moca, 27);
  const velocity = toNumber(state.progression_velocity, 0.08);
  return [6, 12, 24].map((months) => {
    const years = months / 12;
    return {
      horizon_months: months,
      predicted_updrs3: round(currentUpdrs + years * (2.2 + state.motor_burden_index * 4.5 + velocity * 3), 1),
      predicted_moca: round(clamp(currentMoca - years * (0.35 + state.cognitive_burden_index * 1.4), 0, 30), 1),
      risk_level: state.motor_burden_index > 0.55 ? "elevated" : "moderate",
    };
  });
}

function buildTwinFromSnapshots(twinId, patientLabel, snapshots) {
  const latest = snapshots[snapshots.length - 1];
  const prediction = demoPrediction(latest.raw_inputs);
  const burden = burdenFromSnapshot(latest);
  const velocity = progressionVelocity(snapshots);
  const state = {
    current_cohort_estimate: prediction.prediction,
    prediction_source: "static demo scoring",
    confidence: prediction.confidence,
    motor_burden_index: burden.motor,
    cognitive_burden_index: burden.cognitive,
    non_motor_burden_index: burden.nonMotor,
    progression_velocity: velocity,
    treatment_effect: round(Math.max(0, toNumber(latest.motor.updrs3_score) - toNumber(latest.motor.updrs3_score_on)), 1),
    cluster_label: burden.motor > 0.55 ? "fast" : "moderate",
    evidence: [
      "Static demo twin uses deterministic frontend scoring when the Flask API is not attached.",
      "Motor burden combines tremor, rigidity, bradykinesia, postural instability, and UPDRS III values.",
      "Connect VITE_API_BASE_URL to use persisted Flask digital twins and model-backed forecasting.",
    ],
  };

  return {
    profile: {
      twin_id: twinId,
      patient_label: patientLabel,
    },
    twin_id: twinId,
    patient_label: patientLabel,
    current_state: state,
    snapshots,
    forecast: forecastFromSnapshot(latest, state),
    summary: {
      snapshot_count: snapshots.length,
    },
    confidence: prediction.confidence,
    current_cohort_estimate: prediction.prediction,
    snapshot_count: snapshots.length,
    updated_at: new Date().toISOString(),
  };
}

function seedDemoTwin() {
  const baseline = snapshotFromPatient(
    {
      patient_id: "Interview Demo",
      age: 63,
      SEX: 1,
      EDUCYRS: 16,
      BMI: 24.8,
      sym_tremor: 2,
      sym_rigid: 2,
      sym_brady: 3,
      sym_posins: 1,
      moca: 26,
      rem: 1,
      ess: 9,
      gds: 5,
      stai: 42,
      LEDD: 500,
      updrs3_score: 24,
      updrs3_score_on: 17,
      YEAR: 0,
      visit_date: "2026-01-14",
    },
    0,
  );
  const followUp = snapshotFromPatient(
    {
      ...baseline.raw_inputs,
      EVENT_ID: "V02",
      visit_date: "2026-07-14",
      YEAR: 0.5,
      age: 63.5,
      sym_tremor: 3,
      sym_brady: 3,
      moca: 25,
      updrs3_score: 28,
      updrs3_score_on: 20,
    },
    1,
  );
  return buildTwinFromSnapshots("demo_interview_twin", "Interview Demo", [baseline, followUp]);
}

function getStoredTwins() {
  return readJsonStorage(DEMO_TWINS_KEY, []);
}

function getDemoTwins() {
  const stored = getStoredTwins();
  const seed = seedDemoTwin();
  return [seed, ...stored.filter((twin) => twin.twin_id !== seed.twin_id)];
}

function saveDemoTwin(twin) {
  const stored = getStoredTwins().filter((item) => item.twin_id !== twin.twin_id);
  writeJsonStorage(DEMO_TWINS_KEY, [twin, ...stored].slice(0, 8));
}

function findDemoTwin(twinId) {
  return getDemoTwins().find((twin) => twin.twin_id === twinId) || getDemoTwins()[0];
}

export async function predict(patientData) {
  return withDemoFallback(async () => {
    const res = await request("/api/predict", {
      method: "POST",
      body: JSON.stringify(patientData),
    });
    return res.json();
  }, () => demoPrediction(patientData));
}

export async function validateData(patientData) {
  return withDemoFallback(async () => {
    const res = await request("/api/validate_data", {
      method: "POST",
      body: JSON.stringify(patientData),
    });
    return res.json();
  }, () => ({
    valid: true,
    errors: [],
    warnings: patientData?.age > 75 ? ["Advanced age may affect assessment accuracy"] : [],
    demo_mode: true,
  }));
}

export async function generateReport(patientData, patientId) {
  return withDemoFallback(async () => {
    const res = await request("/api/generate_report", {
      method: "POST",
      body: JSON.stringify({
        patient_data: patientData,
        patient_id: patientId,
      }),
    });
    return res.json();
  }, () => demoReport(patientData, patientId));
}

export async function generatePatientReport(patientData, patientId) {
  return withDemoFallback(async () => {
    const res = await request("/api/generate_patient_report", {
      method: "POST",
      body: JSON.stringify({
        patient_data: patientData,
        patient_id: patientId,
      }),
    });
    return res.json();
  }, () => ({
    report: demoReport(patientData, patientId).report,
    report_type: "patient",
    demo_mode: true,
  }));
}

export async function generateDoctorReport(patientData, patientId) {
  return withDemoFallback(async () => {
    const res = await request("/api/generate_doctor_report", {
      method: "POST",
      body: JSON.stringify({
        patient_data: patientData,
        patient_id: patientId,
      }),
    });
    return res.json();
  }, () => ({
    report: demoReport(patientData, patientId).report,
    report_type: "doctor",
    demo_mode: true,
  }));
}

export async function generateBothReports(patientData, patientId) {
  return withDemoFallback(async () => {
    const res = await request("/api/generate_both_reports", {
      method: "POST",
      body: JSON.stringify({
        patient_data: patientData,
        patient_id: patientId,
      }),
    });
    return res.json();
  }, () => {
    const report = demoReport(patientData, patientId).report;
    return {
      patient_report: report,
      doctor_report: report,
      demo_mode: true,
    };
  });
}

export async function generateReportPdf(
  patientData,
  patientId,
  predictionResults,
  reportText = "",
) {
  if (STATIC_DEMO_MODE) {
    return buildDemoPdfBlob(reportText || demoReport(patientData, patientId).report, {
      patientData,
      patientId,
      predictionResults,
    });
  }

  try {
    const res = await fetch(buildUrl("/api/generate_report_pdf"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_data: patientData,
        patient_id: patientId,
        prediction_results: predictionResults,
        report_text: reportText,
      }),
    });

    if (!res.ok) {
      throw new Error(await parseErrorResponse(res));
    }

    if (!isJsonResponse(res) && (res.headers.get("Content-Type") || "").includes("application/pdf")) {
      return res.blob();
    }

    if (DEMO_MODE && !isJsonResponse(res)) {
      throw new Error(BACKEND_UNAVAILABLE_MESSAGE);
    }

    return res.blob();
  } catch (err) {
    if (DEMO_MODE && isDemoFallbackError(err)) {
      return buildDemoPdfBlob(reportText || demoReport(patientData, patientId).report, {
        patientData,
        patientId,
        predictionResults,
      });
    }
    throw err;
  }
}

export async function getSystemStatus() {
  return withDemoFallback(async () => {
    const res = await request("/api/system_status");
    return res.json();
  }, () => ({
    port_ready: true,
    system_initialized: true,
    models_loaded: true,
    demo_mode: true,
    status: "static-demo",
  }));
}

export async function getDocuments() {
  return withDemoFallback(async () => {
    const res = await request("/api/documents");
    return res.json();
  }, () => {
    const documents = getDemoDocuments();
    return {
      documents,
      counts: documentCounts(documents),
      demo_mode: true,
    };
  });
}

export async function getDocument(docId) {
  return withDemoFallback(async () => {
    const res = await request(`/api/documents/${docId}`);
    return res.json();
  }, () => ({
    document: getDemoDocuments().find((doc) => doc.id === docId) || demoDocuments[0],
    demo_mode: true,
  }));
}

export async function uploadDocument(formData) {
  return withDemoFallback(async () => {
    const res = await request("/api/upload_document", {
      method: "POST",
      body: formData,
    });
    return res.json();
  }, () => {
    const file = formData.get("document");
    const title = formData.get("title") || file?.name || "Uploaded document";
    const document = {
      id: `demo-upload-${Date.now()}`,
      title,
      name: title,
      type: formData.get("doc_type") || "paper",
      metadata: {
        title,
        authors: formData.get("author") || "Demo upload",
        source: "Static demo session",
        year: String(new Date().getFullYear()),
        size_bytes: file?.size || 0,
      },
      preview: "Uploaded into the browser demo session. Connect the Flask API to index full content.",
      content: "This document was added in static demo mode and is stored in this browser only.",
    };
    const stored = readJsonStorage(DEMO_DOCS_KEY, []);
    writeJsonStorage(DEMO_DOCS_KEY, [...stored, document]);
    const documents = getDemoDocuments();
    return {
      document,
      counts: documentCounts(documents),
      demo_mode: true,
    };
  });
}

export async function deleteDocument(docId) {
  return withDemoFallback(async () => {
    const res = await request(`/api/delete_document/${docId}`, {
      method: "DELETE",
    });
    return res.json();
  }, () => {
    const stored = readJsonStorage(DEMO_DOCS_KEY, []).filter((doc) => doc.id !== docId);
    writeJsonStorage(DEMO_DOCS_KEY, stored);
    return {
      removed: true,
      counts: documentCounts(getDemoDocuments()),
      demo_mode: true,
    };
  });
}

export function getReportDownloadUrl(filename) {
  return buildUrl(`/api/download_report/${filename}`);
}

export async function getModelMetricsSummary() {
  return withDemoFallback(async () => {
    const res = await request("/api/model_metrics_summary");
    return res.json();
  }, () => ({
    best_traditional: { name: "LightGBM", accuracy_pct: 92.4 },
    best_transformer: { name: "PubMedBERT", accuracy_pct: 89.7 },
    models: [
      { name: "LightGBM", accuracy_pct: 92.4 },
      { name: "XGBoost", accuracy_pct: 91.1 },
      { name: "SVM", accuracy_pct: 87.8 },
      { name: "PubMedBERT", accuracy_pct: 89.7 },
      { name: "BioGPT", accuracy_pct: 86.2 },
      { name: "Clinical-T5", accuracy_pct: 88.5 },
    ],
    demo_mode: true,
  }));
}

export async function listTwins() {
  return withDemoFallback(async () => {
    const res = await request("/api/twins");
    return res.json();
  }, () => ({
    twins: getDemoTwins().map((twin) => ({
      twin_id: twin.twin_id,
      patient_label: twin.patient_label,
      confidence: twin.confidence,
      current_cohort_estimate: twin.current_cohort_estimate,
      snapshot_count: twin.snapshot_count,
      updated_at: twin.updated_at,
    })),
    demo_mode: true,
  }));
}

export async function createTwin(patientData, patientId = null, sourcePatno = null) {
  return withDemoFallback(async () => {
    const res = await request("/api/twins", {
      method: "POST",
      body: JSON.stringify({
        patient_data: patientData,
        patient_id: patientId,
        source_patno: sourcePatno,
      }),
    });
    return res.json();
  }, () => {
    const twinId = `demo_${Date.now()}`;
    const label = patientId || patientData?.patient_id || "Demo Patient";
    const twin = buildTwinFromSnapshots(twinId, label, [snapshotFromPatient(patientData, 0)]);
    saveDemoTwin(twin);
    return {
      message: "Static demo twin created",
      twin_id: twin.twin_id,
      twin,
      demo_mode: true,
    };
  });
}

export async function getTwin(twinId) {
  return withDemoFallback(async () => {
    const res = await request(`/api/twins/${twinId}`);
    return res.json();
  }, () => ({
    twin: findDemoTwin(twinId),
    demo_mode: true,
  }));
}

export async function addTwinSnapshot(twinId, patientData) {
  return withDemoFallback(async () => {
    const res = await request(`/api/twins/${twinId}/snapshot`, {
      method: "POST",
      body: JSON.stringify({
        patient_data: patientData,
      }),
    });
    return res.json();
  }, () => {
    const current = findDemoTwin(twinId);
    const snapshots = [
      ...current.snapshots,
      snapshotFromPatient(patientData, current.snapshots.length),
    ];
    const twin = buildTwinFromSnapshots(current.twin_id, current.patient_label, snapshots);
    if (current.twin_id !== "demo_interview_twin") {
      saveDemoTwin(twin);
    }
    return {
      message: "Static demo snapshot added",
      twin,
      demo_mode: true,
    };
  });
}

export async function simulateTwin(twinId, overrides, scenarioName = "") {
  return withDemoFallback(async () => {
    const res = await request(`/api/twins/${twinId}/simulate`, {
      method: "POST",
      body: JSON.stringify({
        overrides,
        scenario_name: scenarioName,
      }),
    });
    return res.json();
  }, () => {
    const twin = findDemoTwin(twinId);
    const base = twin.snapshots[twin.snapshots.length - 1];
    const simulatedSnapshot = snapshotFromPatient(
      {
        ...base.raw_inputs,
        ...overrides,
        EVENT_ID: "SCENARIO",
        visit_date: todayIso(),
      },
      twin.snapshots.length,
    );
    const simulatedTwin = buildTwinFromSnapshots(
      twin.twin_id,
      twin.patient_label,
      [...twin.snapshots, simulatedSnapshot],
    );
    return {
      simulation: {
        scenario_name: scenarioName || "Static demo scenario",
        simulated_snapshot: simulatedSnapshot,
        state: simulatedTwin.current_state,
        forecast: simulatedTwin.forecast,
      },
      demo_mode: true,
    };
  });
}

export async function getTwinTrajectory(twinId) {
  return withDemoFallback(async () => {
    const res = await request(`/api/twins/${twinId}/trajectory`);
    return res.json();
  }, () => ({
    forecast: findDemoTwin(twinId).forecast,
    demo_mode: true,
  }));
}

export { API_BASE };
