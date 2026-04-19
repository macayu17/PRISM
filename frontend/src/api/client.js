const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function buildUrl(path) {
  return `${API_BASE}${path}`;
}

async function parseErrorResponse(res) {
  try {
    const data = await res.json();
    return data?.error || data?.message || res.statusText || "Request failed";
  } catch {
    return res.statusText || "Request failed";
  }
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

  return res;
}

export async function predict(patientData) {
  const res = await request("/api/predict", {
    method: "POST",
    body: JSON.stringify(patientData),
  });
  return res.json();
}

export async function validateData(patientData) {
  const res = await request("/api/validate_data", {
    method: "POST",
    body: JSON.stringify(patientData),
  });
  return res.json();
}

export async function generateReport(patientData, patientId) {
  const res = await request("/api/generate_report", {
    method: "POST",
    body: JSON.stringify({
      patient_data: patientData,
      patient_id: patientId,
    }),
  });
  return res.json();
}

export async function generatePatientReport(patientData, patientId) {
  const res = await request("/api/generate_patient_report", {
    method: "POST",
    body: JSON.stringify({
      patient_data: patientData,
      patient_id: patientId,
    }),
  });
  return res.json();
}

export async function generateDoctorReport(patientData, patientId) {
  const res = await request("/api/generate_doctor_report", {
    method: "POST",
    body: JSON.stringify({
      patient_data: patientData,
      patient_id: patientId,
    }),
  });
  return res.json();
}

export async function generateBothReports(patientData, patientId) {
  const res = await request("/api/generate_both_reports", {
    method: "POST",
    body: JSON.stringify({
      patient_data: patientData,
      patient_id: patientId,
    }),
  });
  return res.json();
}

export async function generateReportPdf(
  patientData,
  patientId,
  predictionResults,
  reportText = "",
) {
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

  return res.blob();
}

export async function getSystemStatus() {
  const res = await request("/api/system_status");
  return res.json();
}

export async function getDocuments() {
  const res = await request("/api/documents");
  return res.json();
}

export async function getDocument(docId) {
  const res = await request(`/api/documents/${docId}`);
  return res.json();
}

export async function uploadDocument(formData) {
  const res = await request("/api/upload_document", {
    method: "POST",
    body: formData,
  });
  return res.json();
}

export async function deleteDocument(docId) {
  const res = await request(`/api/delete_document/${docId}`, {
    method: "DELETE",
  });
  return res.json();
}

export function getReportDownloadUrl(filename) {
  return buildUrl(`/api/download_report/${filename}`);
}

export async function getModelMetricsSummary() {
  const res = await request("/api/model_metrics_summary");
  return res.json();
}

export async function listTwins() {
  const res = await request("/api/twins");
  return res.json();
}

export async function createTwin(patientData, patientId = null, sourcePatno = null) {
  const res = await request("/api/twins", {
    method: "POST",
    body: JSON.stringify({
      patient_data: patientData,
      patient_id: patientId,
      source_patno: sourcePatno,
    }),
  });
  return res.json();
}

export async function getTwin(twinId) {
  const res = await request(`/api/twins/${twinId}`);
  return res.json();
}

export async function addTwinSnapshot(twinId, patientData) {
  const res = await request(`/api/twins/${twinId}/snapshot`, {
    method: "POST",
    body: JSON.stringify({
      patient_data: patientData,
    }),
  });
  return res.json();
}

export async function simulateTwin(twinId, overrides, scenarioName = "") {
  const res = await request(`/api/twins/${twinId}/simulate`, {
    method: "POST",
    body: JSON.stringify({
      overrides,
      scenario_name: scenarioName,
    }),
  });
  return res.json();
}

export async function getTwinTrajectory(twinId) {
  const res = await request(`/api/twins/${twinId}/trajectory`);
  return res.json();
}

export { API_BASE };
