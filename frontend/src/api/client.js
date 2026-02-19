const API_BASE = 'http://localhost:5000';

async function request(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error || 'Request failed');
    }
    return res;
}

export async function predict(patientData) {
    const res = await request('/api/predict', {
        method: 'POST',
        body: JSON.stringify(patientData),
    });
    return res.json();
}

export async function validateData(patientData) {
    const res = await request('/api/validate_data', {
        method: 'POST',
        body: JSON.stringify(patientData),
    });
    return res.json();
}

export async function generateReport(patientData, patientId) {
    const res = await request('/api/generate_report', {
        method: 'POST',
        body: JSON.stringify({ patient_data: patientData, patient_id: patientId }),
    });
    return res.json();
}

export async function generateReportPdf(patientData, patientId, predictionResults) {
    const res = await fetch(`${API_BASE}/api/generate_report_pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            patient_data: patientData,
            patient_id: patientId,
            prediction_results: predictionResults
        }),
    });
    if (!res.ok) throw new Error('PDF generation failed');
    return res.blob();
}

export async function getSystemStatus() {
    const res = await request('/api/system_status');
    return res.json();
}

export async function getDocuments() {
    const res = await request('/documents');
    // Since Flask returns HTML for /documents, we'll use system_status for now
    return res.json();
}

export async function uploadDocument(formData) {
    const res = await fetch(`${API_BASE}/api/upload_document`, {
        method: 'POST',
        body: formData, // multipart/form-data, no Content-Type header
    });
    if (!res.ok) throw new Error('Upload failed');
    return res.json();
}

export async function deleteDocument(docId) {
    const res = await request(`/api/delete_document/${docId}`, { method: 'DELETE' });
    return res.json();
}

export function getReportDownloadUrl(filename) {
    return `${API_BASE}/api/download_report/${filename}`;
}
