import { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Upload, Trash2, Eye, AlertCircle, Loader, CheckCircle } from 'lucide-react';
import './DocumentsPage.css';

export default function DocumentsPage() {
    const [docs, setDocs] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setUploading(true);
        setError('');
        setSuccess('');

        try {
            const formData = new FormData();
            formData.append('document', file);

            const res = await fetch('http://localhost:5000/api/upload_document', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error('Upload failed');
            const data = await res.json();
            setSuccess(`Document "${file.name}" uploaded successfully!`);
            setDocs(prev => [...prev, { id: data.doc_id || Date.now(), name: file.name, size: file.size }]);
        } catch (err) {
            setError(err.message);
        } finally {
            setUploading(false);
            e.target.value = '';
        }
    };

    const handleDelete = async (doc) => {
        try {
            await fetch(`http://localhost:5000/api/delete_document/${doc.id}`, { method: 'DELETE' });
            setDocs(prev => prev.filter(d => d.id !== doc.id));
            setSuccess(`Document "${doc.name}" deleted.`);
        } catch (err) {
            setError(err.message);
        }
    };

    return (
        <div className="container" style={{ maxWidth: 800 }}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                    <span className="section-title">Knowledge Base</span>
                    <h2>Medical Documents</h2>
                    <p>Upload medical literature to enhance RAG-based report generation</p>
                </div>

                {error && <div className="alert alert-danger"><AlertCircle size={16} /> {error}</div>}
                {success && <div className="alert alert-success"><CheckCircle size={16} /> {success}</div>}

                {/* Upload Area */}
                <div className="glass-card-static upload-area">
                    <label htmlFor="doc-upload" className="upload-label">
                        <Upload size={32} />
                        <span className="upload-text">
                            {uploading ? 'Uploading...' : 'Click to upload a document'}
                        </span>
                        <span className="upload-hint">PDF, TXT, or DOCX • Max 10MB</span>
                        <input
                            id="doc-upload"
                            type="file"
                            accept=".pdf,.txt,.docx"
                            onChange={handleUpload}
                            disabled={uploading}
                            style={{ display: 'none' }}
                        />
                    </label>
                </div>

                {/* Document List */}
                {docs.length > 0 && (
                    <div className="glass-card-static" style={{ marginTop: '1.5rem' }}>
                        <h4 style={{ marginBottom: '1rem' }}>
                            <FileText size={18} style={{ marginRight: '0.5rem' }} />
                            Uploaded Documents ({docs.length})
                        </h4>
                        <div className="doc-list">
                            {docs.map((doc) => (
                                <div key={doc.id} className="doc-item">
                                    <div className="doc-info">
                                        <FileText size={16} />
                                        <span className="doc-name">{doc.name}</span>
                                        <span className="doc-size">{(doc.size / 1024).toFixed(1)} KB</span>
                                    </div>
                                    <div className="doc-actions">
                                        <button className="btn-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
                                            <Eye size={14} /> View
                                        </button>
                                        <button className="btn-danger" onClick={() => handleDelete(doc)}>
                                            <Trash2 size={14} /> Delete
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {docs.length === 0 && (
                    <div style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--text-muted)' }}>
                        <FileText size={48} style={{ marginBottom: '1rem', opacity: 0.3 }} />
                        <p>No documents uploaded yet.<br />Upload medical literature to enhance the RAG system.</p>
                    </div>
                )}
            </motion.div>
        </div>
    );
}
