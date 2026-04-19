import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  FileText,
  Upload,
  Trash2,
  Eye,
  AlertCircle,
  Loader,
  CheckCircle,
  RefreshCw,
  Files,
} from 'lucide-react';
import { getDocuments, getDocument, uploadDocument, deleteDocument } from '../api/client';
import {
  alertClass,
  badgeClass,
  buttonDanger,
  buttonSecondary,
  glassPanel,
  innerPanel,
  pageShellWide,
  sectionTitle,
} from '../lib/ui';

function formatFileSizeFromDocument(doc) {
  const rawSize =
    doc?.size_bytes ?? doc?.size ?? doc?.metadata?.size_bytes ?? null;

  const size = Number(rawSize);

  if (!Number.isFinite(size) || size <= 0) {
    return 'Unknown size';
  }

  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function getDocumentTitle(doc) {
  return (
    doc?.metadata?.title ||
    doc?.title ||
    doc?.name ||
    doc?.id ||
    'Untitled document'
  );
}

function getDocumentType(doc) {
  return doc?.type || doc?.metadata?.type || 'document';
}

function getDocumentDescription(doc) {
  const author = doc?.metadata?.authors || doc?.author;
  const source = doc?.metadata?.source;
  const year = doc?.metadata?.year;

  const parts = [author, source, year].filter(Boolean);
  return parts.length > 0 ? parts.join(' • ') : 'No metadata available';
}

function getDocumentPreview(doc) {
  if (doc?.preview) return doc.preview;
  const content = doc?.content || '';
  if (!content) return 'No preview available.';
  return content.length > 180 ? `${content.slice(0, 180)}...` : content;
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState([]);
  const [counts, setCounts] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documentDetailLoading, setDocumentDetailLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [selectedDoc, setSelectedDoc] = useState(null);

  const stats = useMemo(() => {
    return {
      total: counts?.total ?? docs.length,
      paper:
        counts?.paper ??
        docs.filter((d) => getDocumentType(d) === 'paper').length,
      guideline:
        counts?.guideline ??
        docs.filter((d) => getDocumentType(d) === 'guideline').length,
      textbook:
        counts?.textbook ??
        docs.filter((d) => getDocumentType(d) === 'textbook').length,
    };
  }, [counts, docs]);

  const loadDocs = async ({ silent = false } = {}) => {
    try {
      if (silent) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError('');
      const data = await getDocuments();
      setDocs(Array.isArray(data?.documents) ? data.documents : []);
      setCounts(data?.counts || null);
    } catch (err) {
      setError(err.message || 'Failed to load documents');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadDocs();
  }, []);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError('');
    setSuccess('');

    try {
      const formData = new FormData();
      formData.append('document', file);
      formData.append('title', file.name.replace(/\.[^/.]+$/, ''));
      formData.append('author', 'Unknown');
      formData.append('doc_type', 'paper');

      const data = await uploadDocument(formData);

      setSuccess(`Document "${file.name}" uploaded successfully.`);

      if (data?.document) {
        setDocs((prev) => [...prev, data.document]);
        if (data?.counts) {
          setCounts(data.counts);
        } else {
          await loadDocs({ silent: true });
        }
      } else {
        await loadDocs({ silent: true });
      }
    } catch (err) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleDelete = async (doc) => {
    const docTitle = getDocumentTitle(doc);
    const confirmed = window.confirm(
      `Delete "${docTitle}" from the knowledge base?`,
    );

    if (!confirmed) return;

    setError('');
    setSuccess('');

    try {
      await deleteDocument(doc.id);
      setDocs((prev) => prev.filter((d) => d.id !== doc.id));
      setSelectedDoc((prev) => (prev?.id === doc.id ? null : prev));
      setSuccess(`Document "${docTitle}" deleted.`);
      await loadDocs({ silent: true });
    } catch (err) {
      setError(err.message || 'Delete failed');
    }
  };

  const handleViewDocument = async (doc) => {
    setSelectedDoc(doc);
    setDocumentDetailLoading(true);
    setError('');

    try {
      const data = await getDocument(doc.id);
      if (data?.document) {
        setSelectedDoc(data.document);
      }
    } catch (err) {
      setError(err.message || 'Failed to load document preview');
    } finally {
      setDocumentDetailLoading(false);
    }
  };

  return (
    <div className={pageShellWide}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="mb-8 text-center">
          <span className={sectionTitle}>Knowledge Base</span>
          <h2>Medical Documents</h2>
          <p className="text-base text-slate-400">
            Manage the indexed literature used to support report generation and
            clinical context retrieval.
          </p>
        </div>

        {error && (
          <div className={alertClass('danger')}>
            <AlertCircle size={16} className="mt-0.5 shrink-0" /> {error}
          </div>
        )}

        {success && (
          <div className={alertClass('success')}>
            <CheckCircle size={16} className="mt-0.5 shrink-0" /> {success}
          </div>
        )}

        <div className={`${glassPanel} mb-6 grid gap-4 bg-black/25 sm:grid-cols-2 xl:grid-cols-4`}>
          {[
            {
              label: 'Total Documents',
              value: stats.total,
              icon: <Files size={18} />,
            },
            {
              label: 'Papers',
              value: stats.paper,
              icon: <FileText size={18} />,
            },
            {
              label: 'Guidelines',
              value: stats.guideline,
              icon: <FileText size={18} />,
            },
            {
              label: 'Textbooks',
              value: stats.textbook,
              icon: <FileText size={18} />,
            },
          ].map((item) => (
            <div key={item.label} className={`${innerPanel} bg-white/[0.02]`}>
              <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
                {item.icon}
                <span>{item.label}</span>
              </div>
              <div className="text-3xl font-semibold tracking-[-0.03em] text-white">
                {item.value}
              </div>
            </div>
          ))}
        </div>

        <div className={`${glassPanel} mb-6 border border-dashed border-white/15 bg-black/25 p-0`}>
          <label
            htmlFor="doc-upload"
            className="flex cursor-pointer flex-col items-center justify-center gap-4 rounded-[1.5rem] px-6 py-16 text-center text-slate-400 transition hover:bg-white/[0.03] hover:text-white"
          >
            <Upload size={32} />
            <span className="text-lg font-semibold text-white">
              {uploading
                ? 'Uploading document...'
                : 'Click to upload a medical document'}
            </span>
            <span className="text-sm text-slate-500">
              PDF or TXT recommended • The backend will index the content for
              retrieval
            </span>
            <input
              id="doc-upload"
              type="file"
              accept=".pdf,.txt"
              onChange={handleUpload}
              disabled={uploading}
              className="hidden"
            />
          </label>
        </div>

        <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
          <h4 className="flex items-center gap-2 text-lg">
            <FileText size={18} />
            Indexed Documents
          </h4>

          <button
            className={buttonSecondary}
            onClick={() => loadDocs({ silent: true })}
            disabled={refreshing || loading}
            type="button"
          >
            {refreshing ? (
              <>
                <Loader size={16} className="animate-spin" /> Refreshing
              </>
            ) : (
              <>
                <RefreshCw size={16} /> Refresh
              </>
            )}
          </button>
        </div>

        {loading ? (
          <div className={`${glassPanel} bg-black/25 px-6 py-12 text-center`}>
            <Loader size={24} className="mx-auto animate-spin" />
            <p className="mt-4 text-base text-slate-400">Loading indexed documents...</p>
          </div>
        ) : docs.length === 0 ? (
          <div className={`${glassPanel} bg-black/25 px-6 py-12 text-center text-slate-500`}>
            <FileText size={48} className="mx-auto mb-4 opacity-30" />
            <p className="text-base">
              No documents are currently indexed.
              <br />
              Upload literature to strengthen the medical knowledge base.
            </p>
          </div>
        ) : (
          <div className={`grid gap-6 ${selectedDoc ? 'xl:grid-cols-[1.2fr_1fr]' : 'grid-cols-1'}`}>
            <div className={`${glassPanel} bg-black/25`}>
              <div className="space-y-3">
                {docs.map((doc) => {
                  const title = getDocumentTitle(doc);
                  const preview = getDocumentPreview(doc);
                  const type = getDocumentType(doc);

                  return (
                    <div
                      key={doc.id}
                      className="flex flex-col gap-4 rounded-[1.25rem] border border-white/10 bg-black/30 p-5 transition hover:border-white/20 hover:bg-black/40 lg:flex-row lg:items-start lg:justify-between"
                    >
                      <div className="flex min-w-0 flex-1 items-start gap-4">
                        <FileText size={18} className="mt-1 shrink-0 text-slate-400" />
                        <div className="min-w-0">
                          <div className="mb-2 flex flex-wrap items-center gap-2">
                            <span className="truncate text-base font-medium text-white">{title}</span>
                            <span className={badgeClass('info')}>{type}</span>
                          </div>

                          <div className="mb-2 text-xs text-slate-500">
                            {formatFileSizeFromDocument(doc)} • {getDocumentDescription(doc)}
                          </div>

                          <p className="m-0 text-sm leading-6 text-slate-400">{preview}</p>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-3">
                        <button
                          className={`${buttonSecondary} px-4 py-2 text-xs`}
                          type="button"
                          onClick={() => handleViewDocument(doc)}
                        >
                          <Eye size={14} /> View
                        </button>
                        <button
                          className={buttonDanger}
                          type="button"
                          onClick={() => handleDelete(doc)}
                        >
                          <Trash2 size={14} /> Delete
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {selectedDoc && (
              <div className={`${glassPanel} bg-black/25`}>
                <div className="mb-4 flex items-center justify-between gap-4">
                  <h4 className="text-lg">Document Preview</h4>
                  <button
                    className={buttonSecondary}
                    type="button"
                    onClick={() => setSelectedDoc(null)}
                  >
                    Close
                  </button>
                </div>

                <div className="mb-3 text-lg font-semibold text-white">
                  {getDocumentTitle(selectedDoc)}
                </div>

                <div className="mb-4 text-sm text-slate-400">
                  {getDocumentType(selectedDoc)} • {formatFileSizeFromDocument(selectedDoc)}
                </div>

                <div className="max-h-[420px] overflow-auto rounded-[1.25rem] border border-white/10 bg-white/[0.03] p-4 text-sm leading-7 whitespace-pre-wrap text-slate-300">
                  {documentDetailLoading
                    ? 'Loading full document...'
                    : selectedDoc.content || selectedDoc.preview || 'No content preview available.'}
                </div>
              </div>
            )}
          </div>
        )}
      </motion.div>
    </div>
  );
}
