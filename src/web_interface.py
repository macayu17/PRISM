"""
Web Interface for Parkinson's Disease Assessment System.
Flask-based web application for patient data input and automated report generation.
"""

import io
import csv
import html
import math
import os
import re
import secrets
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, flash, redirect, url_for
from flask_cors import CORS

DEBUG_LOGS = os.getenv('PD_DEBUG_LOGS', '0') == '1'

def dlog(*args, **kwargs):
    if DEBUG_LOGS:
        print(*args, **kwargs)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from rag_system import ReportGenerator, MedicalKnowledgeBase
from document_manager import DocumentManager
from dual_report_generator import DualReportManager
from twin_engine import DigitalTwinEngine

# Set template and static folders to the directories in the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
template_dir = os.path.join(project_root, 'templates')
static_dir = os.path.join(project_root, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
default_allowed_origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
allowed_origins = [
    origin.strip()
    for origin in os.getenv("PD_ALLOWED_ORIGINS", ",".join(default_allowed_origins)).split(",")
    if origin.strip()
]
CORS(
    app,
    resources={r"/api/*": {"origins": "*" if allowed_origins == ["*"] else allowed_origins}},
)
app.secret_key = os.getenv("PD_SECRET_KEY") or secrets.token_hex(32)

# Initialize global components
report_generator: Optional[ReportGenerator] = None
dual_report_manager = DualReportManager()
knowledge_base = MedicalKnowledgeBase()

# Get the correct path for medical_docs - check both src and root
current_dir = os.path.dirname(os.path.abspath(__file__))
medical_docs_path = os.path.join(os.path.dirname(current_dir), "medical_docs")
if not os.path.exists(medical_docs_path):
    medical_docs_path = os.path.join(current_dir, "medical_docs")
    if not os.path.exists(medical_docs_path):
        # Create the directory if it doesn't exist
        os.makedirs(medical_docs_path, exist_ok=True)

document_manager = DocumentManager(medical_docs_path)
digital_twin_engine = DigitalTwinEngine()

ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".txt"}
MODEL_REQUIRED_FIELDS = [
    "age",
    "SEX",
    "EDUCYRS",
    "BMI",
    "sym_tremor",
    "sym_rigid",
    "sym_brady",
    "sym_posins",
]
RACE_MAPPING = {
    "white": 1.0,
    "black": 2.0,
    "black/african american": 2.0,
    "african american": 2.0,
    "asian": 3.0,
    "other": 4.0,
}
FAMPD_LABEL_MAPPING = {
    "no family history": 3.0,
    "first degree relative": 1.0,
    "other relative": 2.0,
}


def _get_report_generator() -> Optional[ReportGenerator]:
    return report_generator


def _ensure_system_initialized() -> Optional[ReportGenerator]:
    global report_generator
    if report_generator is None:
        if not initialize_system():
            return None
    return report_generator


def _get_twin_predictor() -> Optional[ReportGenerator]:
    try:
        return _ensure_system_initialized()
    except Exception:
        return None


def _get_json_payload() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _safe_filename(filename: Optional[str]) -> str:
    if not filename:
        return f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    base = Path(filename).name.strip()
    sanitized = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in base).strip()
    return sanitized or f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def _safe_report_token(value: Optional[Any], fallback: str = "report") -> str:
    token = Path(str(value or fallback)).stem.strip()
    token = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in token).strip(" .")
    return token or fallback


def _build_report_filename(prefix: str, patient_id: Optional[Any], extension: str = ".txt") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    identifier = _safe_report_token(patient_id, fallback=timestamp)
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{prefix}_{identifier}{ext}"


def _reports_dir() -> Path:
    path = Path(project_root) / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_safe_report_filename(filename: str, expected_suffix: str = ".txt") -> bool:
    if not filename or filename != Path(filename).name:
        return False
    if ".." in filename or filename.startswith("."):
        return False
    sanitized = _safe_filename(filename)
    return sanitized == filename and filename.lower().endswith(expected_suffix.lower())


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _normalize_patient_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    numeric_fields = {
        "age",
        "SEX",
        "EDUCYRS",
        "race",
        "BMI",
        "fampd",
        "fampd_bin",
        "sym_tremor",
        "sym_rigid",
        "sym_brady",
        "sym_posins",
        "rem",
        "ess",
        "gds",
        "stai",
        "moca",
        "clockdraw",
        "bjlot",
    }

    for key, value in payload.items():
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                continue
            value = stripped
        normalized[key] = value

    for field in numeric_fields:
        if field in normalized and field not in {"SEX", "race", "fampd"}:
            coerced = _coerce_float(normalized[field])
            if coerced is not None:
                normalized[field] = coerced

    sex_value = normalized.get("SEX")
    if isinstance(sex_value, str):
        lower = sex_value.lower()
        if lower in {"male", "m"}:
            normalized["SEX"] = 1.0
        elif lower in {"female", "f"}:
            normalized["SEX"] = 0.0
        else:
            coerced = _coerce_float(sex_value)
            if coerced is not None:
                normalized["SEX"] = coerced
    elif sex_value is not None:
        coerced = _coerce_float(sex_value)
        if coerced is not None:
            normalized["SEX"] = coerced

    race_value = normalized.get("race")
    if isinstance(race_value, str):
        lower = race_value.lower()
        if lower in RACE_MAPPING:
            normalized["race"] = RACE_MAPPING[lower]
        else:
            coerced = _coerce_float(race_value)
            if coerced is not None:
                normalized["race"] = coerced

    fampd_value = normalized.get("fampd")
    fampd_code: Optional[float] = None
    if isinstance(fampd_value, str):
        lower = fampd_value.lower()
        if lower in FAMPD_LABEL_MAPPING:
            fampd_code = FAMPD_LABEL_MAPPING[lower]
        else:
            coerced = _coerce_float(fampd_value)
            if coerced is not None:
                fampd_code = coerced
    elif fampd_value is not None:
        fampd_code = _coerce_float(fampd_value)

    if fampd_code is not None:
        if fampd_code == 0:
            fampd_code = 3.0
        elif fampd_code not in {1.0, 2.0, 3.0}:
            fampd_code = None

    if fampd_code is not None:
        normalized["fampd"] = fampd_code

    fampd_bin_value = normalized.get("fampd_bin")
    fampd_bin_code: Optional[float] = None
    if isinstance(fampd_bin_value, str):
        fampd_bin_code = _coerce_float(fampd_bin_value)
    elif fampd_bin_value is not None:
        fampd_bin_code = _coerce_float(fampd_bin_value)

    if fampd_bin_code is not None:
        if fampd_bin_code == 0:
            fampd_bin_code = 2.0
        elif fampd_bin_code == 1:
            fampd_bin_code = 1.0
        elif fampd_bin_code == 2:
            fampd_bin_code = 2.0
        else:
            fampd_bin_code = None

    if fampd_bin_code is None and fampd_code is not None:
        fampd_bin_code = 2.0 if fampd_code == 3.0 else 1.0

    if fampd_bin_code is not None:
        normalized["fampd_bin"] = fampd_bin_code

    return normalized


def _missing_required_model_fields(patient_data: Dict[str, Any]) -> list[str]:
    missing = []
    for field in MODEL_REQUIRED_FIELDS:
        value = patient_data.get(field)
        if not isinstance(value, (int, float)) or not _has_value(float(value)):
            missing.append(field)
    return missing


def _load_metrics_summary() -> Dict[str, Any]:
    candidate_paths = [
        Path(project_root) / "evaluation_results" / "summary_metrics.csv",
        Path(project_root) / "evaluation_results" / "model_metrics" / "model_metrics_summary.csv",
    ]
    rows = []
    for path in candidate_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    accuracy = float(row.get("Accuracy", 0) or 0)
                except ValueError:
                    accuracy = 0.0
                rows.append(
                    {
                        "name": row.get("Model", "Unknown"),
                        "type": row.get("Type", "Unknown"),
                        "accuracy": accuracy,
                        "accuracy_pct": round(accuracy * 100, 2),
                    }
                )
        if rows:
            break

    rows.sort(key=lambda item: item["accuracy"], reverse=True)
    best_traditional = next((row for row in rows if "traditional" in row["type"].lower()), None)
    best_transformer = next((row for row in rows if "transformer" in row["type"].lower()), None)

    return {
        "models": rows,
        "best_overall": rows[0] if rows else None,
        "best_traditional": best_traditional,
        "best_transformer": best_transformer,
        "generated_at": datetime.now().isoformat(),
    }


def _json_error(message: str, status_code: int = 400):
    return jsonify({'error': message}), status_code


def _document_extension_allowed(filename: Optional[str]) -> bool:
    suffix = Path(filename or "").suffix.lower()
    return suffix in ALLOWED_DOCUMENT_EXTENSIONS


def initialize_system():
    """Initialize the ML models and report generator."""
    global report_generator
    try:
        # Initialize document manager with medical documents
        doc_count = document_manager.get_document_count()
        print(f"Loaded {doc_count} medical documents")
        
        # Initialize report generator with document manager - use correct path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        docs_dir = os.path.join(os.path.dirname(current_dir), "medical_docs")
        if not os.path.exists(docs_dir):
            docs_dir = os.path.join(current_dir, "medical_docs")
        
        report_generator = ReportGenerator(knowledge_base, docs_dir=docs_dir)
        print("Loading ML models...")
        report_generator.load_models()
        print("System initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing system: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main page with patient assessment form."""
    return render_template('index.html', metrics_summary=_load_metrics_summary())

@app.route('/assessment')
def assessment():
    """Patient assessment form page."""
    return render_template('assessment.html')


@app.route('/twin')
def twin_page():
    """Digital twin page for listing and inspecting saved twins."""
    return render_template('twin.html')

@app.route('/about')
def about():
    """About page with system information."""
    return render_template(
        'about.html',
        knowledge_base=knowledge_base,
        metrics_summary=_load_metrics_summary(),
        generated_month=datetime.now().strftime('%B %Y'),
    )

@app.route('/documents')
def documents():
    """Document management page."""
    docs = document_manager.get_all_documents(include_content=False)
    return render_template('documents.html', documents=docs)

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Handle document upload from the legacy Flask form."""
    try:
        file = request.files.get('document')
        title = (request.form.get('title') or '').strip()
        author = (request.form.get('author') or '').strip()
        doc_type = (request.form.get('doc_type') or 'paper').strip().lower()

        if not file or not file.filename:
            flash('Please provide a document file', 'danger')
            return redirect(url_for('documents'))

        if not _document_extension_allowed(file.filename):
            allowed = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
            flash(f'Unsupported document type. Allowed types: {allowed}', 'danger')
            return redirect(url_for('documents'))

        if not title:
            title = Path(file.filename).stem

        filename = _safe_filename(file.filename)
        temp_path = os.path.join(str(document_manager.main_dir), filename)
        file.save(temp_path)

        doc_id = document_manager.add_document(
            temp_path,
            doc_type=doc_type,
            title=title,
            author=author or None,
        )

        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass

        flash(f'Document uploaded successfully ({doc_id})!', 'success')
    except Exception as e:
        flash(f'Error uploading document: {str(e)}', 'danger')

    return redirect(url_for('documents'))


@app.route('/api/upload_document', methods=['POST'])
def api_upload_document():
    """JSON API for document upload used by the React frontend."""
    try:
        file = request.files.get('document')
        if not file or not file.filename:
            return _json_error('No document file provided')

        if not _document_extension_allowed(file.filename):
            allowed = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
            return _json_error(f'Unsupported document type. Allowed types: {allowed}')

        title = (request.form.get('title') or Path(file.filename).stem).strip()
        author = (request.form.get('author') or '').strip()
        doc_type = (request.form.get('doc_type') or 'paper').strip().lower()

        filename = _safe_filename(file.filename)
        temp_path = os.path.join(str(document_manager.main_dir), filename)
        file.save(temp_path)

        doc_id = document_manager.add_document(
            temp_path,
            doc_type=doc_type,
            title=title,
            author=author or None,
        )

        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass

        return jsonify({
            'message': 'Document uploaded successfully',
            'doc_id': doc_id,
            'document': document_manager.get_document_summary(doc_id),
            'counts': document_manager.get_document_count(),
        })
    except Exception as e:
        print(f"Document upload error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/delete_document/<doc_id>', methods=['POST'])
def delete_document(doc_id):
    """Delete a document from the legacy Flask form."""
    try:
        removed = document_manager.remove_document(doc_id)
        if removed:
            flash('Document deleted successfully!', 'success')
        else:
            flash('Document not found', 'danger')
    except Exception as e:
        flash(f'Error deleting document: {str(e)}', 'danger')

    return redirect(url_for('documents'))


@app.route('/api/delete_document/<doc_id>', methods=['DELETE'])
def api_delete_document(doc_id):
    """JSON API for deleting a document."""
    try:
        removed = document_manager.remove_document(doc_id)
        if not removed:
            return _json_error('Document not found', 404)
        return jsonify({'message': 'Document deleted successfully', 'doc_id': doc_id})
    except Exception as e:
        print(f"Document delete error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/documents', methods=['GET'])
def api_documents():
    """JSON API for listing indexed documents."""
    try:
        return jsonify({
            'documents': document_manager.get_all_documents(include_content=False),
            'counts': document_manager.get_document_count(),
        })
    except Exception as e:
        print(f"Document list error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/documents/<doc_id>', methods=['GET'])
def api_document_detail(doc_id):
    """JSON API for fetching one indexed document with full content."""
    try:
        document = document_manager.get_document(doc_id)
        if document is None:
            return _json_error('Document not found', 404)
        return jsonify({'document': document})
    except Exception as e:
        print(f"Document detail error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/twins', methods=['GET'])
def api_list_twins():
    """List saved digital twins."""
    try:
        return jsonify({'twins': digital_twin_engine.list_twins()})
    except Exception as e:
        print(f"Twin list error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/twins', methods=['POST'])
def api_create_twin():
    """Create a new digital twin from patient assessment data."""
    try:
        data = _get_json_payload()
        patient_data = _normalize_patient_data(cast(Dict[str, Any], data.get('patient_data', data)))
        patient_id = data.get('patient_id') or patient_data.get('patient_id')
        source_patno_raw = data.get('source_patno') or patient_data.get('PATNO')

        if not patient_data:
            return _json_error('No patient data provided')

        missing_fields = _missing_required_model_fields(patient_data)
        if missing_fields:
            return _json_error(f'Missing required fields: {missing_fields}')

        source_patno = _coerce_float(source_patno_raw)
        twin = digital_twin_engine.create_twin(
            patient_data=patient_data,
            patient_label=cast(Optional[str], patient_id),
            source_patno=int(source_patno) if source_patno is not None else None,
            predictor=_get_twin_predictor(),
        )
        return jsonify({
            'message': 'Digital twin created successfully',
            'twin_id': twin['profile']['twin_id'],
            'twin': twin,
        })
    except Exception as e:
        print(f"Twin create error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/twins/<twin_id>', methods=['GET'])
def api_get_twin(twin_id):
    """Fetch one digital twin with snapshots and forecast."""
    try:
        twin = digital_twin_engine.get_twin(twin_id)
        if twin is None:
            return _json_error('Digital twin not found', 404)
        return jsonify({'twin': twin})
    except Exception as e:
        print(f"Twin detail error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/twins/<twin_id>/snapshot', methods=['POST'])
def api_add_twin_snapshot(twin_id):
    """Append a new snapshot to an existing digital twin."""
    try:
        data = _get_json_payload()
        patient_data = _normalize_patient_data(cast(Dict[str, Any], data.get('patient_data', data)))
        if not patient_data:
            return _json_error('No patient data provided')

        missing_fields = _missing_required_model_fields(patient_data)
        if missing_fields:
            return _json_error(f'Missing required fields: {missing_fields}')

        twin = digital_twin_engine.add_snapshot(
            twin_id=twin_id,
            patient_data=patient_data,
            predictor=_get_twin_predictor(),
        )
        if twin is None:
            return _json_error('Digital twin not found', 404)
        return jsonify({
            'message': 'Digital twin snapshot added successfully',
            'twin': twin,
        })
    except Exception as e:
        print(f"Twin snapshot error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/twins/<twin_id>/simulate', methods=['POST'])
def api_simulate_twin(twin_id):
    """Run a non-persistent digital twin simulation from the latest snapshot."""
    try:
        data = _get_json_payload()
        overrides = _normalize_patient_data(cast(Dict[str, Any], data.get('overrides', {})))
        scenario_name = cast(Optional[str], data.get('scenario_name'))
        simulation = digital_twin_engine.simulate(
            twin_id=twin_id,
            overrides=overrides,
            scenario_name=scenario_name,
            predictor=_get_twin_predictor(),
        )
        if simulation is None:
            return _json_error('Digital twin not found', 404)
        return jsonify({'simulation': simulation})
    except Exception as e:
        print(f"Twin simulate error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/twins/<twin_id>/trajectory', methods=['GET'])
def api_twin_trajectory(twin_id):
    """Return only the forecast trajectory for a digital twin."""
    try:
        twin = digital_twin_engine.get_twin(twin_id)
        if twin is None:
            return _json_error('Digital twin not found', 404)
        return jsonify({'forecast': twin.get('forecast', [])})
    except Exception as e:
        print(f"Twin trajectory error: {e}")
        traceback.print_exc()
        return _json_error(str(e), 500)

@app.route('/view_document/<doc_id>')
def view_document(doc_id):
    """View a document."""
    doc = document_manager.get_document(doc_id)
    if doc:
        return render_template('view_document.html', document=doc)
    else:
        flash('Document not found', 'danger')
        return redirect(url_for('documents'))

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    try:
        patient_data = _normalize_patient_data(_get_json_payload())

        if not patient_data:
            return _json_error('No patient data provided')

        dlog(f"Received patient data: {patient_data}")

        missing_fields = _missing_required_model_fields(patient_data)

        if missing_fields:
            return _json_error(f'Missing required fields: {missing_fields}')

        generator = _ensure_system_initialized()
        if generator is None:
            return _json_error('System initialization failed', 500)

        dlog("Making prediction...")
        prediction_results = generator.predict_patient(patient_data)
        dlog(f"Prediction results: {prediction_results}")

        class_names = ['Healthy Control', 'Parkinson\'s Disease', 'SWEDD', 'Prodromal PD']
        predicted_class = class_names[prediction_results['ensemble_prediction']]
        
        # Prepare response
        response = {
            'prediction': predicted_class,
            'confidence': float(prediction_results['confidence']),
            'probabilities': {
                'Healthy Control': float(prediction_results['ensemble_probabilities'][0]),
                'Parkinson\'s Disease': float(prediction_results['ensemble_probabilities'][1]),
                'SWEDD': float(prediction_results['ensemble_probabilities'][2]),
                'Prodromal PD': float(prediction_results['ensemble_probabilities'][3])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        dlog(f"Returning response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """API endpoint for generating comprehensive medical reports."""
    try:
        data = _get_json_payload()
        patient_data = _normalize_patient_data(cast(Dict[str, Any], data.get('patient_data', {})))
        patient_id = data.get('patient_id')

        if not patient_data:
            return _json_error('No patient data provided')

        missing_fields = _missing_required_model_fields(patient_data)
        if missing_fields:
            return _json_error(f'Missing required fields: {missing_fields}')

        generator = _ensure_system_initialized()
        if generator is None:
            return _json_error('System initialization failed', 500)

        report = generator.generate_full_report(patient_data, cast(Optional[str], patient_id))

        filename = _build_report_filename("report", patient_id, ".txt")
        filepath = generator.save_report(report, filename)
        
        response = {
            'report': report,
            'filename': filename,
            'filepath': filepath,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Report generation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_report_pdf', methods=['POST'])
def generate_report_pdf():
    """API endpoint for generating PDF reports."""
    try:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            from reportlab.graphics.shapes import Drawing, Rect, String
        except ImportError as e:
            print(f"PDF generation library import error: {e}")
            return _json_error(
                'PDF generation not available. ReportLab is not installed. Please run: pip install reportlab',
                500,
            )

        data = _get_json_payload()
        if not data:
            return _json_error('No data provided')
            
        patient_data = _normalize_patient_data(cast(Dict[str, Any], data.get('patient_data', {})))
        patient_id = data.get('patient_id', 'Unknown')
        prediction_results = cast(Dict[str, Any], data.get('prediction_results', {}))
        report_text = str(data.get('report_text', '') or '')

        if not patient_data:
            return _json_error('No patient data provided')

        missing_fields = _missing_required_model_fields(patient_data)
        if missing_fields:
            return _json_error(f'Missing required fields: {missing_fields}')

        if not prediction_results or not report_text:
            generator = _ensure_system_initialized()
            if generator is None:
                return _json_error('System initialization failed', 500)
            if not prediction_results:
                raw_prediction = generator.predict_patient(patient_data)
                class_names = ['Healthy Control', 'Parkinson\'s Disease', 'SWEDD', 'Prodromal PD']
                prediction_results = {
                    'prediction': class_names[raw_prediction['ensemble_prediction']],
                    'confidence': float(raw_prediction['confidence']),
                    'probabilities': {
                        'Healthy Control': float(raw_prediction['ensemble_probabilities'][0]),
                        'Parkinson\'s Disease': float(raw_prediction['ensemble_probabilities'][1]),
                        'SWEDD': float(raw_prediction['ensemble_probabilities'][2]),
                        'Prodromal PD': float(raw_prediction['ensemble_probabilities'][3]),
                    },
                }
            if not report_text:
                report_text = generator.generate_full_report(patient_data, cast(Optional[str], patient_id))
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                                rightMargin=50, leftMargin=50,
                                topMargin=50, bottomMargin=50)
        
        # Container for PDF elements
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=10,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#64748b'), # Slate-500
            spaceAfter=30,
            alignment=TA_LEFT,
            fontName='Helvetica'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0ea5e9'), # Sky-500
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#334155'), # Slate-700
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        # --- Header ---
        elements.append(Paragraph("NeuroAssess", title_style))
        elements.append(Paragraph("Parkinson's Disease Assessment Report", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 0.2*inch))
        
        # --- Meta Info Table ---
        meta_data = [
            [f"Patient ID: {patient_id}", f"Date: {datetime.now().strftime('%Y-%m-%d')}"],
            [f"Age: {patient_data.get('age', 'N/A')}", f"Sex: {'Male' if _coerce_float(patient_data.get('SEX')) == 1.0 else 'Female'}"],
        ]
        meta_table = Table(meta_data, colWidths=[3.5*inch, 3*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#475569')),
            ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 0.3*inch))

        # --- Diagnostic Score Card ---
        if prediction_results:
            pred_class = prediction_results.get('prediction', 'Unknown')
            confidence = prediction_results.get('confidence', 0)
            
            # Color coding
            bg_color = colors.HexColor('#f0f9ff') # Light blue
            border_color = colors.HexColor('#bae6fd')
            
            if 'Parkinson' in pred_class:
                status_color = colors.HexColor('#ef4444') # Red
            elif 'Healthy' in pred_class:
                status_color = colors.HexColor('#10b981') # Green
            else:
                status_color = colors.HexColor('#f59e0b') # Amber

            score_data = [
                [Paragraph("<b>PRIMARY DIAGNOSIS</b>", body_style), Paragraph("<b>CONFIDENCE SCORE</b>", body_style)],
                [Paragraph(f"<font size=16 color='{status_color.hexval()}'><b>{pred_class}</b></font>", body_style), 
                 Paragraph(f"<font size=16><b>{confidence*100:.1f}%</b></font>", body_style)]
            ]
            
            score_table = Table(score_data, colWidths=[3.5*inch, 3*inch])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), bg_color),
                ('BOX', (0,0), (-1,-1), 1, border_color),
                ('PADDING', (0,0), (-1,-1), 12),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(score_table)
            elements.append(Spacer(1, 0.3*inch))
            
            # --- Probability Chart ---
            elements.append(Paragraph("Probability Analysis", heading_style))
            
            probs = prediction_results.get('probabilities', {})
            if probs:
                # Custom Drawing for simple bars
                d = Drawing(400, 100)
                
                # Classes and their percentages
                labels = list(probs.keys())
                values = [p * 100 for p in probs.values()]
                colors_list = [colors.HexColor('#10b981'), colors.HexColor('#ef4444'), colors.HexColor('#f59e0b'), colors.HexColor('#3b82f6')]
                
                y_pos = 75
                for i, label in enumerate(labels):
                    val = values[i]
                    
                    # Label
                    d.add(String(0, y_pos, label, fontName="Helvetica", fontSize=9, fillColor=colors.HexColor('#475569')))
                    
                    # Background Bar
                    bg_rect = Rect(120, y_pos - 2, 200, 8)
                    bg_rect.fillColor = colors.HexColor('#f1f5f9')
                    bg_rect.strokeColor = colors.HexColor('#f1f5f9')
                    d.add(bg_rect)

                    # Foreground Bar
                    bar_width = (val / 100.0) * 200
                    fg_rect = Rect(120, y_pos - 2, bar_width, 8)
                    fg_rect.fillColor = colors_list[i % 4]
                    fg_rect.strokeColor = colors_list[i % 4]
                    d.add(fg_rect)
                    
                    # Percent text
                    d.add(String(330, y_pos, f"{val:.1f}%", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#334155')))
                    
                    y_pos -= 20

                elements.append(d)
                elements.append(Spacer(1, 0.2*inch))

        # --- Clinical Data Summary ---
        elements.append(Paragraph("Clinical Measurements", heading_style))
        
        # Organize data into a readable table
        clinical_data = []
        headers = ["Parameter", "Value", "Parameter", "Value"]
        clinical_data.append(headers)
        
        row = []
        for k, v in patient_data.items():
            if k in ['patient_id', 'age', 'SEX']: continue 
            # Format key nicely
            key_formatted = k.replace('_', ' ').title()
            # Format value
            val_formatted = str(v)
            if k == 'SEX':
                val_formatted = 'Male' if _coerce_float(v) == 1.0 else 'Female'
            elif k == 'fampd':
                family_history_display = {
                    1.0: 'First degree relative',
                    2.0: 'Other relative',
                    3.0: 'No family history',
                }
                val_formatted = family_history_display.get(_coerce_float(v), str(v))
            elif k == 'rem':
                val_formatted = 'Yes' if _coerce_float(v) == 1.0 else 'No'
                
            row.append(key_formatted)
            row.append(val_formatted)
            
            if len(row) == 4:
                clinical_data.append(row)
                row = []
        
        if row:  # remaining
            while len(row) < 4:
                row.append("")
            clinical_data.append(row)
            
        clinical_table = Table(clinical_data, colWidths=[1.8*inch, 1.4*inch, 1.8*inch, 1.4*inch])
        clinical_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8fafc')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#334155')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
             ('FONTSIZE', (0,1), (-1,-1), 9),
        ]))
        elements.append(clinical_table)
        elements.append(Spacer(1, 0.3*inch))


        # --- Detailed Report Text ---
        if report_text:
            elements.append(Paragraph("Detailed Clinical Analysis", heading_style))
            
            # Simple markdown parsing (bolding)
            lines = report_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    elements.append(Spacer(1, 0.05*inch))
                    continue
                
                # Identifying bold headings in text
                if line.startswith('**') and line.endswith('**'):
                    heading_text = html.escape(line.strip('* '))
                    elements.append(Paragraph(heading_text, ParagraphStyle('SubHead', parent=body_style, fontName='Helvetica-Bold', fontSize=11, spaceBefore=6)))
                    continue
                
                escaped_line = html.escape(line)
                formatted_line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped_line)
                
                # Handle bullet points
                if line.startswith('- '):
                    bullet_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html.escape(line[2:]))
                    elements.append(Paragraph(f"• {bullet_text}", ParagraphStyle('Bullet', parent=body_style, leftIndent=10)))
                else:
                    elements.append(Paragraph(formatted_line, body_style))


        # --- Footer ---
        elements.append(Spacer(1, 0.5*inch))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#94a3b8'),
            alignment=TA_CENTER,
            spaceBefore=10
        )
        
        footer_text = """
        <b>DISCLAIMER:</b> This report is generated by an AI-powered system (NeuroAssess) for research and educational purposes only.<br/>
        It should not be used as a substitute for professional medical diagnosis or treatment.
        """
        elements.append(Paragraph(footer_text, footer_style))
        
        # Build PDF
        try:
            doc.build(elements)
        except Exception as build_error:
            print(f"Error building PDF document: {build_error}")
            traceback.print_exc()
            return jsonify({'error': f'Error creating PDF document: {str(build_error)}'}), 500
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        if len(pdf_data) == 0:
            return jsonify({'error': 'Generated PDF is empty'}), 500
        
        # Return PDF as response
        response = send_file(
            io.BytesIO(pdf_data),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'PD_Assessment_{_safe_report_token(patient_id, "patient")}_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
        return response
        
    except Exception as e:
        error_msg = f"PDF generation error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/download_report/<filename>')
def download_report(filename):
    """Download generated report file."""
    try:
        if not _is_safe_report_filename(filename):
            return _json_error('Invalid report filename')
        reports_dir = _reports_dir()
        filepath = reports_dir / filename
        if filepath.exists():
            return send_from_directory(str(reports_dir), filename, as_attachment=True)
        return jsonify({'error': 'Report file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate_data', methods=['POST'])
def validate_data():
    """Validate patient data before processing."""
    try:
        patient_data = _normalize_patient_data(_get_json_payload())

        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Age validation
        age = patient_data.get('age')
        if age is not None:
            if age < 18 or age > 100:
                validation_results['errors'].append('Age must be between 18 and 100')
                validation_results['valid'] = False
            elif age > 80:
                validation_results['warnings'].append('Advanced age may affect assessment accuracy')
        
        # BMI validation
        bmi = patient_data.get('BMI')
        if bmi is not None:
            if bmi < 15 or bmi > 50:
                validation_results['errors'].append('BMI must be between 15 and 50')
                validation_results['valid'] = False

        fampd = patient_data.get('fampd')
        if fampd is not None and fampd not in {1.0, 2.0, 3.0}:
            validation_results['errors'].append('Family history must be one of: No family history, First degree relative, Other relative')
            validation_results['valid'] = False
        
        # MoCA score validation
        moca = patient_data.get('moca')
        if moca is not None:
            if moca < 0 or moca > 30:
                validation_results['errors'].append('MoCA score must be between 0 and 30')
                validation_results['valid'] = False
        
        # Symptom scores validation (typically 0-4 scale)
        symptom_fields = ['sym_tremor', 'sym_rigid', 'sym_brady', 'sym_posins']
        for field in symptom_fields:
            value = patient_data.get(field)
            if value is not None and (value < 0 or value > 4):
                validation_results['errors'].append(f'{field} must be between 0 and 4')
                validation_results['valid'] = False
        
        return jsonify(validation_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_patient_report', methods=['POST'])
def generate_patient_report():
    """Generate patient-friendly report."""
    try:
        data = _get_json_payload()
        patient_data = _normalize_patient_data(cast(Dict[str, Any], data.get('patient_data', {})))
        patient_id = data.get('patient_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400

        missing_fields = _missing_required_model_fields(patient_data)
        if missing_fields:
            return _json_error(f'Missing required fields: {missing_fields}')
        
        # Initialize system if needed
        if report_generator is None:
            if not initialize_system():
                return jsonify({'error': 'System initialization failed'}), 500
        
        # Get predictions
        prediction_results = report_generator.predict_patient(patient_data)
        
        # Generate patient report
        patient_report = dual_report_manager.patient_generator.generate_report(
            prediction_results, patient_data
        )
        
        # Save report
        report_dir = str(_reports_dir())
        filename = _build_report_filename("patient_report", patient_id, ".txt")
        filepath = os.path.join(report_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(patient_report)
        
        return jsonify({
            'report': patient_report,
            'filename': filename,
            'filepath': filepath,
            'report_type': 'patient',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Patient report generation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_doctor_report', methods=['POST'])
def generate_doctor_report():
    """Generate clinical report for healthcare professionals."""
    try:
        data = _get_json_payload()
        patient_data = _normalize_patient_data(cast(Dict[str, Any], data.get('patient_data', {})))
        patient_id = data.get('patient_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400

        missing_fields = _missing_required_model_fields(patient_data)
        if missing_fields:
            return _json_error(f'Missing required fields: {missing_fields}')
        
        # Initialize system if needed
        if report_generator is None:
            if not initialize_system():
                return jsonify({'error': 'System initialization failed'}), 500
        
        # Get predictions
        prediction_results = report_generator.predict_patient(patient_data)
        
        # Get literature insights
        literature_insights = ""
        try:
            # Try to get relevant medical literature
            class_names = ['HC', 'PD', 'SWEDD', 'PRODROMAL']
            pred_class = class_names[prediction_results['ensemble_prediction']]
            literature_insights = report_generator._get_literature_insights(pred_class, patient_data)
        except:
            pass
        
        # Generate doctor report
        doctor_report = dual_report_manager.doctor_generator.generate_report(
            prediction_results, patient_data, literature_insights
        )
        
        # Save report
        report_dir = str(_reports_dir())
        filename = _build_report_filename("clinical_report", patient_id, ".txt")
        filepath = os.path.join(report_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doctor_report)
        
        return jsonify({
            'report': doctor_report,
            'filename': filename,
            'filepath': filepath,
            'report_type': 'doctor',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Doctor report generation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_both_reports', methods=['POST'])
def generate_both_reports():
    """Generate both patient and doctor reports."""
    try:
        data = _get_json_payload()
        patient_data = _normalize_patient_data(cast(Dict[str, Any], data.get('patient_data', {})))
        patient_id = data.get('patient_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400

        missing_fields = _missing_required_model_fields(patient_data)
        if missing_fields:
            return _json_error(f'Missing required fields: {missing_fields}')
        
        # Initialize system if needed
        if report_generator is None:
            if not initialize_system():
                return jsonify({'error': 'System initialization failed'}), 500
        
        # Get predictions
        prediction_results = report_generator.predict_patient(patient_data)
        
        # Get literature insights for doctor report
        literature_insights = ""
        try:
            class_names = ['HC', 'PD', 'SWEDD', 'PRODROMAL']
            pred_class = class_names[prediction_results['ensemble_prediction']]
            literature_insights = report_generator._get_literature_insights(pred_class, patient_data)
        except:
            pass
        
        # Generate both reports
        reports = dual_report_manager.generate_both_reports(
            prediction_results, patient_data, literature_insights
        )
        
        # Save both reports
        report_dir = str(_reports_dir())
        saved_paths = dual_report_manager.save_reports(reports, report_dir, patient_id)
        
        return jsonify({
            'patient_report': reports['patient_report'],
            'doctor_report': reports['doctor_report'],
            'patient_report_path': saved_paths['patient_report_path'],
            'doctor_report_path': saved_paths['doctor_report_path'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Dual report generation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_status')
def system_status():
    """Get system status and model information."""
    try:
        bridge = digital_twin_engine.bridge
        bridge_status = bridge.get_status() if bridge else {"models_loaded": False}
        report_models_loaded = bool(
            report_generator is not None
            and report_generator.ensemble is not None
            and report_generator.preprocessor is not None
        )
        bridge_ready = bool(bridge_status.get('models_loaded', False))

        status = {
            'system_initialized': bool(report_generator is not None or bridge_ready),
            'models_loaded': bool(report_models_loaded or bridge_ready),
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/favicon.ico')
def favicon():
    """Serve favicon if present; otherwise return no-content to avoid noisy 404 logs."""
    icon_path = Path(static_dir) / 'favicon.ico'
    if icon_path.exists():
        return send_from_directory(static_dir, 'favicon.ico')
    return ('', 204)


@app.route('/api/model_metrics_summary')
def model_metrics_summary():
    """Expose the checked-in evaluation summary to frontend clients."""
    try:
        return jsonify(_load_metrics_summary())
    except Exception as e:
        return _json_error(str(e), 500)

@app.route('/api/health')
def api_health():
    """Quick health check with MODELS_LOADED status flag."""
    try:
        bridge = digital_twin_engine.bridge
        bridge_status = bridge.get_status() if bridge else {"models_loaded": False}
        return jsonify({
            'status': 'ok',
            'models_loaded': bridge_status.get('models_loaded', False),
            'system_initialized': report_generator is not None,
            'progression_fitted': bridge_status.get('progression_fitted', False),
            'treatment_fitted': bridge_status.get('treatment_fitted', False),
            'risk_available': bridge_status.get('risk_available', False),
            'silhouette_score': bridge_status.get('silhouette_score'),
            'treatment_r_squared': bridge_status.get('treatment_r_squared'),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'models_loaded': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }), 500


@app.route('/api/health/deep')
def health_deep():
    """Deep health check: validates required artifacts and basic loadability."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_dir = os.path.join(project_root, 'models', 'saved')

        required = {
            'lightgbm_model.joblib': os.path.join(model_dir, 'lightgbm_model.joblib'),
            'xgboost_model.joblib': os.path.join(model_dir, 'xgboost_model.joblib'),
            'svm_model.joblib': os.path.join(model_dir, 'svm_model.joblib'),
            'multimodal_ensemble.joblib': os.path.join(model_dir, 'multimodal_ensemble.joblib'),
            'traditional_preprocessor.joblib': os.path.join(model_dir, 'traditional_preprocessor.joblib'),
            'traditional_class_mapping.json': os.path.join(model_dir, 'traditional_class_mapping.json'),
        }

        artifacts = {k: os.path.exists(v) for k, v in required.items()}

        details = {
            'artifacts': artifacts,
            'docs_count': document_manager.get_document_count(),
            'system_initialized': report_generator is not None,
            'timestamp': datetime.now().isoformat(),
        }

        # Optional deeper load check
        load_ok = False
        load_error = None
        try:
            from rag_system import ReportGenerator
            rg = ReportGenerator(knowledge_base, docs_dir=os.path.join(project_root, 'medical_docs'))
            rg.load_models()
            load_ok = True
        except Exception as e:
            load_error = str(e)

        details['model_load_ok'] = load_ok
        if load_error:
            details['model_load_error'] = load_error

        ok = all(artifacts.values()) and load_ok
        code = 200 if ok else 503
        details['status'] = 'ok' if ok else 'degraded'
        return jsonify(details), code

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('reports', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("Starting Parkinson's Disease Assessment Web Interface...")
    print("Initializing ML models...")
    
    # Initialize system on startup
    if initialize_system():
        print("System ready!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize system. Please check model files.")
