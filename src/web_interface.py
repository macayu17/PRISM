"""
Web Interface for Parkinson's Disease Assessment System.
Flask-based web application for patient data input and automated report generation.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import json
from datetime import datetime
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from rag_system import ReportGenerator, MedicalKnowledgeBase
from document_manager import DocumentManager
from dual_report_generator import DualReportManager

# Set template and static folders to the directories in the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
template_dir = os.path.join(project_root, 'templates')
static_dir = os.path.join(project_root, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'parkinson_assessment_secret_key_2024'

# Initialize global components
report_generator = None
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
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    """Patient assessment form page."""
    return render_template('assessment.html')

@app.route('/about')
#def about():
 #   """About page with system information."""
  #  return render_template('about.html', knowledge_base=knowledge_base)

@app.route('/documents')
def documents():
    """Document management page."""
    docs = document_manager.get_all_documents()
    return render_template('documents.html', documents=docs)

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Handle document upload."""
    try:
        file = request.files['document']
        title = request.form['title']
        author = request.form['author']
        
        if file and title:
            filename = file.filename
            file_path = os.path.join('medical_docs', filename)
            file.save(file_path)
            
            # Add document to document manager
            document_manager.add_document(file_path, title=title, author=author)
            
            flash('Document uploaded successfully!', 'success')
        else:
            flash('Please provide a file and title', 'danger')
            
    except Exception as e:
        flash(f'Error uploading document: {str(e)}', 'danger')
        
    return redirect(url_for('documents'))

@app.route('/delete_document/<doc_id>', methods=['POST'])
def delete_document(doc_id):
    """Delete a document."""
    try:
        document_manager.remove_document(doc_id)
        flash('Document deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting document: {str(e)}', 'danger')
        
    return redirect(url_for('documents'))

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
        # Get patient data from request
        patient_data = request.json
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        print(f"Received patient data: {patient_data}")
        
        # Validate required fields
        required_fields = ['age', 'SEX', 'EDUCYRS', 'BMI']
        missing_fields = [field for field in required_fields if field not in patient_data]
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Initialize system if not already done
        if report_generator is None:
            print("Initializing system...")
            if not initialize_system():
                return jsonify({'error': 'System initialization failed'}), 500
        
        # Make prediction
        print("Making prediction...")
        prediction_results = report_generator.predict_patient(patient_data)
        print(f"Prediction results: {prediction_results}")
        
        # Map prediction to class name - 4 classes
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
        
        print(f"Returning response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """API endpoint for generating comprehensive medical reports."""
    try:
        # Get patient data from request
        data = request.json
        patient_data = data.get('patient_data', {})
        patient_id = data.get('patient_id', None)
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Initialize system if not already done
        if report_generator is None:
            if not initialize_system():
                return jsonify({'error': 'System initialization failed'}), 500
        
        # Generate report
        report = report_generator.generate_full_report(patient_data, patient_id)
        
        # Save report
        filename = f"report_{patient_id or datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = report_generator.save_report(report, filename)
        
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
        # Import here to catch any issues
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, HRFlowable
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            import io
        except ImportError as e:
            print(f"PDF generation library import error: {e}")
            return jsonify({
                'error': 'PDF generation not available. ReportLab is not installed. Please run: pip install reportlab'
            }), 500
        
        # Get report data from request
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        report_text = data.get('report_text', '')
        patient_id = data.get('patient_id', 'Unknown')
        
        if not report_text:
            return jsonify({'error': 'No report text provided'}), 400
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for PDF elements
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leading=16
        )
        
        # Add title
        title = Paragraph("Parkinson's Disease Assessment Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add metadata
        metadata = [
            ['Report Date:', datetime.now().strftime('%B %d, %Y %I:%M %p')],
            ['Patient ID:', str(patient_id)],
            ['Generated By:', 'AI-Powered Assessment System']
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#7f8c8d')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(metadata_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add horizontal line
        elements.append(HRFlowable(width="100%", thickness=1, 
                                   color=colors.HexColor('#bdc3c7'),
                                   spaceAfter=0.2*inch))
        
        # Parse and format report content
        lines = report_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 0.1*inch))
                continue
            
            # Check if line is a heading (starts with specific patterns)
            if any(line.startswith(prefix) for prefix in [
                'PATIENT INFORMATION', 'ASSESSMENT RESULTS', 'CLINICAL ANALYSIS',
                'MEDICAL RECOMMENDATIONS', 'LITERATURE INSIGHTS', 'DIAGNOSTIC SUMMARY',
                'RISK ASSESSMENT', 'TREATMENT RECOMMENDATIONS', 'FOLLOW-UP',
                '═══', '---'
            ]):
                if line.startswith('═') or line.startswith('---'):
                    continue  # Skip separator lines
                para = Paragraph(line, heading_style)
                elements.append(para)
            else:
                # Regular body text - sanitize HTML characters
                line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Handle bold markers (convert ** to <b> tags properly)
                parts = line.split('**')
                if len(parts) > 1:
                    formatted_line = ''
                    for i, part in enumerate(parts):
                        if i % 2 == 1:  # Odd indices should be bold
                            formatted_line += f'<b>{part}</b>'
                        else:
                            formatted_line += part
                    line = formatted_line
                
                try:
                    para = Paragraph(line, body_style)
                    elements.append(para)
                except Exception as e:
                    # If there's an error with this line, add it as plain text
                    print(f"Error formatting line: {e}")
                    elements.append(Paragraph(str(line)[:500], body_style))  # Limit length
        
        # Add footer
        elements.append(Spacer(1, 0.5*inch))
        elements.append(HRFlowable(width="100%", thickness=1, 
                                   color=colors.HexColor('#bdc3c7'),
                                   spaceBefore=0.2*inch))
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#95a5a6'),
            alignment=TA_CENTER
        )
        
        footer_text = """
        <b>DISCLAIMER:</b> This report is generated by an AI-powered system for research and educational purposes only.<br/>
        It should not be used as a substitute for professional medical diagnosis or treatment.<br/>
        Always consult qualified healthcare professionals for medical advice.
        """
        elements.append(Spacer(1, 0.2*inch))
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
            download_name=f'PD_Report_{patient_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
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
        filepath = os.path.join('reports', filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'Report file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate_data', methods=['POST'])
def validate_data():
    """Validate patient data before processing."""
    try:
        patient_data = request.json
        
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
        data = request.json
        patient_data = data.get('patient_data', {})
        patient_id = data.get('patient_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
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
        report_dir = os.path.join(os.path.dirname(current_dir), 'reports')
        os.makedirs(report_dir, exist_ok=True)
        filename = f"patient_report_{patient_id}.txt"
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
        data = request.json
        patient_data = data.get('patient_data', {})
        patient_id = data.get('patient_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
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
        report_dir = os.path.join(os.path.dirname(current_dir), 'reports')
        os.makedirs(report_dir, exist_ok=True)
        filename = f"clinical_report_{patient_id}.txt"
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
        data = request.json
        patient_data = data.get('patient_data', {})
        patient_id = data.get('patient_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
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
        report_dir = os.path.join(os.path.dirname(current_dir), 'reports')
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
        status = {
            'system_initialized': report_generator is not None,
            'models_loaded': False,
            'timestamp': datetime.now().isoformat()
        }
        
        if report_generator is not None:
            status['models_loaded'] = (
                report_generator.ensemble is not None and 
                report_generator.preprocessor is not None
            )
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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