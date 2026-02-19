"""
Web Interface for Parkinson's Disease Assessment System.
Flask-based web application for patient data input and automated report generation.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from flask_cors import CORS
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
CORS(app)
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
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, mm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, HRFlowable, Image
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
            from reportlab.graphics.shapes import Drawing, Rect, String, Group, Line
            from reportlab.graphics.charts.barcharts import HorizontalBarChart
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
            
        patient_data = data.get('patient_data', {})
        patient_id = data.get('patient_id', 'Unknown')
        prediction_results = data.get('prediction_results', {})
        report_text = data.get('report_text', '')
        
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
            [f"Patient ID: {patient_id}", f"Date: {datetime.now().strftime('%Y-%m-%d')}__"],
            [f"Age: {patient_data.get('age', 'N/A')}", f"Sex: {'Male' if str(patient_data.get('SEX','')) == '1' else 'Female'}"],
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
                    d.add(Rect(120, y_pos-2, 200, 8, fillColor=colors.HexColor('#f1f5f9'), strokeColor=None))
                    
                    # Foreground Bar
                    bar_width = (val / 100.0) * 200
                    d.add(Rect(120, y_pos-2, bar_width, 8, fillColor=colors_list[i % 4], strokeColor=None))
                    
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
                val_formatted = 'Male' if str(v) == '1' else 'Female'
            elif k == 'fampd':
                val_formatted = 'Yes' if str(v) == '1' else 'No'
            elif k == 'rem':
                val_formatted = 'Yes' if str(v) == '1' else 'No'
                
            row.append(key_formatted)
            row.append(val_formatted)
            
            if len(row) == 4:
                clinical_data.append(row)
                row = []
        
        if row: # remaining
            while len(row) < 4: row.append("")
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
                    elements.append(Paragraph(line.replace('**', ''), ParagraphStyle('SubHead', parent=body_style, fontName='Helvetica-Bold', fontSize=11, spaceBefore=6)))
                    continue
                
                # Replace markdown bold with HTML bold
                formatted_line = line.replace('**', '<b>').replace('**', '</b>')
                
                # Handle bullet points
                if line.startswith('- '):
                    elements.append(Paragraph(f"• {formatted_line[2:]}", ParagraphStyle('Bullet', parent=body_style, leftIndent=10)))
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
            download_name=f'PD_Assessment_{patient_id}_{datetime.now().strftime("%Y%m%d")}.pdf'
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