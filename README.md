# Parkinson's Disease Assessment Portal

A comprehensive medical assessment system for Parkinson's disease diagnosis using the PPMI (Parkinson's Progression Markers Initiative) curated dataset. The system combines traditional machine learning, medical transformer models, and RAG-based report generation.

## Dataset

Uses the **PPMI Curated Data Cut** CSV files containing patient data with the following key features:

| Category | Features |
|----------|----------|
| Demographics | Age, Sex, Education Years, Race, BMI |
| Family History | Family PD history (categorical + binary) |
| Motor Symptoms | Tremor, Rigidity, Bradykinesia, Postural Instability |
| Non-Motor | REM sleep, Epworth Sleepiness, Depression (GDS), Anxiety (STAI) |
| Cognitive | MoCA, Clock Drawing, Benton JLO |

**Target classes (COHORT):** HC (Healthy Control) · PD (Parkinson's Disease) · SWEDD · PRODROMAL

## Architecture

```
├── src/
│   ├── data_preprocessing.py      # Patient-level leak-free data pipeline
│   ├── web_interface.py            # Flask web app
│   ├── rag_system.py               # Medical knowledge base + report generation
│   ├── document_manager.py         # PDF/text document indexing (TF-IDF)
│   ├── feature_mapping.py          # Patient questionnaire ↔ PPMI feature mapping
│   ├── analyze_data.py             # Dataset EDA script
│   ├── train_traditional_models.py # Train LightGBM, XGBoost, SVM
│   ├── train_transformer_models.py # Train PubMedBERT, BioGPT, Clinical-T5
│   ├── train_multimodal.py         # Train multimodal ensemble
│   ├── evaluate_traditional_models.py
│   └── models/
│       ├── traditional_ml.py       # LightGBM, XGBoost, SVM wrappers
│       ├── transformer_models.py   # DistilBERT, BioBERT, PubMedBERT for tabular
│       ├── medical_transformers.py # PubMedBERT, BioGPT, Clinical-T5 classifiers
│       └── multimodal_ml.py        # Stacking ensemble
├── templates/                      # Flask HTML templates
├── static/                         # CSS, JS assets
├── medical_docs/                   # Medical literature for RAG
├── models/saved/                   # Trained model weights
├── start_server.py                 # Entry point for web app
└── requirements.txt
```

## Features

- **Leak-Free Preprocessing**: Patient-level train/test split ensures no patient appears in both sets
- **Traditional ML**: LightGBM, XGBoost, SVM with class weight balancing
- **Medical Transformers**: PubMedBERT (encoder), BioGPT (decoder), Clinical-T5 (encoder-decoder)
- **Multimodal Ensemble**: Stacking ensemble combining all model predictions
- **RAG-Enhanced Reports**: Retrieves medical literature to generate comprehensive diagnostic reports
- **Web Interface**: Patient assessment form with automated report generation and PDF export

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (recommended for GPU training)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

`requirements.txt` now includes `sacremoses`, which BioGPT needs for tokenization. If the A4000 preflight fails on `sacremoses`, rerun `pip install -r requirements.txt` inside the training venv.

## RTX A4000 Setup

For the RTX A4000 training machine, use this flow from the project root.

Ubuntu:

```bash
source venv/bin/activate
bash check_a4000_ready.sh
bash train_a4000_models.sh
```

Windows:

```bat
venv\Scripts\activate
python check_a4000_ready.py
train_a4000_models.bat
```

What the preflight checks:

- CUDA-enabled PyTorch import and `torch.cuda.is_available()`
- detected GPU name, CUDA version, and VRAM
- BioGPT tokenizer dependency (`sacremoses`)
- required PPMI CSV files
- `medical_docs/` availability for RAG training
- free disk space and output path write access

Helper scripts:

- `check_a4000_ready.sh` / `check_a4000_ready.bat` run the GPU/data preflight
- `train_a4000_models.sh` / `train_a4000_models.bat` run preflight, then start training with `--gpu-profile rtx-a4000`
- `resume_a4000_training.sh` / `resume_a4000_training.bat` resume the same run if the session is interrupted

The A4000 training recipe now defaults to class-weighted focal loss and keeps the best transformer checkpoint by validation F1, with validation loss used only as a tie-breaker.

Recommended direct commands:

```bash
python src/train_model_suite.py train --run-name a4000_full --gpu-profile rtx-a4000 --epochs 30 --patience 8 --traditional-trials 6 --transformer-trials 6 --transformer-loss focal --focal-gamma 1.5
python src/train_model_suite.py resume --run-name a4000_full --gpu-profile rtx-a4000 --epochs 30 --patience 8 --traditional-trials 6 --transformer-trials 6 --transformer-loss focal --focal-gamma 1.5
python src/train_model_suite.py status --run-name a4000_full
```

## Usage

### Train Models

```bash
cd src

# Train traditional ML models
python train_traditional_models.py

# Train transformer models (requires GPU recommended)
python train_transformer_models.py

# Train multimodal ensemble
python train_multimodal.py
```

For the full resumable training pipeline with the A4000 profile, run from the project root instead of `src/`:

```bash
bash train_a4000_models.sh
```

### Run Web App

```bash
# From project root
python start_server.py
# Access at http://localhost:5000
```

### Evaluate Models

```bash
cd src
python evaluate_traditional_models.py
```

## Model Performance

Models are evaluated on a held-out test set using patient-level splitting:

| Model | Type |
|-------|------|
| LightGBM | Gradient Boosting |
| XGBoost | Gradient Boosting |
| SVM (RBF) | Support Vector Machine |
| PubMedBERT | Encoder-only Transformer |
| BioGPT | Decoder-only Transformer |
| Clinical-T5 | Encoder-Decoder Transformer |
| Multimodal Ensemble | Stacking (all above) |

## License

This project uses data from the [Parkinson's Progression Markers Initiative (PPMI)](https://www.ppmi-info.org/).
