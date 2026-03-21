"""Evaluate transformer models (BioGPT, PubMedBERT, Clinical-T5) on the leak-free split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
from data_preprocessing import DataPreprocessor  # type: ignore
from document_manager import DocumentManager  # type: ignore
from models.medical_transformers import (  # type: ignore
    BioMistralClassifier as BioGPTForTabular,
    ClinicalT5Classifier as ClinicalT5ForTabular,
    PubMedBERTClassifier as PubMedBERTForTabular,
)
from models.transformer_models import TabularDataset  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluation_results" / "model_metrics"
CLASS_REPORT_DIR = EVAL_DIR / "classification_reports"
CONF_MATRIX_DIR = EVAL_DIR / "confusion_matrices"
ROC_DIR = EVAL_DIR / "roc_curves"
SUMMARY_PATH = EVAL_DIR / "model_metrics_summary_transformers.csv"
LATEST_JSON = EVAL_DIR / "transformer_metrics_latest.json"
LEAK_FREE_SPLIT_PATH = ROOT / "evaluation_results" / "leak_free_split.npz"
LEAK_FREE_META_PATH = ROOT / "evaluation_results" / "leak_free_split_meta.joblib"
CLASS_NAMES = ["HC", "PD", "SWEDD", "PRODROMAL"]

for directory in (CLASS_REPORT_DIR, CONF_MATRIX_DIR, ROC_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _load_or_create_leak_free_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if LEAK_FREE_SPLIT_PATH.exists() and LEAK_FREE_META_PATH.exists():
        split = np.load(LEAK_FREE_SPLIT_PATH)
        meta = joblib.load(LEAK_FREE_META_PATH)
        feature_names = meta.get("feature_names") if isinstance(meta, dict) else None
        if feature_names is None:
            raise ValueError("Leak-free metadata missing feature names")
        print("Loaded cached leak-free split artifacts.")
        return split["X_train"], split["X_test"], split["y_train"], split["y_test"], feature_names

    preprocessor = DataPreprocessor()
    file_paths = [
        ROOT / "PPMI_Curated_Data_Cut_Public_20240129.csv",
        ROOT / "PPMI_Curated_Data_Cut_Public_20241211.csv",
        ROOT / "PPMI_Curated_Data_Cut_Public_20250321.csv",
        ROOT / "PPMI_Curated_Data_Cut_Public_20250714.csv",
    ]
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        file_paths,
        test_size=0.2,
        use_patient_split=True,
    )
    feature_names = preprocessor.get_feature_names()
    LEAK_FREE_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        LEAK_FREE_SPLIT_PATH,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    joblib.dump({"feature_names": feature_names}, LEAK_FREE_META_PATH)
    print("Saved new leak-free split artifacts.")
    return X_train, X_test, y_train, y_test, feature_names


def _prepare_batch(batch, device):
    if len(batch) == 3:
        data, targets, contexts = batch
        contexts = list(contexts)
    else:
        data, targets = batch
        contexts = None
    data = data.to(device)
    targets = targets.to(device)
    return data, targets, contexts


def _build_context_cache(features: np.ndarray, feature_names: list[str], doc_manager: DocumentManager) -> list[str]:
    cache = []
    total = len(features)
    for idx, row in enumerate(features):
        feature_desc = {name: float(val) for name, val in zip(feature_names, row)}
        query_parts = []
        for symptom_key, col in {
            "tremor": "sym_tremor",
            "rigidity": "sym_rigid",
            "bradykinesia": "sym_brady",
            "postural instability": "sym_posins",
        }.items():
            if feature_desc.get(col, 0) > 0:
                query_parts.append(f"{symptom_key} severity:{feature_desc[col]:.2f}")
        moca = feature_desc.get("moca", 30)
        if moca < 26:
            query_parts.append("cognitive impairment")
        age = feature_desc.get("age", 0)
        if age:
            query_parts.append(f"age {int(age)}")
        if feature_desc.get("fampd", 0) > 0:
            query_parts.append("family history Parkinson's disease")
        query = "Parkinson's disease " + " ".join(query_parts)
        passages = doc_manager.extract_relevant_passages(query, top_k=2)
        if not passages:
            cache.append("")
        else:
            combined = []
            for passage in passages:
                title = passage.get("doc_title") or passage.get("doc_id") or "document"
                combined.append(f"From '{title}' {passage['text'][:300]}...")
            cache.append(" ".join(combined))
        if (idx + 1) % 250 == 0 or idx + 1 == total:
            print(f"  Cached RAG context for {idx + 1}/{total} samples")
    return cache


def _save_outputs(model_name: str, report: str, cm: np.ndarray, y_test: np.ndarray, y_prob: np.ndarray) -> None:
    (CLASS_REPORT_DIR / f"{model_name}.txt").write_text(
        f"{model_name} Classification Report (leak-free split)\n" + "-" * 60 + "\n" + report
    )
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(CONF_MATRIX_DIR / f"{model_name}_confusion_matrix.csv")

    y_bin = label_binarize(y_test, classes=range(len(CLASS_NAMES)))
    for idx, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.to_csv(ROC_DIR / f"{model_name}_class_{class_name}_roc.csv", index=False)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating transformers on {device} (leak-free split)")
    X_train, X_test, y_train, y_test, feature_names = _load_or_create_leak_free_split()
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    docs_path = ROOT / "medical_docs"
    doc_manager = DocumentManager(docs_dir=str(docs_path))
    print("Building RAG contexts for test set...")
    test_contexts = _build_context_cache(X_test, feature_names, doc_manager)

    test_dataset = TabularDataset(X_test, y_test, feature_names, contexts=test_contexts)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_dir = ROOT / "models" / "saved"
    model_configs = {
        "BioGPT": {
            "builder": lambda: BioGPTForTabular(
                input_dim=X_train.shape[1],
                num_classes=len(CLASS_NAMES),
                dropout=0.15,
                train_decoder_layers=6,
            ),
            "checkpoints": [
                model_dir / "biogpt_transformer.pth",
                model_dir / "biogpt.pth",
                model_dir / "biomistral.pth",
            ],
        },
        "PubMedBERT": {
            "builder": lambda: PubMedBERTForTabular(
                input_dim=X_train.shape[1],
                num_classes=len(CLASS_NAMES),
                dropout=0.15,
                freeze_bert=False,
            ),
            "checkpoints": [
                model_dir / "pubmedbert_transformer.pth",
                model_dir / "pubmedbert.pth",
            ],
        },
        "Clinical-T5": {
            "builder": lambda: ClinicalT5ForTabular(
                input_dim=X_train.shape[1],
                num_classes=len(CLASS_NAMES),
                dropout=0.15,
                freeze_encoder=False,
            ),
            "checkpoints": [
                model_dir / "clinical_t5_transformer.pth",
                model_dir / "clinicalt5_transformer.pth",
                model_dir / "clinical_t5.pth",
            ],
        },
    }

    summary_rows = []

    for pretty_name, cfg in model_configs.items():
        checkpoint_path = next((path for path in cfg["checkpoints"] if path.exists()), None)
        if checkpoint_path is None:
            expected = ", ".join(path.name for path in cfg["checkpoints"])
            print(f"[WARN] Skipping {pretty_name}: no checkpoint found ({expected})")
            continue

        print(f"\nEvaluating {pretty_name}...")
        model = cfg["builder"]().to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        all_targets = []
        all_preds = []
        all_prob = []

        with torch.no_grad():
            for batch in test_loader:
                data, targets, contexts = _prepare_batch(batch, device)
                outputs = model(data, contexts)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_prob.append(probs.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.vstack(all_prob)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        report = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        _save_outputs(pretty_name, report, cm, y_true, y_prob)

        summary_rows.append(
            {
                "model": pretty_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        print(
            f"{pretty_name} -> accuracy {accuracy:.4f}, precision {precision:.4f}, recall {recall:.4f}, f1 {f1:.4f}"
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_PATH, index=False)
        LATEST_JSON.write_text(json.dumps(summary_rows, indent=2))
        print(f"Saved transformer summary to {SUMMARY_PATH}")
    else:
        print("No transformer metrics were generated; check checkpoints.")


if __name__ == "__main__":
    main()
