"""Re-train and evaluate traditional models on the leak-free patient split."""
import os
import sys
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).parent))
from data_preprocessing import DataPreprocessor  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluation_results" / "model_metrics"
CLASS_REPORT_DIR = EVAL_DIR / "classification_reports"
CONF_MATRIX_DIR = EVAL_DIR / "confusion_matrices"
PLOTS_DIR = EVAL_DIR / "plots"
ROC_DIR = EVAL_DIR / "roc_curves"

for path in (CLASS_REPORT_DIR, CONF_MATRIX_DIR, PLOTS_DIR, ROC_DIR):
    path.mkdir(parents=True, exist_ok=True)

FILE_PATHS = [
    ROOT / "PPMI_Curated_Data_Cut_Public_20240129.csv",
    ROOT / "PPMI_Curated_Data_Cut_Public_20241211.csv",
    ROOT / "PPMI_Curated_Data_Cut_Public_20250321.csv",
    ROOT / "PPMI_Curated_Data_Cut_Public_20250714.csv",
]
CLASS_NAMES = ["HC", "PD", "SWEDD", "PRODROMAL"]


def load_or_create_split():
    split_path = ROOT / "evaluation_results" / "leak_free_split.npz"
    meta_path = ROOT / "evaluation_results" / "leak_free_split_meta.joblib"
    if split_path.exists() and meta_path.exists():
        split = np.load(split_path)
        meta = joblib.load(meta_path)
        return split, meta

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        FILE_PATHS,
        test_size=0.2,
        use_patient_split=True,
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    joblib.dump(
        {
            "feature_names": preprocessor.get_feature_names(),
            "class_mapping": preprocessor.get_class_mapping(),
        },
        meta_path,
    )
    return np.load(split_path), joblib.load(meta_path)


def train_models(X_train, y_train, class_weight_dict):
    models = {}

    lgb_params = dict(
        random_state=42,
        objective="multiclass",
        num_class=len(CLASS_NAMES),
        n_estimators=400,
        learning_rate=0.03,
        max_depth=7,
        class_weight=class_weight_dict,
    )
    models["LightGBM"] = LGBMClassifier(**lgb_params)

    xgb_params = dict(
        random_state=42,
        objective="multi:softmax",
        num_class=len(CLASS_NAMES),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
    )
    models["XGBoost"] = XGBClassifier(**xgb_params)

    models["SVM"] = SVC(
        random_state=42,
        probability=True,
        kernel="rbf",
        C=8.0,
        gamma="scale",
        class_weight=class_weight_dict,
    )

    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    # Save classification report
    (CLASS_REPORT_DIR / f"{name}.txt").write_text(
        f"{name} Classification Report (leak-free split)\n" + "-" * 60 + "\n" + report
    )

    # Save confusion matrix csv
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(CONF_MATRIX_DIR / f"{name}_confusion_matrix.csv")

    # Save ROC curves
    y_bin = label_binarize(y_test, classes=range(len(CLASS_NAMES)))
    roc_data = []
    for idx, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        roc_auc = auc(fpr, tpr)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.to_csv(ROC_DIR / f"{name}_class_{class_name}_roc.csv", index=False)
        roc_data.append({"class": class_name, "auc": roc_auc})

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    split, meta = load_or_create_split()
    feature_names = meta.get("feature_names") if isinstance(meta, dict) else None
    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]

    if feature_names is not None:
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

    models = train_models(X_train, y_train, class_weight_dict)

    metrics = []
    for name, model in models.items():
        metrics.append(evaluate_model(name, model, X_test, y_test))
        joblib.dump(model, ROOT / "models" / "saved" / f"{name.lower()}_model.joblib")

    summary_path = EVAL_DIR / "model_metrics_summary_traditional.csv"
    pd.DataFrame(metrics).to_csv(summary_path, index=False)
    print(f"Saved traditional summary to {summary_path}")

    (EVAL_DIR / "traditional_metrics_latest.json").write_text(
        json.dumps(metrics, indent=2)
    )


if __name__ == "__main__":
    main()
