from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))

from data_preprocessing import DataPreprocessor  # type: ignore
from training_runtime import PauseRequested, StopRequested, TrainingRunController  # type: ignore


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "saved"
RUNS_DIR = MODEL_DIR / "training_runs"
EVAL_DIR = ROOT / "evaluation_results"
MODEL_METRICS_DIR = EVAL_DIR / "model_metrics"
CLASS_NAMES_DEFAULT = ["HC", "PD", "SWEDD", "PRODROMAL"]
TRADITIONAL_MODELS = ("lightgbm", "xgboost", "svm")
TRANSFORMER_MODELS = ("pubmedbert", "biogpt", "clinical_t5")
ALL_BASE_MODELS = TRADITIONAL_MODELS + TRANSFORMER_MODELS
TRANSFORMER_SELECTION_METRIC = "val_f1"
DEFAULT_TRANSFORMER_LOSS = "focal"
DEFAULT_FOCAL_GAMMA = 1.5


TRADITIONAL_SEARCH_SPACES: Dict[str, List[Dict[str, Any]]] = {
    "lightgbm": [
        {"n_estimators": 350, "learning_rate": 0.03, "max_depth": -1, "num_leaves": 31, "min_child_samples": 20, "subsample": 0.90, "colsample_bytree": 0.90},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": -1, "num_leaves": 63, "min_child_samples": 25, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 450, "learning_rate": 0.025, "max_depth": 10, "num_leaves": 47, "min_child_samples": 15, "subsample": 0.90, "colsample_bytree": 0.80},
        {"n_estimators": 650, "learning_rate": 0.015, "max_depth": 12, "num_leaves": 95, "min_child_samples": 30, "subsample": 0.80, "colsample_bytree": 0.80},
        {"n_estimators": 800, "learning_rate": 0.012, "max_depth": -1, "num_leaves": 127, "min_child_samples": 20, "subsample": 0.85, "colsample_bytree": 0.80},
        {"n_estimators": 420, "learning_rate": 0.03, "max_depth": 8, "num_leaves": 63, "min_child_samples": 10, "subsample": 0.95, "colsample_bytree": 0.85},
    ],
    "xgboost": [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 2, "subsample": 0.90, "colsample_bytree": 0.90, "gamma": 0.0, "reg_lambda": 1.0},
        {"n_estimators": 450, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 1, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.05, "reg_lambda": 1.0},
        {"n_estimators": 500, "learning_rate": 0.025, "max_depth": 7, "min_child_weight": 3, "subsample": 0.80, "colsample_bytree": 0.80, "gamma": 0.10, "reg_lambda": 1.5},
        {"n_estimators": 350, "learning_rate": 0.04, "max_depth": 4, "min_child_weight": 1, "subsample": 0.95, "colsample_bytree": 0.90, "gamma": 0.0, "reg_lambda": 0.8},
        {"n_estimators": 650, "learning_rate": 0.015, "max_depth": 8, "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.05, "reg_lambda": 1.2},
        {"n_estimators": 520, "learning_rate": 0.02, "max_depth": 6, "min_child_weight": 4, "subsample": 0.90, "colsample_bytree": 0.80, "gamma": 0.08, "reg_lambda": 1.8},
    ],
    "svm": [
        {"C": 6.0, "gamma": "scale", "kernel": "rbf"},
        {"C": 8.0, "gamma": "scale", "kernel": "rbf"},
        {"C": 10.0, "gamma": 0.01, "kernel": "rbf"},
        {"C": 12.0, "gamma": 0.005, "kernel": "rbf"},
        {"C": 14.0, "gamma": "scale", "kernel": "rbf"},
        {"C": 16.0, "gamma": 0.003, "kernel": "rbf"},
    ],
}


TRANSFORMER_TRIALS: Dict[str, List[Dict[str, Any]]] = {
    "pubmedbert": [
        {"model_kwargs": {"dropout": 0.10, "freeze_bert": False, "train_encoder_layers": 8}, "optimizer": {"lr": 8.0e-6, "weight_decay": 0.02}, "grad_accum": 2},
        {"model_kwargs": {"dropout": 0.08, "freeze_bert": False, "train_encoder_layers": 6}, "optimizer": {"lr": 1.0e-5, "weight_decay": 0.02}, "grad_accum": 2},
        {"model_kwargs": {"dropout": 0.12, "freeze_bert": False, "train_encoder_layers": 4}, "optimizer": {"lr": 1.5e-5, "weight_decay": 0.01}, "grad_accum": 2},
        {"model_kwargs": {"dropout": 0.18, "freeze_bert": True}, "optimizer": {"lr": 2.5e-5, "weight_decay": 0.01}, "grad_accum": 2},
        {"model_kwargs": {"dropout": 0.06, "freeze_bert": False, "train_encoder_layers": 10}, "optimizer": {"lr": 6.0e-6, "weight_decay": 0.03}, "grad_accum": 2},
        {"model_kwargs": {"dropout": 0.10, "freeze_bert": False, "train_encoder_layers": 12}, "optimizer": {"lr": 5.0e-6, "weight_decay": 0.03}, "grad_accum": 1},
    ],
    "biogpt": [
        {"model_kwargs": {"dropout": 0.10, "train_decoder_layers": 4}, "optimizer": {"lr": 3.0e-5, "weight_decay": 0.01}, "grad_accum": 8},
        {"model_kwargs": {"dropout": 0.15, "train_decoder_layers": 6}, "optimizer": {"lr": 2.5e-5, "weight_decay": 0.02}, "grad_accum": 8},
        {"model_kwargs": {"dropout": 0.20, "train_decoder_layers": 8}, "optimizer": {"lr": 2.0e-5, "weight_decay": 0.02}, "grad_accum": 10},
        {"model_kwargs": {"dropout": 0.12, "train_decoder_layers": 10}, "optimizer": {"lr": 1.5e-5, "weight_decay": 0.02}, "grad_accum": 12},
        {"model_kwargs": {"dropout": 0.10, "train_decoder_layers": 12}, "optimizer": {"lr": 1.0e-5, "weight_decay": 0.02}, "grad_accum": 8},
        {"model_kwargs": {"dropout": 0.08, "train_decoder_layers": 10}, "optimizer": {"lr": 1.2e-5, "weight_decay": 0.015}, "grad_accum": 6},
    ],
    "clinical_t5": [
        {"model_kwargs": {"dropout": 0.10, "freeze_encoder": False}, "optimizer": {"lr": 2.0e-5, "weight_decay": 0.01}, "grad_accum": 8},
        {"model_kwargs": {"dropout": 0.15, "freeze_encoder": False}, "optimizer": {"lr": 1.5e-5, "weight_decay": 0.02}, "grad_accum": 8},
        {"model_kwargs": {"dropout": 0.20, "freeze_encoder": True}, "optimizer": {"lr": 2.5e-5, "weight_decay": 0.01}, "grad_accum": 6},
        {"model_kwargs": {"dropout": 0.08, "freeze_encoder": False}, "optimizer": {"lr": 1.0e-5, "weight_decay": 0.02}, "grad_accum": 10},
        {"model_kwargs": {"dropout": 0.12, "freeze_encoder": False}, "optimizer": {"lr": 8.0e-6, "weight_decay": 0.02}, "grad_accum": 8},
        {"model_kwargs": {"dropout": 0.10, "freeze_encoder": False}, "optimizer": {"lr": 1.2e-5, "weight_decay": 0.015}, "grad_accum": 6},
    ],
}


@dataclass
class TrainingBundle:
    X_train_dense: np.ndarray
    X_test_dense: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_groups: np.ndarray
    test_groups: np.ndarray
    feature_names: List[str]
    class_mapping: Dict[str, int]
    class_names: List[str]
    preprocessor: Any


@dataclass(frozen=True)
class GPUExecutionProfile:
    name: str
    train_batch_by_model: Dict[str, int]
    eval_batch_by_model: Dict[str, int]
    grad_accum_cap_by_model: Dict[str, int]
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    notes: str


class FocalLoss(torch.nn.Module):
    """Class-weighted focal loss for imbalanced multi-class classification."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = DEFAULT_FOCAL_GAMMA,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported focal-loss reduction: {reduction}")
        self.gamma = float(gamma)
        self.reduction = reduction
        if class_weights is not None:
            normalized = class_weights.float() / class_weights.float().mean().clamp_min(1e-6)
            self.register_buffer("class_weights", normalized)
        else:
            self.register_buffer("class_weights", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        ce_loss = -target_log_probs
        focal_factor = (1.0 - target_probs).pow(self.gamma)
        loss = focal_factor * ce_loss
        if self.class_weights is not None:
            loss = self.class_weights[targets] * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _as_dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _candidate_file_paths() -> List[Path]:
    return [
        ROOT / "PPMI_Curated_Data_Cut_Public_20240129.csv",
        ROOT / "PPMI_Curated_Data_Cut_Public_20241211.csv",
        ROOT / "PPMI_Curated_Data_Cut_Public_20250321.csv",
        ROOT / "PPMI_Curated_Data_Cut_Public_20250714.csv",
    ]


def _prepare_training_bundle() -> TrainingBundle:
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        _candidate_file_paths(),
        test_size=0.2,
        use_patient_split=True,
    )
    train_df, test_df = preprocessor.get_split_frames()
    feature_names = preprocessor.get_feature_names()
    class_mapping = preprocessor.get_class_mapping()
    class_names = [None] * len(class_mapping)
    for label, idx in class_mapping.items():
        class_names[idx] = label

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        EVAL_DIR / "leak_free_split.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    joblib.dump(
        {
            "feature_names": feature_names,
            "class_mapping": class_mapping,
            "train_groups": train_df["PATNO"].to_numpy(),
            "test_groups": test_df["PATNO"].to_numpy(),
        },
        EVAL_DIR / "leak_free_split_meta.joblib",
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor.get_preprocessor(), MODEL_DIR / "traditional_preprocessor.joblib")
    (MODEL_DIR / "traditional_class_mapping.json").write_text(
        json.dumps(_json_ready(class_mapping), indent=2),
        encoding="utf-8",
    )

    return TrainingBundle(
        X_train_dense=_as_dense(X_train).astype(np.float32),
        X_test_dense=_as_dense(X_test).astype(np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        train_groups=train_df["PATNO"].to_numpy(),
        test_groups=test_df["PATNO"].to_numpy(),
        feature_names=list(feature_names),
        class_mapping=dict(class_mapping),
        class_names=[
            str(name if name is not None else CLASS_NAMES_DEFAULT[idx])
            for idx, name in enumerate(class_names)
        ],
        preprocessor=preprocessor.get_preprocessor(),
    )


def _compute_class_weight_dict(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def _build_transformer_criterion(
    loss_name: str,
    class_weights_tensor: torch.Tensor,
    focal_gamma: float,
) -> Tuple[torch.nn.Module, str]:
    normalized_name = (loss_name or DEFAULT_TRANSFORMER_LOSS).strip().lower()
    if normalized_name in {"ce", "cross_entropy", "cross-entropy"}:
        return torch.nn.CrossEntropyLoss(weight=class_weights_tensor), "cross_entropy"
    if normalized_name == "focal":
        return FocalLoss(class_weights=class_weights_tensor, gamma=focal_gamma), f"focal(gamma={focal_gamma:.2f})"
    raise ValueError(f"Unsupported transformer loss: {loss_name}")


def _is_better_validation_epoch(
    val_f1: float,
    val_loss: float,
    best_val_f1: float,
    best_val_loss: float,
    min_delta: float = 1e-4,
) -> bool:
    if val_f1 > (best_val_f1 + min_delta):
        return True
    if abs(val_f1 - best_val_f1) <= min_delta and val_loss < (best_val_loss - min_delta):
        return True
    return False


def _evaluate_predictions(
    model_name: str,
    model_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Optional[np.ndarray],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    normalized_class_names = [str(name) for name in class_names]
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=normalized_class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    auroc = None
    if probabilities is not None:
        try:
            auroc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted")
        except ValueError:
            auroc = None
    return {
        "Model": model_name,
        "Type": model_type,
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Score": float(f1),
        "AUROC": None if auroc is None else float(auroc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def _traditional_model_builder(model_name: str, params: Dict[str, Any], class_weight_dict: Dict[int, float], num_classes: int):
    if model_name == "lightgbm":
        return LGBMClassifier(random_state=42, objective="multiclass", num_class=num_classes, verbose=-1, class_weight=class_weight_dict, **params)
    if model_name == "xgboost":
        return XGBClassifier(random_state=42, objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss", verbosity=0, **params)
    if model_name == "svm":
        return SVC(random_state=42, probability=True, decision_function_shape="ovr", class_weight=class_weight_dict, cache_size=2048, **params)
    raise ValueError(f"Unsupported traditional model: {model_name}")


def _run_grouped_traditional_search(
    model_name: str,
    search_space: List[Dict[str, Any]],
    bundle: TrainingBundle,
    controller: TrainingRunController,
    max_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    class_weight_dict = _compute_class_weight_dict(bundle.y_train)
    checkpoint_state = controller.load_checkpoint_state(
        model_name,
        {
            "phase": "search",
            "trials": {},
            "best_trial_index": None,
            "best_score": None,
        },
    )

    search_space = list(search_space[:max_trials])
    n_splits = max(2, min(3, len(np.unique(bundle.train_groups))))
    splitter = GroupKFold(n_splits=n_splits)

    controller.mark_running("traditional_search", model_name=model_name)
    controller.update_model_state(
        model_name,
        status="running",
        family="traditional",
        trials_total=len(search_space),
    )

    for trial_index, params in enumerate(search_space):
        trial_key = str(trial_index)
        trial_state = checkpoint_state["trials"].setdefault(
            trial_key,
            {
                "params": params,
                "fold_scores": [],
                "status": "pending",
            },
        )
        if trial_state.get("status") == "completed":
            continue

        controller.mark_running("traditional_trial", model_name=model_name, extra={"trial_index": trial_index})

        for fold_index, (train_idx, val_idx) in enumerate(
            splitter.split(bundle.X_train_dense, bundle.y_train, groups=bundle.train_groups)
        ):
            if fold_index < len(trial_state["fold_scores"]):
                continue

            train_features = np.asarray(bundle.X_train_dense[train_idx], dtype=np.float32)
            val_features = np.asarray(bundle.X_train_dense[val_idx], dtype=np.float32)
            train_targets = np.asarray(bundle.y_train[train_idx], dtype=np.int64)
            val_targets = np.asarray(bundle.y_train[val_idx], dtype=np.int64)

            estimator = _traditional_model_builder(
                model_name,
                params,
                class_weight_dict=class_weight_dict,
                num_classes=len(bundle.class_names),
            )
            estimator.fit(train_features, train_targets)
            preds = estimator.predict(val_features)
            score = f1_score(val_targets, preds, average="weighted")

            trial_state["fold_scores"].append(float(score))
            trial_state["status"] = "running"
            checkpoint_state["phase"] = "search"
            controller.save_checkpoint_state(model_name, _json_ready(checkpoint_state))
            controller.update_model_state(
                model_name,
                current_trial=trial_index,
                current_fold=fold_index,
                latest_fold_score=float(score),
            )
            controller.raise_if_requested()

        trial_state["mean_score"] = float(np.mean(trial_state["fold_scores"]))
        trial_state["status"] = "completed"
        controller.append_trial_result(
            model_name,
            {
                "trial_index": trial_index,
                "params": params,
                "mean_score": trial_state["mean_score"],
            },
        )
        if checkpoint_state.get("best_score") is None or trial_state["mean_score"] > checkpoint_state["best_score"]:
            checkpoint_state["best_score"] = float(trial_state["mean_score"])
            checkpoint_state["best_trial_index"] = trial_index
        controller.save_checkpoint_state(model_name, _json_ready(checkpoint_state))
        controller.raise_if_requested()

    if checkpoint_state.get("best_trial_index") is None:
        raise RuntimeError(f"No completed trials were recorded for {model_name}")

    best_index = int(checkpoint_state["best_trial_index"])
    best_params = dict(search_space[best_index])

    checkpoint_state["phase"] = "fit_final"
    controller.save_checkpoint_state(model_name, _json_ready(checkpoint_state))
    controller.mark_running("traditional_fit_final", model_name=model_name)

    best_model = _traditional_model_builder(
        model_name,
        best_params,
        class_weight_dict=class_weight_dict,
        num_classes=len(bundle.class_names),
    )
    best_model.fit(
        np.asarray(bundle.X_train_dense, dtype=np.float32),
        np.asarray(bundle.y_train, dtype=np.int64),
    )

    artifact_path = MODEL_DIR / f"{model_name}_model.joblib"
    joblib.dump(best_model, artifact_path)

    test_features = np.asarray(bundle.X_test_dense, dtype=np.float32)
    y_pred = best_model.predict(test_features)
    probabilities = best_model.predict_proba(test_features) if hasattr(best_model, "predict_proba") else None
    metrics = _evaluate_predictions(
        model_name=model_name.replace("_", " ").title().replace("Svm", "SVM").replace("Lightgbm", "LightGBM").replace("Xgboost", "XGBoost"),
        model_type="Traditional ML",
        y_true=bundle.y_test,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=bundle.class_names,
    )

    checkpoint_state["phase"] = "complete"
    checkpoint_state["artifact_path"] = str(artifact_path)
    checkpoint_state["best_params"] = best_params
    checkpoint_state["metrics"] = {
        key: value for key, value in metrics.items()
        if key in {"Accuracy", "Precision", "Recall", "F1_Score", "AUROC", "Model", "Type"}
    }
    controller.save_checkpoint_state(model_name, _json_ready(checkpoint_state))
    controller.update_model_state(
        model_name,
        status="completed",
        artifact_path=str(artifact_path),
        best_params=best_params,
        best_cv_score=float(checkpoint_state["best_score"]),
        metrics=checkpoint_state["metrics"],
    )
    return metrics, best_params, artifact_path


def _build_rag_contexts(features: np.ndarray, feature_names: Sequence[str], doc_manager: Any) -> List[str]:
    contexts: List[str] = []
    for row in features:
        feature_desc = {name: float(val) for name, val in zip(feature_names, row)}
        query_parts = []
        for symptom_key, col in {
            "tremor": "sym_tremor",
            "rigidity": "sym_rigid",
            "bradykinesia": "sym_brady",
            "postural instability": "sym_posins",
        }.items():
            severity = feature_desc.get(col, 0)
            if severity > 0:
                query_parts.append(f"{symptom_key} severity:{severity:.2f}")
        if feature_desc.get("moca", 30) < 26:
            query_parts.append("cognitive impairment")
        if feature_desc.get("fampd", 0) > 0:
            query_parts.append("family history Parkinson's disease")
        query = "Parkinson's disease " + " ".join(query_parts)
        passages = doc_manager.extract_relevant_passages(query, top_k=2)
        if not passages:
            contexts.append("")
            continue
        contexts.append(
            " ".join(
                f"From '{item.get('doc_title') or item.get('doc_id') or 'document'}' {item['text'][:240]}..."
                for item in passages
            )
        )
    return contexts


def _encode_contexts_for_model(model_name: str, model: Any, contexts: Optional[List[str]]) -> Optional[Dict[str, torch.Tensor]]:
    if not contexts:
        return None
    if model_name == "clinical_t5":
        texts = [f"classify patient: {text}" for text in contexts]
    else:
        texts = contexts
    encoded = model.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    return {key: value.cpu() for key, value in encoded.items()}


def _prepare_batch(batch: Any, device: torch.device):
    if len(batch) == 3:
        features, targets, contexts = batch
        if isinstance(contexts, dict):
            contexts = {key: value.to(device, non_blocking=True) for key, value in contexts.items()}
        else:
            contexts = list(contexts)
    else:
        features, targets = batch
        contexts = None
    return features.to(device), targets.to(device), contexts


def _create_transformer_model(model_name: str, input_dim: int, num_classes: int, model_kwargs: Dict[str, Any]):
    from models.medical_transformers import (  # type: ignore
        BioMistralClassifier as BioGPTForTabular,
        ClinicalT5Classifier as ClinicalT5ForTabular,
        PubMedBERTClassifier as PubMedBERTForTabular,
    )

    if model_name == "pubmedbert":
        return PubMedBERTForTabular(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    if model_name == "biogpt":
        return BioGPTForTabular(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    if model_name == "clinical_t5":
        return ClinicalT5ForTabular(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    raise ValueError(f"Unsupported transformer model: {model_name}")


def _torch_checkpoint_path(controller: TrainingRunController, model_name: str, trial_index: int) -> Path:
    return controller.paths.checkpoints_dir / f"{model_name}_trial{trial_index}.pth"


def _canonical_transformer_artifacts(model_name: str) -> List[Path]:
    aliases = {
        "pubmedbert": ["pubmedbert_transformer.pth", "pubmedbert.pth"],
        "biogpt": ["biogpt_transformer.pth", "biogpt.pth", "biomistral.pth"],
        "clinical_t5": ["clinical_t5_transformer.pth", "clinicalt5_transformer.pth", "clinical_t5.pth"],
    }
    return [MODEL_DIR / filename for filename in aliases[model_name]]


def _run_transformer_search(
    model_name: str,
    trial_space: List[Dict[str, Any]],
    bundle: TrainingBundle,
    controller: TrainingRunController,
    max_trials: int,
    num_epochs: int,
    patience: int,
    use_rag: bool,
    gpu_profile_name: str,
    transformer_loss_name: str,
    focal_gamma: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    from models.transformer_models import TabularDataset  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_profile = _detect_gpu_execution_profile(requested_profile=gpu_profile_name)
    train_bs = gpu_profile.train_batch_by_model.get(model_name, 8)
    eval_bs = gpu_profile.eval_batch_by_model.get(model_name, max(train_bs * 2, 8))
    loader_kwargs = _build_loader_kwargs(device, gpu_profile)

    class_weights = compute_class_weight("balanced", classes=np.unique(bundle.y_train), y=bundle.y_train)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion, criterion_label = _build_transformer_criterion(
        transformer_loss_name,
        class_weights_tensor,
        focal_gamma,
    )

    split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_index, val_index = next(split.split(bundle.X_train_dense, bundle.y_train, groups=bundle.train_groups))

    train_features = bundle.X_train_dense[train_index]
    train_targets = bundle.y_train[train_index]
    val_features = bundle.X_train_dense[val_index]
    val_targets = bundle.y_train[val_index]

    train_contexts = None
    val_contexts = None
    if use_rag:
        from document_manager import DocumentManager  # type: ignore

        doc_manager = DocumentManager(docs_dir=str(ROOT / "medical_docs"))
        train_contexts = _build_rag_contexts(train_features, bundle.feature_names, doc_manager)
        val_contexts = _build_rag_contexts(val_features, bundle.feature_names, doc_manager)

    test_ds = TabularDataset(bundle.X_test_dense, bundle.y_test, bundle.feature_names, contexts=None)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, **loader_kwargs)

    checkpoint_state = controller.load_checkpoint_state(
        model_name,
        {
            "phase": "search",
            "trials": {},
            "best_trial_index": None,
            "best_f1": None,
            "selection_metric": TRANSFORMER_SELECTION_METRIC,
            "loss_name": criterion_label,
            "focal_gamma": float(focal_gamma),
        },
    )
    if (
        checkpoint_state.get("selection_metric") != TRANSFORMER_SELECTION_METRIC
        or checkpoint_state.get("loss_name") != criterion_label
        or float(checkpoint_state.get("focal_gamma", focal_gamma)) != float(focal_gamma)
    ):
        print(
            f"[TRANSFORMER] {model_name} resetting saved trial state to match "
            f"selection={TRANSFORMER_SELECTION_METRIC} and loss={criterion_label}"
        )
        checkpoint_state = {
            "phase": "search",
            "trials": {},
            "best_trial_index": None,
            "best_f1": None,
            "selection_metric": TRANSFORMER_SELECTION_METRIC,
            "loss_name": criterion_label,
            "focal_gamma": float(focal_gamma),
        }
        for stale_checkpoint in controller.paths.checkpoints_dir.glob(f"{model_name}_trial*.pth"):
            stale_checkpoint.unlink(missing_ok=True)
    trial_space = list(trial_space[:max_trials])

    controller.mark_running("transformer_search", model_name=model_name)
    controller.update_model_state(
        model_name,
        status="running",
        family="transformer",
        trials_total=len(trial_space),
        device=str(device),
        gpu_profile=gpu_profile.name,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        num_workers=loader_kwargs.get("num_workers", 0),
        gpu_notes=gpu_profile.notes,
        selection_metric=TRANSFORMER_SELECTION_METRIC,
        transformer_loss=criterion_label,
    )

    for trial_index, trial_cfg in enumerate(trial_space):
        trial_key = str(trial_index)
        trial_state = checkpoint_state["trials"].setdefault(
            trial_key,
            {
                "config": trial_cfg,
                "status": "pending",
                "history": {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": [], "lr": []},
                "best_val_loss": None,
                "best_val_f1": None,
                "best_epoch": None,
                "selection_metric": TRANSFORMER_SELECTION_METRIC,
                "loss_name": criterion_label,
            },
        )
        if trial_state.get("status") == "completed":
            continue

        controller.mark_running("transformer_trial", model_name=model_name, extra={"trial_index": trial_index})

        model = _create_transformer_model(
            model_name,
            input_dim=bundle.X_train_dense.shape[1],
            num_classes=len(bundle.class_names),
            model_kwargs=trial_cfg["model_kwargs"],
        ).to(device)
        encoded_train_contexts = _encode_contexts_for_model(model_name, model, train_contexts)
        encoded_val_contexts = _encode_contexts_for_model(model_name, model, val_contexts)
        train_ds = TabularDataset(train_features, train_targets, bundle.feature_names, contexts=encoded_train_contexts)
        val_ds = TabularDataset(val_features, val_targets, bundle.feature_names, contexts=encoded_val_contexts)
        train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, **loader_kwargs)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), **trial_cfg["optimizer"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7)
        scaler = torch.amp.GradScaler(device=device.type, enabled=device.type == "cuda")
        grad_accum = min(
            int(trial_cfg.get("grad_accum", 8)),
            int(gpu_profile.grad_accum_cap_by_model.get(model_name, trial_cfg.get("grad_accum", 8))),
        )
        print(
            f"[TRANSFORMER] {model_name} trial {trial_index + 1}/{len(trial_space)} "
            f"| train_bs={train_bs} eval_bs={eval_bs} grad_accum={grad_accum} "
            f"effective_batch={train_bs * grad_accum} "
            f"train_steps={len(train_loader)} val_steps={len(val_loader)} "
            f"| selection={TRANSFORMER_SELECTION_METRIC} loss={criterion_label}"
        )
        start_epoch = int(trial_state.get("next_epoch", 0))
        early_stop_counter = int(trial_state.get("early_stop_counter", 0))
        best_val_loss = float(trial_state["best_val_loss"]) if trial_state.get("best_val_loss") is not None else float("inf")
        best_val_f1 = float(trial_state["best_val_f1"]) if trial_state.get("best_val_f1") is not None else float("-inf")
        best_model_state_dict = None
        torch_ckpt_path = _torch_checkpoint_path(controller, model_name, trial_index)

        if torch_ckpt_path.exists():
            saved = torch.load(torch_ckpt_path, map_location=device, weights_only=False)
            if (
                saved.get("selection_metric") == TRANSFORMER_SELECTION_METRIC
                and saved.get("loss_name") == criterion_label
                and float(saved.get("focal_gamma", focal_gamma)) == float(focal_gamma)
            ):
                model.load_state_dict(saved["model_state_dict"])
                optimizer.load_state_dict(saved["optimizer_state_dict"])
                scheduler.load_state_dict(saved["scheduler_state_dict"])
                if saved.get("scaler_state_dict") and device.type == "cuda":
                    scaler.load_state_dict(saved["scaler_state_dict"])
                start_epoch = int(saved.get("epoch", -1)) + 1
                early_stop_counter = int(saved.get("early_stop_counter", early_stop_counter))
                best_val_loss = float(saved.get("best_val_loss", best_val_loss))
                best_val_f1 = float(saved.get("best_val_f1", best_val_f1))
                best_model_state_dict = saved.get("best_model_state_dict")
                trial_state["history"] = saved.get("history", trial_state["history"])
            else:
                print(
                    f"[TRANSFORMER] {model_name} trial {trial_index + 1} ignoring stale "
                    f"checkpoint with incompatible selection/loss settings"
                )
                start_epoch = 0
                early_stop_counter = 0
                best_val_loss = float("inf")
                best_val_f1 = float("-inf")
                best_model_state_dict = None
                trial_state["history"] = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": [], "lr": []}
                torch_ckpt_path.unlink(missing_ok=True)

        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_train_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            progress = tqdm(
                train_loader,
                total=len(train_loader),
                dynamic_ncols=True,
                leave=False,
                desc=f"{model_name} t{trial_index + 1}/{len(trial_space)} e{epoch + 1}/{num_epochs}",
            )
            for batch_index, batch in enumerate(progress):
                features, targets, contexts = _prepare_batch(batch, device)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    outputs = model(features, contexts)
                    loss = criterion(outputs, targets) / grad_accum
                scaler.scale(loss).backward()

                if (batch_index + 1) % grad_accum == 0 or (batch_index + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_train_loss += float(loss.item() * grad_accum)
                current_loss = float(loss.item() * grad_accum)
                current_lr = float(optimizer.param_groups[0]["lr"])
                if device.type == "cuda":
                    gpu_mem_gb = torch.cuda.memory_allocated(device) / 1024**3
                    progress.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}", gpu_gb=f"{gpu_mem_gb:.2f}")
                else:
                    progress.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}")

            progress.close()

            avg_train_loss = running_train_loss / max(len(train_loader), 1)

            model.eval()
            running_val_loss = 0.0
            all_preds: List[int] = []
            all_targets: List[int] = []
            with torch.no_grad():
                for batch in val_loader:
                    features, targets, contexts = _prepare_batch(batch, device)
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        outputs = model(features, contexts)
                        val_loss = criterion(outputs, targets)
                    running_val_loss += float(val_loss.item())
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_targets.extend(targets.cpu().numpy().tolist())

            avg_val_loss = running_val_loss / max(len(val_loader), 1)
            val_acc = float(np.mean(np.asarray(all_preds) == np.asarray(all_targets)))
            val_f1 = float(f1_score(all_targets, all_preds, average="weighted"))
            scheduler.step(val_f1)

            history = trial_state["history"]
            history["train_loss"].append(float(avg_train_loss))
            history["val_loss"].append(float(avg_val_loss))
            history["val_f1"].append(val_f1)
            history["val_acc"].append(val_acc)
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))
            gpu_peak_gb = 0.0
            if device.type == "cuda":
                gpu_peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            print(
                f"[TRANSFORMER] {model_name} trial {trial_index + 1}/{len(trial_space)} "
                f"epoch {epoch + 1}/{num_epochs} "
                f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
                f"val_f1={val_f1:.4f} val_acc={val_acc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"gpu_peak_gb={gpu_peak_gb:.2f}"
            )

            if _is_better_validation_epoch(
                val_f1=val_f1,
                val_loss=avg_val_loss,
                best_val_f1=best_val_f1,
                best_val_loss=best_val_loss,
            ):
                best_val_loss = avg_val_loss
                best_val_f1 = val_f1
                trial_state["best_val_loss"] = float(best_val_loss)
                trial_state["best_val_f1"] = float(best_val_f1)
                trial_state["best_epoch"] = epoch
                best_model_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            trial_state["status"] = "running"
            trial_state["next_epoch"] = epoch + 1
            trial_state["early_stop_counter"] = early_stop_counter
            checkpoint_state["phase"] = "search"
            controller.save_checkpoint_state(model_name, _json_ready(checkpoint_state))
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else None,
                    "history": history,
                    "early_stop_counter": early_stop_counter,
                    "best_val_loss": best_val_loss,
                    "best_val_f1": best_val_f1,
                    "best_model_state_dict": best_model_state_dict,
                    "selection_metric": TRANSFORMER_SELECTION_METRIC,
                    "loss_name": criterion_label,
                    "focal_gamma": float(focal_gamma),
                },
                torch_ckpt_path,
            )
            controller.update_model_state(
                model_name,
                current_trial=trial_index,
                current_epoch=epoch,
                grad_accum=grad_accum,
                best_val_f1=float(best_val_f1),
                best_val_loss=float(best_val_loss),
                selection_metric=TRANSFORMER_SELECTION_METRIC,
                transformer_loss=criterion_label,
            )
            controller.raise_if_requested()

            if early_stop_counter >= patience:
                print(
                    f"[TRANSFORMER] {model_name} trial {trial_index + 1}/{len(trial_space)} "
                    f"early-stopped after epoch {epoch + 1}"
                )
                break

        trial_state["status"] = "completed"
        controller.append_trial_result(
            model_name,
            {
                "trial_index": trial_index,
                "config": trial_cfg,
                "best_val_f1": float(trial_state.get("best_val_f1") or 0.0),
                "best_epoch": trial_state.get("best_epoch"),
            },
        )
        best_so_far = checkpoint_state.get("best_f1")
        if best_so_far is None or float(trial_state.get("best_val_f1") or -1.0) > float(best_so_far):
            checkpoint_state["best_f1"] = float(trial_state["best_val_f1"])
            checkpoint_state["best_trial_index"] = trial_index
        controller.save_checkpoint_state(model_name, _json_ready(checkpoint_state))
        controller.raise_if_requested()

    if checkpoint_state.get("best_trial_index") is None:
        raise RuntimeError(f"No completed transformer trial found for {model_name}")

    best_trial_index = int(checkpoint_state["best_trial_index"])
    best_trial_cfg = dict(trial_space[best_trial_index])
    checkpoint_state["phase"] = "evaluate"
    controller.save_checkpoint_state(model_name, _json_ready(checkpoint_state))

    best_model = _create_transformer_model(
        model_name,
        input_dim=bundle.X_train_dense.shape[1],
        num_classes=len(bundle.class_names),
        model_kwargs=best_trial_cfg["model_kwargs"],
    ).to(device)
    best_state = torch.load(_torch_checkpoint_path(controller, model_name, best_trial_index), map_location=device, weights_only=False)
    best_model.load_state_dict(best_state.get("best_model_state_dict") or best_state["model_state_dict"])
    best_model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probabilities: List[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            features, targets, contexts = _prepare_batch(batch, device)
            outputs = best_model(features, contexts)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probabilities.append(probs.cpu().numpy())

    probabilities = np.vstack(all_probabilities) if all_probabilities else None
    metrics = _evaluate_predictions(
        model_name={"pubmedbert": "PubMedBERT", "biogpt": "BioGPT", "clinical_t5": "Clinical-T5"}[model_name],
        model_type="Transformer",
        y_true=np.asarray(all_targets),
        y_pred=np.asarray(all_preds),
        probabilities=probabilities,
        class_names=bundle.class_names,
    )

    primary_artifact = _canonical_transformer_artifacts(model_name)[0]
    for path in _canonical_transformer_artifacts(model_name):
        torch.save(best_model.state_dict(), path)

    checkpoint_state["phase"] = "complete"
    checkpoint_state["artifact_path"] = str(primary_artifact)
    checkpoint_state["best_trial_index"] = best_trial_index
    checkpoint_state["best_config"] = best_trial_cfg
    checkpoint_state["metrics"] = {
        key: value for key, value in metrics.items()
        if key in {"Accuracy", "Precision", "Recall", "F1_Score", "AUROC", "Model", "Type"}
    }
    controller.save_checkpoint_state(model_name, _json_ready(checkpoint_state))
    controller.update_model_state(
        model_name,
        status="completed",
        artifact_path=str(primary_artifact),
        best_trial_index=best_trial_index,
        best_config=best_trial_cfg,
        metrics=checkpoint_state["metrics"],
    )
    return metrics, best_trial_cfg, primary_artifact


def _train_ensemble(bundle: TrainingBundle, controller: TrainingRunController) -> Optional[Dict[str, Any]]:
    from models.multimodal_ml import MultimodalEnsemble  # type: ignore

    controller.mark_running("ensemble_training", model_name="ensemble")
    controller.raise_if_requested()

    ensemble = MultimodalEnsemble()
    ensemble.load_traditional_models(model_dir=str(MODEL_DIR))
    ensemble.load_transformer_models(model_dir=str(MODEL_DIR), input_dim=bundle.X_train_dense.shape[1], num_classes=len(bundle.class_names))
    ensemble.train_ensemble(bundle.X_train_dense, bundle.y_train, ensemble_type="stacking")
    ensemble.save_ensemble(str(MODEL_DIR / "multimodal_ensemble.joblib"))
    results = ensemble.evaluate_ensemble(bundle.X_test_dense, bundle.y_test)

    probabilities = results.get("probabilities")
    metrics = _evaluate_predictions(
        model_name="Multimodal Ensemble",
        model_type="Ensemble",
        y_true=bundle.y_test,
        y_pred=np.asarray(results["predictions"]),
        probabilities=np.asarray(probabilities) if probabilities is not None else None,
        class_names=bundle.class_names,
    )
    controller.update_model_state(
        "ensemble",
        status="completed",
        artifact_path=str(MODEL_DIR / "multimodal_ensemble.joblib"),
        metrics={key: value for key, value in metrics.items() if key in {"Accuracy", "Precision", "Recall", "F1_Score", "AUROC", "Model", "Type"}},
    )
    return metrics


def _write_metric_outputs(base_metrics: List[Dict[str, Any]], ensemble_metrics: Optional[Dict[str, Any]]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    traditional_rows = []
    transformer_rows = []
    for metrics in base_metrics:
        summary_rows.append({
            "Model": metrics["Model"],
            "Type": metrics["Type"],
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1_Score": metrics["F1_Score"],
            "AUROC": metrics["AUROC"] if metrics["AUROC"] is not None else "",
        })
        if "traditional" in metrics["Type"].lower():
            traditional_rows.append(metrics)
        if "transformer" in metrics["Type"].lower():
            transformer_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(EVAL_DIR / "summary_metrics.csv", index=False)
    summary_df.to_csv(MODEL_METRICS_DIR / "model_metrics_summary.csv", index=False)

    (EVAL_DIR / "traditional_metrics_latest.json").write_text(json.dumps(_json_ready(traditional_rows), indent=2), encoding="utf-8")
    (EVAL_DIR / "transformer_metrics_latest.json").write_text(json.dumps(_json_ready(transformer_rows), indent=2), encoding="utf-8")
    if ensemble_metrics is not None:
        (EVAL_DIR / "ensemble_metrics_latest.json").write_text(json.dumps(_json_ready(ensemble_metrics), indent=2), encoding="utf-8")


def _write_run_manifest_snapshot(controller: TrainingRunController, base_metrics: List[Dict[str, Any]], ensemble_metrics: Optional[Dict[str, Any]]) -> None:
    controller.write_metrics_file(
        "final_metrics.json",
        {
            "base_metrics": _json_ready(base_metrics),
            "ensemble_metrics": _json_ready(ensemble_metrics),
        },
    )


def _parse_selected_models(raw: str) -> List[str]:
    raw = (raw or "all").strip().lower()
    if raw in {"all", "*"}:
        return list(ALL_BASE_MODELS)
    aliases = {
        "lgbm": "lightgbm",
        "lightgbm": "lightgbm",
        "xgb": "xgboost",
        "xgboost": "xgboost",
        "svm": "svm",
        "pubmed": "pubmedbert",
        "pubmedbert": "pubmedbert",
        "biogpt": "biogpt",
        "bio": "biogpt",
        "clinical": "clinical_t5",
        "clinical_t5": "clinical_t5",
        "t5": "clinical_t5",
    }
    selected: List[str] = []
    for piece in [part.strip() for part in raw.split(",") if part.strip()]:
        canonical = aliases.get(piece)
        if canonical and canonical not in selected:
            selected.append(canonical)
    if not selected:
        raise ValueError(f"No valid models were selected from '{raw}'")
    return selected


def _detect_gpu_execution_profile(
    requested_profile: str = "auto",
    cuda_available: Optional[bool] = None,
    device_name: Optional[str] = None,
    total_memory_gb: Optional[float] = None,
) -> GPUExecutionProfile:
    cuda_ready = torch.cuda.is_available() if cuda_available is None else cuda_available
    if not cuda_ready:
        return GPUExecutionProfile(
            name="cpu",
            train_batch_by_model={"pubmedbert": 4, "biogpt": 2, "clinical_t5": 2},
            eval_batch_by_model={"pubmedbert": 8, "biogpt": 4, "clinical_t5": 4},
            grad_accum_cap_by_model={"pubmedbert": 12, "biogpt": 16, "clinical_t5": 16},
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=False,
            notes="CPU fallback profile.",
        )

    name = (device_name or torch.cuda.get_device_name(0)).lower()
    memory_gb = float(total_memory_gb if total_memory_gb is not None else (torch.cuda.get_device_properties(0).total_memory / 1024**3))
    requested = (requested_profile or "auto").strip().lower()

    if requested in {"rtx-a4000", "a4000"} or (requested == "auto" and ("a4000" in name or memory_gb >= 15.0)):
        return GPUExecutionProfile(
            name="rtx-a4000",
            train_batch_by_model={"pubmedbert": 32, "biogpt": 12, "clinical_t5": 12},
            eval_batch_by_model={"pubmedbert": 64, "biogpt": 24, "clinical_t5": 24},
            grad_accum_cap_by_model={"pubmedbert": 2, "biogpt": 5, "clinical_t5": 5},
            num_workers=6,
            prefetch_factor=4,
            persistent_workers=True,
            notes="Optimized for RTX A4000 / ~16 GB VRAM with larger per-step batches.",
        )

    if requested in {"high-vram", "16gb"} or (requested == "auto" and memory_gb >= 11.0):
        return GPUExecutionProfile(
            name="high-vram",
            train_batch_by_model={"pubmedbert": 12, "biogpt": 8, "clinical_t5": 8},
            eval_batch_by_model={"pubmedbert": 32, "biogpt": 16, "clinical_t5": 16},
            grad_accum_cap_by_model={"pubmedbert": 6, "biogpt": 8, "clinical_t5": 8},
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            notes="Generic 12 GB+ CUDA profile.",
        )

    return GPUExecutionProfile(
        name="compat",
        train_batch_by_model={"pubmedbert": 8, "biogpt": 6, "clinical_t5": 6},
        eval_batch_by_model={"pubmedbert": 16, "biogpt": 8, "clinical_t5": 8},
        grad_accum_cap_by_model={"pubmedbert": 8, "biogpt": 10, "clinical_t5": 10},
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        notes="Compatibility profile for lower-VRAM GPUs.",
    )


def _build_loader_kwargs(device: torch.device, profile: GPUExecutionProfile) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": profile.num_workers if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = profile.persistent_workers
        kwargs["prefetch_factor"] = profile.prefetch_factor
    return kwargs


def _configure_transformer_runtime(selected_models: Sequence[str], allow_cpu_transformers: bool, cuda_available: Optional[bool] = None) -> str:
    cuda_ready = torch.cuda.is_available() if cuda_available is None else cuda_available
    transformer_requested = any(model_name in TRANSFORMER_MODELS for model_name in selected_models)
    if not transformer_requested:
        return "not-applicable"
    if not cuda_ready and not allow_cpu_transformers:
        raise RuntimeError(
            "Transformer retraining requires CUDA for this accuracy profile. "
            "Use a CUDA-enabled PyTorch install/GPU, or rerun with --allow-cpu-transformers if you explicitly accept slower, lower-throughput training."
        )
    if cuda_ready:
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        return "cuda"
    return "cpu"


def _run_training(args: argparse.Namespace) -> int:
    selected_models = _parse_selected_models(args.models)
    transformer_runtime = _configure_transformer_runtime(
        selected_models,
        allow_cpu_transformers=args.allow_cpu_transformers,
    )
    controller = TrainingRunController(RUNS_DIR, args.run_name)
    controller.initialize(
        selected_models=selected_models,
        config={
            "epochs": args.epochs,
            "patience": args.patience,
            "traditional_trials": args.traditional_trials,
            "transformer_trials": args.transformer_trials,
            "use_rag": args.use_rag,
            "train_ensemble": not args.skip_ensemble,
            "allow_cpu_transformers": args.allow_cpu_transformers,
            "transformer_runtime": transformer_runtime,
            "gpu_profile": args.gpu_profile,
            "transformer_loss": args.transformer_loss,
            "focal_gamma": args.focal_gamma,
        },
        resume=args.resume,
    )

    if args.dry_run:
        controller.mark_paused("dry-run initialized")
        print(json.dumps(controller.status_summary(), indent=2))
        return 0

    try:
        controller.clear_stop()
        if args.resume:
            controller.clear_pause()

        bundle = _prepare_training_bundle()
        base_metrics: List[Dict[str, Any]] = []

        for model_name in selected_models:
            if model_name in TRADITIONAL_MODELS:
                metrics, best_config, artifact_path = _run_grouped_traditional_search(
                    model_name=model_name,
                    search_space=TRADITIONAL_SEARCH_SPACES[model_name],
                    bundle=bundle,
                    controller=controller,
                    max_trials=args.traditional_trials,
                )
            else:
                metrics, best_config, artifact_path = _run_transformer_search(
                    model_name=model_name,
                    trial_space=TRANSFORMER_TRIALS[model_name],
                    bundle=bundle,
                    controller=controller,
                    max_trials=args.transformer_trials,
                    num_epochs=args.epochs,
                    patience=args.patience,
                    use_rag=args.use_rag,
                    gpu_profile_name=args.gpu_profile,
                    transformer_loss_name=args.transformer_loss,
                    focal_gamma=args.focal_gamma,
                )
            base_metrics.append(metrics)
            controller.update_model_state(
                model_name,
                best_config=_json_ready(best_config),
                artifact_path=str(artifact_path),
                metrics={key: value for key, value in metrics.items() if key in {"Accuracy", "Precision", "Recall", "F1_Score", "AUROC", "Model", "Type"}},
            )

        ensemble_metrics = None
        should_train_ensemble = (not args.skip_ensemble) and set(selected_models) == set(ALL_BASE_MODELS)
        if should_train_ensemble:
            ensemble_metrics = _train_ensemble(bundle, controller)
        elif not args.skip_ensemble:
            controller.update_model_state(
                "ensemble",
                status="skipped",
                reason="ensemble retraining is only automatic when all six base models are selected",
            )

        _write_metric_outputs(base_metrics, ensemble_metrics)
        _write_run_manifest_snapshot(controller, base_metrics, ensemble_metrics)
        controller.mark_completed()
        print(json.dumps(controller.status_summary(), indent=2))
        return 0
    except PauseRequested as exc:
        controller.mark_paused(str(exc))
        print(json.dumps(controller.status_summary(), indent=2))
        return 0
    except StopRequested as exc:
        controller.mark_stopped(str(exc))
        print(json.dumps(controller.status_summary(), indent=2))
        return 0
    except KeyboardInterrupt:
        controller.mark_paused("keyboard interrupt")
        print(json.dumps(controller.status_summary(), indent=2))
        return 1
    except Exception as exc:
        controller.mark_failed(str(exc))
        raise


def _run_status(args: argparse.Namespace) -> int:
    controller = TrainingRunController(RUNS_DIR, args.run_name)
    print(json.dumps(controller.status_summary(), indent=2))
    return 0


def _run_pause(args: argparse.Namespace) -> int:
    controller = TrainingRunController(RUNS_DIR, args.run_name)
    controller.request_pause()
    controller.mark_paused("pause requested from CLI")
    print(json.dumps(controller.status_summary(), indent=2))
    return 0


def _run_stop(args: argparse.Namespace) -> int:
    controller = TrainingRunController(RUNS_DIR, args.run_name)
    controller.request_stop()
    controller.mark_stopped("stop requested from CLI")
    print(json.dumps(controller.status_summary(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrain the six base Parkinson's models with resumable checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Start a new training run.")
    train_parser.add_argument("--run-name", default="full_retrain", help="Name of the training run.")
    train_parser.add_argument("--models", default="all", help="Comma-separated list of models or 'all'.")
    train_parser.add_argument("--epochs", type=int, default=30, help="Max epochs per transformer trial.")
    train_parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience for transformers.")
    train_parser.add_argument("--traditional-trials", type=int, default=6, help="Number of traditional-model trials per model.")
    train_parser.add_argument("--transformer-trials", type=int, default=6, help="Number of transformer trials per model.")
    train_parser.add_argument("--use-rag", dest="use_rag", action="store_true", help="Build RAG contexts for transformer training.")
    train_parser.add_argument("--no-rag", dest="use_rag", action="store_false", help="Disable RAG context enrichment for transformer training.")
    train_parser.add_argument("--gpu-profile", default="auto", choices=["auto", "rtx-a4000", "high-vram", "compat"], help="CUDA loader/batch preset for transformer training.")
    train_parser.add_argument("--transformer-loss", default=DEFAULT_TRANSFORMER_LOSS, choices=["cross_entropy", "focal"], help="Loss function for transformer fine-tuning.")
    train_parser.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA, help="Gamma value for focal loss when --transformer-loss focal is used.")
    train_parser.add_argument("--allow-cpu-transformers", action="store_true", help="Allow transformer training on CPU when CUDA is unavailable.")
    train_parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble retraining after the six base models.")
    train_parser.add_argument("--resume", action="store_true", help="Resume an existing run from saved state.")
    train_parser.add_argument("--dry-run", action="store_true", help="Initialize the run manifest without training.")
    train_parser.set_defaults(func=_run_training, use_rag=True)

    resume_parser = subparsers.add_parser("resume", help="Resume a paused training run.")
    resume_parser.add_argument("--run-name", default="full_retrain", help="Name of the training run.")
    resume_parser.add_argument("--models", default="all", help="Comma-separated list of models or 'all'.")
    resume_parser.add_argument("--epochs", type=int, default=30, help="Max epochs per transformer trial.")
    resume_parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience for transformers.")
    resume_parser.add_argument("--traditional-trials", type=int, default=6, help="Number of traditional-model trials per model.")
    resume_parser.add_argument("--transformer-trials", type=int, default=6, help="Number of transformer trials per model.")
    resume_parser.add_argument("--use-rag", dest="use_rag", action="store_true", help="Build RAG contexts for transformer training.")
    resume_parser.add_argument("--no-rag", dest="use_rag", action="store_false", help="Disable RAG context enrichment for transformer training.")
    resume_parser.add_argument("--gpu-profile", default="auto", choices=["auto", "rtx-a4000", "high-vram", "compat"], help="CUDA loader/batch preset for transformer training.")
    resume_parser.add_argument("--transformer-loss", default=DEFAULT_TRANSFORMER_LOSS, choices=["cross_entropy", "focal"], help="Loss function for transformer fine-tuning.")
    resume_parser.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA, help="Gamma value for focal loss when --transformer-loss focal is used.")
    resume_parser.add_argument("--allow-cpu-transformers", action="store_true", help="Allow transformer training on CPU when CUDA is unavailable.")
    resume_parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble retraining after the six base models.")
    resume_parser.add_argument("--dry-run", action="store_true", help="Load and print the current run state without training.")
    resume_parser.set_defaults(func=_run_training, resume=True, use_rag=True)

    pause_parser = subparsers.add_parser("pause", help="Request a graceful pause for a running training job.")
    pause_parser.add_argument("--run-name", default="full_retrain", help="Name of the training run.")
    pause_parser.set_defaults(func=_run_pause)

    stop_parser = subparsers.add_parser("stop", help="Request a graceful stop for a running training job.")
    stop_parser.add_argument("--run-name", default="full_retrain", help="Name of the training run.")
    stop_parser.set_defaults(func=_run_stop)

    status_parser = subparsers.add_parser("status", help="Show the current run manifest.")
    status_parser.add_argument("--run-name", default="full_retrain", help="Name of the training run.")
    status_parser.set_defaults(func=_run_status)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
