"""
Training script for transformer-based models (BioGPT, Clinical-T5, PubMedBERT)
on the PPMI dataset with RAG integration, CUDA acceleration, and leak-free patient split.

Usage:
    cd src
    python train_transformer_models.py
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import time
import json
from tqdm import tqdm

from data_preprocessing import DataPreprocessor
from models.transformer_models import TabularDataset
from models.medical_transformers import (
    BioMistralClassifier as BioGPTForTabular,
    ClinicalT5Classifier as ClinicalT5ForTabular,
    PubMedBERTClassifier as PubMedBERTForTabular,
)
from document_manager import DocumentManager


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
LEAK_FREE_SPLIT_PATH = ROOT / "evaluation_results" / "leak_free_split.npz"
LEAK_FREE_META_PATH = ROOT / "evaluation_results" / "leak_free_split_meta.joblib"
MODEL_DIR = ROOT / "models" / "saved"
RESULTS_DIR = ROOT / "evaluation_results"
PLOTS_DIR = ROOT / "evaluation_results" / "transformer_plots"

# [CONFIG] Set to True if you want to use RAG (slower start), False for faster training
USE_RAG = True



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_or_create_leak_free_split(preprocessor, file_paths):
    """Load the cached leak-free split or regenerate it if missing."""
    if LEAK_FREE_SPLIT_PATH.exists() and LEAK_FREE_META_PATH.exists():
        split = np.load(LEAK_FREE_SPLIT_PATH)
        meta = joblib.load(LEAK_FREE_META_PATH)
        feature_names = meta.get("feature_names") if isinstance(meta, dict) else None
        class_mapping = meta.get("class_mapping") if isinstance(meta, dict) else None
        print("[DATA] Loaded cached leak-free split from evaluation_results.")
        return (
            split["X_train"], split["X_test"],
            split["y_train"], split["y_test"],
            feature_names, class_mapping,
        )

    print("[DATA] Leak-free split not found – regenerating via DataPreprocessor ...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        file_paths, test_size=0.2, use_patient_split=True,
    )
    feature_names = preprocessor.get_feature_names()
    class_mapping = preprocessor.get_class_mapping()

    LEAK_FREE_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        LEAK_FREE_SPLIT_PATH,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )
    joblib.dump(
        {"feature_names": feature_names, "class_mapping": class_mapping},
        LEAK_FREE_META_PATH,
    )
    print(f"[DATA] Saved fresh leak-free split → {LEAK_FREE_SPLIT_PATH}")
    return X_train, X_test, y_train, y_test, feature_names, class_mapping


def _stratified_val_indices(y_train, val_fraction=0.15, seed=42):
    """Return (train_idx, val_idx) using a stratified split so every class
    is proportionally represented in the validation set."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(y_train)), y_train))
    return train_idx, val_idx


def _prepare_batch(batch, device):
    """Move tensors to the target device and keep optional RAG contexts aligned."""
    if len(batch) == 3:
        data, targets, contexts = batch
    else:
        data, targets = batch
        contexts = None
    data = data.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if contexts is not None:
        contexts = list(contexts)
    return data, targets, contexts


def _build_context_cache(features, build_fn, split_name="dataset", cache_path=None):
    """Pre-compute RAG contexts with parallel processing and caching."""
    if not USE_RAG:
        return [""] * len(features)

    if cache_path and os.path.exists(cache_path):
        print(f"  [RAG] Loading cached contexts from {cache_path}")
        return joblib.load(cache_path)

    print(f"  [RAG] Generating contexts for {len(features)} samples (Parallel)...")
    from joblib import Parallel, delayed
    
    # Run in parallel to speed up regex/cosine-sim
    contexts = Parallel(n_jobs=-1, verbose=5)(
        delayed(build_fn)(row) for row in features
    )
    
    if cache_path:
        joblib.dump(contexts, cache_path)
        print(f"  [RAG] Saved contexts to {cache_path}")
        
    return contexts



def _print_gpu_info(device):
    """Print GPU diagnostics."""
    if device.type != "cuda":
        return
    print(f"  GPU Name       : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version   : {torch.version.cuda}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  Compute Cap.   : {cap[0]}.{cap[1]}")
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Total VRAM     : {mem_total:.1f} GB")
    print(f"  cuDNN Enabled  : {torch.backends.cudnn.enabled}")
    print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_model(
    model, optimizer, scheduler, criterion, scaler,
    train_loader, val_loader, device, model_name,
    num_epochs=25, patience=8, grad_accum_steps=2,
    checkpoint_dir=None,
):
    """Train a single model with mixed precision, gradient accumulation, and
    early stopping.  Returns the best model state dict and training history."""

    checkpoint_path = checkpoint_dir / f"{model_name}_ckpt.pth" if checkpoint_dir else None
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}
    best_val_loss = float("inf")
    early_stop_counter = 0
    start_epoch = 0

    # Resume from checkpoint if available
    if checkpoint_path and checkpoint_path.exists():
        print(f"  [CKPT] Found checkpoint at {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            history = ckpt.get("history", history)
            print(f"  [CKPT] Resuming from epoch {start_epoch} (best val loss {best_val_loss:.4f})")
        except Exception as e:
            print(f"  [CKPT] Could not load checkpoint: {e}. Starting fresh.")
            start_epoch = 0

    use_amp = device.type == "cuda"

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        # Progress bar for training
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", ncols=100, ascii=True)
        
        for batch_idx, batch in pbar:
            data, targets, contexts = _prepare_batch(batch, device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(data, contexts)
                loss = criterion(outputs, targets) / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            current_loss = loss.item() * grad_accum_steps
            running_loss += current_loss
            
            # Update progress bar every few batches to reduce overhead
            if batch_idx % 10 == 0:
                pbar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val  {epoch+1}/{num_epochs}", unit="batch", leave=False, ncols=100, ascii=True):
                data, targets, contexts = _prepare_batch(batch, device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(data, contexts)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
        val_f1 = f1_score(all_targets, all_preds, average="weighted")

        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        scheduler.step(avg_val_loss)

        elapsed = time.time() - t0
        gpu_mem = torch.cuda.memory_allocated(0) / 1024**2 if device.type == "cuda" else 0

        print(
            f"  Epoch {epoch+1:02d}/{num_epochs} │ "
            f"Train Loss {avg_train_loss:.4f} │ Val Loss {avg_val_loss:.4f} │ "
            f"Val Acc {val_acc:.2f}% │ Val F1 {val_f1:.4f} │ "
            f"LR {optimizer.param_groups[0]['lr']:.2e} │ "
            f"GPU {gpu_mem:.0f}MB │ {elapsed:.1f}s"
        )

        # ---- Checkpointing & early stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if checkpoint_path:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                }, checkpoint_path)
            early_stop_counter = 0
            print(f"  [*] New best (val loss {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"  [X] Early stopping after {epoch+1} epochs")
                break

    # Load best weights
    if checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        checkpoint_path.unlink()

    return model, history, best_val_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_on_test(model, test_loader, criterion, device, model_name, class_names):
    """Full evaluation on the held-out test set."""
    model.eval()
    use_amp = device.type == "cuda"
    all_preds, all_targets, all_probs = [], [], []
    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}", unit="batch"):
            data, targets, contexts = _prepare_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(data, contexts)
                loss = criterion(outputs, targets)
            test_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    accuracy = np.mean(all_preds == all_targets)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    precision = precision_score(all_targets, all_preds, average="weighted")
    recall = recall_score(all_targets, all_preds, average="weighted")

    try:
        auroc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="weighted")
    except Exception:
        auroc = 0.0

    report = classification_report(all_targets, all_preds, target_names=class_names)
    cm = confusion_matrix(all_targets, all_preds)

    print(f"\n{'='*70}")
    print(f"  {model_name.upper()} — TEST SET RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"\n{report}")

    return {
        "accuracy": accuracy, "f1": f1, "precision": precision,
        "recall": recall, "auroc": auroc,
        "classification_report": report, "confusion_matrix": cm,
        "predictions": all_preds, "targets": all_targets, "probabilities": all_probs,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_plots(results, history_dict, class_names, plots_dir):
    """Save confusion matrices, training curves, and comparison charts."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for name, res in results.items():
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{name} — Confusion Matrix (Leak-Free Split)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{name}_confusion_matrix.png", dpi=200)
        plt.close()

        # Training curves
        hist = history_dict[name]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(hist["train_loss"], label="Train", color="#3498db")
        axes[0].plot(hist["val_loss"], label="Val", color="#e74c3c")
        axes[0].set_title(f"{name} — Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(hist["val_acc"], color="#2ecc71")
        axes[1].set_title(f"{name} — Val Accuracy (%)")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(hist["lr"], color="#9b59b6")
        axes[2].set_title(f"{name} — Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / f"{name}_training_curves.png", dpi=200)
        plt.close()

    # Comparison bar chart
    model_names = list(results.keys())
    metric_names = ["accuracy", "f1", "precision", "recall", "auroc"]
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for ax, metric in zip(axes, metric_names):
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(model_names, values, color=colors[:len(model_names)])
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_dir / "transformer_comparison.png", dpi=200)
    plt.close()

    print(f"[PLOT] Saved all plots → {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  TRANSFORMER MODEL TRAINING — LEAK-FREE SPLIT + CUDA")
    print("=" * 70)

    # ---- Seed everything ----
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Using: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        _print_gpu_info(device)
    else:
        print("[WARNING] CUDA not available -- training will be VERY slow on CPU!")
        print("[WARNING] Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124")

    # ---- Data ----
    preprocessor = DataPreprocessor()
    base_dir = str(ROOT)
    file_paths = [
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20240129.csv"),
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20241211.csv"),
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20250321.csv"),
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20250714.csv"),
    ]

    X_train, X_test, y_train, y_test, feature_names, class_mapping = \
        _load_or_create_leak_free_split(preprocessor, file_paths)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    print(f"[DATA] Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"[DATA] Classes: {len(np.unique(y_train))}  Distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ---- Class weights ----
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.FloatTensor(cw).to(device)
    print(f"[DATA] Class weights: {dict(zip(np.unique(y_train), np.round(cw, 3)))}")

    # ---- Stratified validation split (preserves class ratios) ----
    train_idx, val_idx = _stratified_val_indices(y_train, val_fraction=0.15)
    print(f"[DATA] Stratified split: {len(train_idx)} train / {len(val_idx)} val")

    # ---- Feature names ----
    if feature_names is None:
        feature_names = preprocessor.get_feature_names()

    # ---- RAG context ----
    docs_path = str(ROOT / "medical_docs")
    doc_manager = DocumentManager(docs_dir=docs_path)
    doc_count = doc_manager.get_document_count()
    print(f"[RAG] Loaded {doc_count.get('total', doc_count)} documents for context enrichment")

    def get_rag_context(sample_features):
        feature_desc = {name: float(val) for name, val in zip(feature_names, sample_features)}
        query_parts = []
        symptoms = {
            "tremor": feature_desc.get("sym_tremor", 0),
            "rigidity": feature_desc.get("sym_rigid", 0),
            "bradykinesia": feature_desc.get("sym_brady", 0),
            "postural instability": feature_desc.get("sym_posins", 0),
        }
        for symptom, severity in symptoms.items():
            if severity > 0:
                query_parts.append(f"{symptom} severity:{severity}")
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
            return ""
        return " ".join(
            f"From '{p['doc_title']}' {p['text'][:300]}..." for p in passages
        )

    print(f"\n[RAG] RAG Enabled: {USE_RAG}")
    if USE_RAG:
        print("[RAG] Pre-computing context for train + test splits ...")
    
    train_cache = RESULTS_DIR / "rag_contexts_train.pkl"
    test_cache = RESULTS_DIR / "rag_contexts_test.pkl"
    
    train_contexts = _build_context_cache(X_train, get_rag_context, "train", train_cache)
    test_contexts = _build_context_cache(X_test, get_rag_context, "test", test_cache)


    # ---- Datasets ----
    full_train_ds = TabularDataset(X_train, y_train, feature_names, contexts=train_contexts)
    test_ds = TabularDataset(X_test, y_test, feature_names, contexts=test_contexts)

    train_subset = Subset(full_train_ds, train_idx)
    val_subset = Subset(full_train_ds, val_idx)

    # ---- DataLoaders ----
    # Windows does not support multiprocessing well with num_workers > 0
    # unless guarded by if __name__ == '__main__'. We set 0 for safety.
    pin = device.type == "cuda"
    nw = 0  # safe for Windows

    # Batch sizes — adjusted for GTX 1650 Ti (4GB VRAM)
    # train_bs=8 uses ~2-3GB VRAM. We use Gradient Accumulation to reach effective batch size of 64.
    train_bs = 8 if device.type == "cuda" else 8
    eval_bs = 32 if device.type == "cuda" else 16

    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, pin_memory=pin, num_workers=nw)
    val_loader = DataLoader(val_subset, batch_size=eval_bs, shuffle=False, pin_memory=pin, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, pin_memory=pin, num_workers=nw)

    print(f"\n[LOADER] Train batches: {len(train_loader)}  Val batches: {len(val_loader)}  Test batches: {len(test_loader)}")
    print(f"[LOADER] Batch sizes: train={train_bs}  eval={eval_bs}  pin_memory={pin}")

    # ---- Model definitions (lazy — created one at a time to fit in 4GB VRAM) ----
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    class_names = ["HC", "PD", "SWEDD", "PRODROMAL"]

    print(f"\n[MODEL] Input dim: {input_dim}  Num classes: {num_classes}")

    # Each entry: (display_name, save_name, model_factory)
    # Models are created lazily inside the loop to avoid GPU OOM.
    model_configs = [
        (
            "PubMedBERT (Encoder-Only)", "pubmedbert",
            lambda: PubMedBERTForTabular(input_dim, num_classes, dropout=0.15, freeze_bert=False),
            {"lr": 2e-5, "weight_decay": 0.01},
        ),
        (
            "BioGPT", "biogpt",
            lambda: BioGPTForTabular(input_dim, num_classes, dropout=0.15, train_decoder_layers=6),
            {"lr": 3e-5, "weight_decay": 0.01},
        ),
        (
            "Clinical-T5", "clinical_t5",
            lambda: ClinicalT5ForTabular(input_dim, num_classes, dropout=0.15, freeze_encoder=False),
            {"lr": 2e-5, "weight_decay": 0.01},
        ),
    ]

    # ---- Training config ----
    NUM_EPOCHS = 25
    PATIENCE = 8
    GRAD_ACCUM = 8  # effective batch size = train_bs(8) * GRAD_ACCUM(8) = 64

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    checkpoint_dir = MODEL_DIR / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_histories = {}

    for display_name, save_name, model_factory, opt_kwargs in model_configs:
        final_path = MODEL_DIR / f"{display_name}_best.pth"

        # ---- Skip if already trained ----
        if final_path.exists():
            print(f"\n{'='*70}")
            print(f"  SKIPPING: {display_name}  (already trained)")
            print(f"  Loading saved weights from: {final_path}")
            print(f"{'='*70}")
            try:
                # Create the model on CPU first, load weights, then move to GPU
                print(f"\n[MODEL] Initializing {display_name} for evaluation ...")
                model = model_factory()
                model.load_state_dict(torch.load(final_path, map_location="cpu", weights_only=True))
                model.to(device)

                # Evaluate on test set
                result = evaluate_on_test(
                    model, test_loader, criterion, device, display_name, class_names,
                )
                all_results[display_name] = result
                all_histories[display_name] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}

                # Free GPU memory before next model
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                continue
            except RuntimeError as e:
                print(f"  [WARN] Cannot load saved weights (architecture changed?): {e}")
                print(f"  [WARN] Deleting stale checkpoint and retraining ...")
                final_path.unlink(missing_ok=True)

        # ---- Create model fresh for training ----
        print(f"\n[MODEL] Initializing {display_name} for training ...")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model = model_factory().to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            **opt_kwargs,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7,
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"\n{'='*70}")
        print(f"  TRAINING: {display_name}")
        print(f"  Trainable params: {trainable_params:,} / {total_params:,}")
        print(f"  Epochs: {NUM_EPOCHS}  Patience: {PATIENCE}  Grad Accum: {GRAD_ACCUM}")
        print(f"{'='*70}")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        scaler = torch.amp.GradScaler(device=device.type, enabled=device.type == "cuda")

        trained_model, history, best_val = train_one_model(
            model, optimizer, scheduler, criterion, scaler,
            train_loader, val_loader, device, save_name,
            num_epochs=NUM_EPOCHS, patience=PATIENCE, grad_accum_steps=GRAD_ACCUM,
            checkpoint_dir=checkpoint_dir,
        )

        # Save final model
        torch.save(trained_model.state_dict(), final_path)
        print(f"  [SAVE] Model saved → {final_path}")

        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(0) / 1024**3
            print(f"  [GPU] Peak VRAM usage: {peak:.2f} GB")

        # Evaluate on test set
        result = evaluate_on_test(
            trained_model, test_loader, criterion, device, display_name, class_names,
        )
        all_results[display_name] = result
        all_histories[display_name] = history

        # Free GPU memory before next model
        del model, trained_model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # NOTE: We intentionally keep the checkpoint dir so that interrupted
    # training can resume from the last saved epoch on the next run.

    # ---- Save metrics to JSON + CSV ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, res in all_results.items():
        summary_rows.append({
            "Model": name,
            "Type": "Transformer",
            "Accuracy": round(res["accuracy"], 4),
            "F1_Score": round(res["f1"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "AUROC": round(res["auroc"], 4),
        })
    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS_DIR / "transformer_metrics_latest.csv", index=False)
    with open(RESULTS_DIR / "transformer_metrics_latest.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n[SAVE] Metrics  → {RESULTS_DIR / 'transformer_metrics_latest.csv'}")

    # ---- Plots ----
    save_plots(all_results, all_histories, class_names, PLOTS_DIR)

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'AUROC':>10}")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['Model']:<30} {row['Accuracy']:>10.4f} {row['F1_Score']:>10.4f} "
              f"{row['Precision']:>10.4f} {row['Recall']:>10.4f} {row['AUROC']:>10.4f}")

    best_by_f1 = max(all_results, key=lambda k: all_results[k]["f1"])
    print(f"\n  [*] Best model (by F1): {best_by_f1} -- F1 {all_results[best_by_f1]['f1']:.4f}")
    print(f"\n{'='*70}")
    print("  TRAINING COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()