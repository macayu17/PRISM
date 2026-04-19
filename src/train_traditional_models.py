"""
Train and evaluate traditional ML models (LightGBM, XGBoost, SVM) on PPMI dataset
with patient-level split to prevent data leakage.
"""
from data_preprocessing import DataPreprocessor
from models.traditional_ml import TraditionalMLModels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


def plot_confusion_matrices(results, save_dir='../notebooks'):
    """Plot confusion matrices for all models."""
    os.makedirs(save_dir, exist_ok=True)

    for model_name, result in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()


def plot_roc_curves(results, y_test, save_dir='../notebooks'):
    """Plot ROC curves for all models using one-vs-rest approach."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    n_classes = len(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i in range(n_classes):
        for model_name, result in results.items():
            if 'probabilities' not in result:
                continue
            y_prob = result['probabilities'][:, i]
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob)
            roc_auc = auc(fpr, tpr)

            axes[i].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            axes[i].plot([0, 1], [0, 1], 'k--')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'ROC Curve - Class {i}')
            axes[i].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()


def main():
    print("=" * 80)
    print("TRAINING TRADITIONAL ML MODELS WITH PATIENT-LEVEL SPLIT")
    print("=" * 80)

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Use all available datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_paths = [
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20241211.csv"),
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20250321.csv"),
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20250714.csv"),
    ]

    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 80)

    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        file_paths,
        test_size=0.2,
        use_patient_split=True,
    )

    print("\n" + "=" * 80)
    print("STEP 2: CALCULATING CLASS WEIGHTS")
    print("=" * 80)

    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print("\nClass weights:", class_weight_dict)

    print("\n" + "=" * 80)
    print("STEP 3: INITIALIZING MODELS")
    print("=" * 80)

    ml_models = TraditionalMLModels()

    ml_models.models['lightgbm'] = LGBMClassifier(
        random_state=42,
        class_weight=class_weight_dict,
        objective='multiclass',
        num_class=len(class_weight_dict),
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
    )
    ml_models.models['xgboost'] = XGBClassifier(
        random_state=42,
        objective='multi:softmax',
        num_class=len(class_weight_dict),
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
    )
    ml_models.models['svm'] = SVC(
        random_state=42,
        class_weight=class_weight_dict,
        probability=True,
        decision_function_shape='ovr',
        kernel='rbf',
        C=10.0,
        gamma='scale',
    )

    print("\n" + "=" * 80)
    print("STEP 4: TRAINING MODELS")
    print("=" * 80)
    print("\nTraining models...")
    cv_scores = ml_models.train_all_models(X_train, y_train)

    print("\n" + "=" * 80)
    print("STEP 5: TRAINING SCORES")
    print("=" * 80)
    print("\nTraining scores:")
    for model_name, score in cv_scores.items():
        print(f"  {model_name}: {score:.4f}")

    print("\n" + "=" * 80)
    print("STEP 6: EVALUATING ON TEST SET")
    print("=" * 80)
    print("\nEvaluating models on test set...")
    results = ml_models.evaluate_all_models(X_test, y_test)

    print("\n" + "=" * 80)
    print("STEP 7: FINAL RESULTS")
    print("=" * 80)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"\nClassification Report:")
        print(result['classification_report'])

    print("\n" + "=" * 80)
    print("STEP 8: GENERATING VISUALIZATIONS")
    print("=" * 80)
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    print("\nVisualizations saved to notebooks/")

    print("\n" + "=" * 80)
    print("STEP 9: SAVING MODELS")
    print("=" * 80)
    print("\nSaving models...")
    ml_models.save_models()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n[OK] Models trained with patient-level split (no leakage)")
    print(f"[OK] {len(results)} models saved to models/saved/")
    print(f"[OK] Visualizations saved to notebooks/")
    print(f"[OK] Training data: {X_train.shape[0]} samples")
    print(f"[OK] Test data: {X_test.shape[0]} samples")
    print(f"[OK] Features: {X_train.shape[1]}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()