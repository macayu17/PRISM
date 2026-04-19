"""
Train and evaluate multimodal machine learning models for Parkinson's disease classification.
This script combines traditional ML, transformer models, and ensemble methods.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from data_preprocessing import DataPreprocessor
from models.multimodal_ml import MultimodalEnsemble, AdvancedFeatureEngineering, create_multimodal_pipeline


def main():
    """Main function to run multimodal ML pipeline."""
    print("Multimodal Machine Learning for Parkinson's Disease Classification")
    print("=" * 70)

    # --- Load and prepare data ---
    preprocessor = DataPreprocessor()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_paths = [
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20241211.csv"),
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20250321.csv"),
        os.path.join(base_dir, "PPMI_Curated_Data_Cut_Public_20250714.csv"),
    ]

    print("\nLoading and preparing data with patient-level split...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        file_paths,
        test_size=0.2,
        use_patient_split=True,
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Convert to numpy arrays if they are DataFrames
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values
        X_test_np = X_test.values
    else:
        X_train_np = X_train
        X_test_np = X_test

    if isinstance(y_train, pd.Series):
        y_train_np = y_train.values
        y_test_np = y_test.values
    else:
        y_train_np = y_train
        y_test_np = y_test

    # Create multimodal pipeline
    print("\nCreating multimodal ML pipeline...")
    ensemble, results = create_multimodal_pipeline(X_train, X_test, y_train_np, y_test_np)

    # Detailed evaluation of ensemble
    if ensemble.ensemble_model is not None:
        print("\nDetailed Ensemble Evaluation:")
        print("-" * 40)

        ensemble_results = ensemble.evaluate_ensemble(X_test, y_test_np)

        print(f"Ensemble Accuracy: {ensemble_results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(ensemble_results['classification_report'])

        # Plot confusion matrix
        os.makedirs('notebooks', exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(ensemble_results['confusion_matrix'],
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=['HC', 'PD', 'SWEDD', 'PRODROMAL'],
                   yticklabels=['HC', 'PD', 'SWEDD', 'PRODROMAL'])
        plt.title('Multimodal Ensemble - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('notebooks/multimodal_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Confusion matrix saved to notebooks/multimodal_confusion_matrix.png")

    # Advanced analysis
    print("\nAdvanced Multimodal Analysis:")
    print("-" * 40)

    # Feature importance analysis (if available)
    if hasattr(ensemble.ensemble_model, 'coef_'):
        feature_importance = np.abs(ensemble.ensemble_model.coef_).mean(axis=0)
        print("Top 10 most important ensemble features:")
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"{i+1:2d}. Feature {idx}: {feature_importance[idx]:.4f}")

    # Model diversity analysis
    print("\nModel Diversity Analysis:")
    if len(ensemble.traditional_models) > 0 and len(ensemble.transformer_models) > 0:
        trad_preds, _ = ensemble.get_traditional_predictions(X_test)
        trans_preds, _ = ensemble.get_transformer_predictions(X_test_np)

        if trad_preds and trans_preds:
            trad_pred = list(trad_preds.values())[0]
            trans_pred = list(trans_preds.values())[0]

            agreement = np.mean(trad_pred == trans_pred)
            print(f"Agreement between traditional and transformer models: {agreement:.4f}")

    # Performance summary
    print("\nFinal Performance Summary:")
    print("-" * 40)
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"Best performing model: {best_model[0]} (Accuracy: {best_model[1]:.4f})")

    if 'Ensemble' in results:
        ensemble_acc = results['Ensemble']
        individual_accs = [acc for name, acc in results.items() if name != 'Ensemble']
        if individual_accs:
            avg_individual = np.mean(individual_accs)
            improvement = ensemble_acc - avg_individual
            print(f"Ensemble improvement over average individual model: {improvement:.4f}")

    print("\nMultimodal ML pipeline completed successfully!")
    print("Results and visualizations saved to notebooks/ directory")


if __name__ == "__main__":
    main()