"""
Multimodal Machine Learning approach for Parkinson's disease classification.
This module combines traditional ML, transformer models, and ensemble methods.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os
import warnings
warnings.filterwarnings('ignore')

from traditional_ml import TraditionalMLModels
from transformer_models import TransformerModels, TabularDataset
from torch.utils.data import DataLoader


class MultimodalEnsemble:
    """
    Multimodal ensemble that combines traditional ML and transformer models.
    """
    
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.traditional_models = {}
        self.transformer_models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.model_weights = {}
        
    def load_traditional_models(self, model_dir: str = "models/saved"):
        """Load pre-trained traditional ML models."""
        model_files = {
            'lightgbm': 'lightgbm_model.joblib',
            'xgboost': 'xgboost_model.joblib', 
            'svm': 'svm_model.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                self.traditional_models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model from {model_path}")
            else:
                print(f"Warning: {model_name} model not found at {model_path}")
    
    def load_transformer_models(self, model_dir: str = "models/saved", input_dim: int = 31, num_classes: int = 4):
        """Load pre-trained transformer models (3 medical transformers only)."""
        
        # Load only the 3 working medical transformers
        try:
            from medical_transformers import MedicalTransformerTrainer
            medical_trainer = MedicalTransformerTrainer(device=self.device)
            
            # Only use 3 working medical transformer models
            new_model_configs = {
                'pubmedbert': {
                    'type': 'pubmedbert',
                    'params': {'dropout': 0.1, 'freeze_bert': True}
                },
                'biomistral': {
                    'type': 'biomistral',
                    'params': {'dropout': 0.2, 'use_quantization': False}  # Disable quantization to avoid bitsandbytes dependency
                }
            }
            
            print("Attempting to load new medical transformer models...")
            for model_name, config in new_model_configs.items():
                model_path = os.path.join(model_dir, f"{model_name}.pth")
                
                if os.path.exists(model_path):
                    try:
                        model = medical_trainer.load_model(
                            config['type'], model_name, input_dim, num_classes,
                            model_dir, **config['params']
                        )
                        self.transformer_models[model_name] = model
                        print(f"Loaded {model_name} medical transformer model")
                    except Exception as e:
                        print(f"Warning: Could not load {model_name}: {e}")
                else:
                    print(f"Info: {model_name} medical transformer not found (not yet trained)")
        
        except ImportError as e:
            print(f"Info: medical_transformers module not found: {e}")
        
        # Load simple feedforward transformer as 3rd transformer
        try:
            from transformer_models import TransformerModels
            transformer_trainer = TransformerModels(device=self.device)
            
            # Only load feedforward model (skip corrupted legacy transformers)
            model_name = 'feedforward'
            model_path = os.path.join(model_dir, f"{model_name}_transformer.pth")
            
            print("Attempting to load feedforward transformer model...")
            if os.path.exists(model_path):
                try:
                    model = transformer_trainer.load_model(
                        'feedforward', model_name, input_dim, num_classes, 
                        model_dir, hidden_dims=[256, 128, 64], dropout=0.3
                    )
                    self.transformer_models[model_name] = model
                    print(f"Loaded {model_name} transformer model")
                except Exception as e:
                    print(f"Warning: Could not load {model_name}: {e}")
            else:
                print(f"Info: {model_name} transformer not found")
        
        except ImportError as e:
            print(f"Warning: Could not import transformer_models: {e}")
        
        if len(self.transformer_models) > 0:
            print(f"Successfully loaded {len(self.transformer_models)} transformer model(s)")
        else:
            print("WARNING: No transformer models loaded! Ensemble will use traditional models only.")
    
    def get_traditional_predictions(self, X):
        """Get predictions from traditional ML models."""
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.traditional_models.items():
            try:
                pred = model.predict(X)
                pred_proba = model.predict_proba(X)
                predictions[model_name] = pred
                probabilities[model_name] = pred_proba
            except Exception as e:
                print(f"Error getting predictions from {model_name}: {e}")
        
        return predictions, probabilities
    
    def get_transformer_predictions(self, X):
        """Get predictions from transformer models."""
        predictions = {}
        probabilities = {}
        
        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            # Handle DataFrame conversion
            if hasattr(X, 'values'):
                X_vals = X.values
            else:
                X_vals = X
            X_tensor = torch.FloatTensor(X_vals).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        for model_name, model in self.transformer_models.items():
            try:
                model.eval()
                with torch.no_grad():
                    outputs = model(X_tensor)
                    proba = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(outputs, dim=1)
                    
                    predictions[model_name] = pred.cpu().numpy()
                    probabilities[model_name] = proba.cpu().numpy()
            except Exception as e:
                print(f"Error getting predictions from {model_name}: {e}")
        
        return predictions, probabilities
    
    def create_ensemble_features(self, X):
        """Create ensemble features from all models with optimized weights."""
        # Get predictions from all models
        trad_preds, trad_probas = self.get_traditional_predictions(X)
        trans_preds, trans_probas = self.get_transformer_predictions(X)
        
        # Define model weights for better performance
        model_weights = {
            # Traditional models - higher weights for better performers
            'lightgbm': 1.5,
            'xgboost': 1.3,
            'svm': 1.0,
            # New medical transformer models - highest weights for specialized medical models
            'pubmedbert': 2.2,  # Encoder model trained on PubMed abstracts
            'biomistral': 2.0,  # Decoder model with medical knowledge
            'clinical_t5': 2.1,  # Encoder-decoder model for clinical tasks
            # Legacy transformer models (for backward compatibility)
            'transformer_small': 1.2,
            'transformer_medium': 1.5,
            'transformer_large': 1.8,
            'feedforward': 1.0
        }
        
        # Combine all probability predictions as features with weights
        ensemble_features = []
        
        # Add traditional model probabilities with weights
        for model_name, proba in trad_probas.items():
            weight = model_weights.get(model_name, 1.0)
            ensemble_features.append(proba * weight)
        
        # Add transformer model probabilities with weights
        for model_name, proba in trans_probas.items():
            weight = model_weights.get(model_name, 1.0)
            ensemble_features.append(proba * weight)
        
        # Add original features (scaled down)
        if hasattr(X, 'values'):
            X_vals = X.values
        else:
            X_vals = X
        ensemble_features.append(X_vals * 0.15)  # Slightly increase original feature weight
        
        # Concatenate all features
        if ensemble_features:
            return np.concatenate(ensemble_features, axis=1)
        else:
            return X_vals
    
    def train_ensemble(self, X_train, y_train, ensemble_type: str = 'stacking'):
        """Train ensemble model on predictions from base models."""
        print(f"Training {ensemble_type} ensemble...")
        
        # Create ensemble features
        ensemble_features = self.create_ensemble_features(X_train)
        print(f"Ensemble features shape: {ensemble_features.shape}")
        
        if ensemble_type == 'stacking':
            # Use XGBoost as meta-learner for better performance
            self.ensemble_model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=2,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softproba',
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif ensemble_type == 'voting':
            # Create voting classifier (if we have sklearn-compatible models)
            available_models = []
            for name, model in self.traditional_models.items():
                available_models.append((name, model))
            
            if available_models:
                self.ensemble_model = VotingClassifier(
                    estimators=available_models,
                    voting='soft'
                )
            else:
                print("No traditional models available for voting ensemble")
                return
        
        # Train ensemble model
        self.ensemble_model.fit(ensemble_features, y_train)
        print(f"{ensemble_type.capitalize()} ensemble trained successfully")
    
    def predict_ensemble(self, X):
        """Make predictions using the ensemble model."""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained yet")
        
        ensemble_features = self.create_ensemble_features(X)
        predictions = self.ensemble_model.predict(ensemble_features)
        probabilities = self.ensemble_model.predict_proba(ensemble_features)
        
        return predictions, probabilities
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble model performance."""
        predictions, probabilities = self.predict_ensemble(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def compare_all_models(self, X_test, y_test):
        """Compare performance of all individual models and ensemble."""
        results = {}
        
        # Evaluate traditional models
        trad_preds, trad_probas = self.get_traditional_predictions(X_test)
        for model_name, pred in trad_preds.items():
            accuracy = accuracy_score(y_test, pred)
            results[f"Traditional_{model_name}"] = accuracy
        
        # Evaluate transformer models
        trans_preds, trans_probas = self.get_transformer_predictions(X_test)
        for model_name, pred in trans_preds.items():
            accuracy = accuracy_score(y_test, pred)
            results[f"Transformer_{model_name}"] = accuracy
        
        # Evaluate ensemble
        if self.ensemble_model is not None:
            ensemble_results = self.evaluate_ensemble(X_test, y_test)
            results["Ensemble"] = ensemble_results['accuracy']
        
        return results
    
    def plot_model_comparison(self, results: Dict, save_path: str = "notebooks/multimodal_comparison.png"):
        """Plot comparison of all models."""
        models = list(results.keys())
        accuracies = list(results.values())
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 
                                                 'gold', 'pink', 'lightgray', 'orange', 'red'])
        
        plt.title('Multimodal Model Comparison - Test Accuracy', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plot saved to {save_path}")
    
    def cross_validate_ensemble(self, X, y, cv_folds: int = 5):
        """Perform cross-validation on ensemble model."""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained yet")
        
        # Create ensemble features for full dataset
        ensemble_features = self.create_ensemble_features(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.ensemble_model, ensemble_features, y, 
            cv=cv_folds, scoring='accuracy'
        )
        
        return cv_scores
    
    def save_ensemble(self, save_path: str = "models/saved/multimodal_ensemble.joblib"):
        """Save the trained ensemble model."""
        if self.ensemble_model is not None:
            joblib.dump(self.ensemble_model, save_path)
            print(f"Ensemble model saved to {save_path}")
        else:
            print("No ensemble model to save")
    
    def load_ensemble(self, load_path: str = "models/saved/multimodal_ensemble.joblib"):
        """Load a pre-trained ensemble model."""
        if os.path.exists(load_path):
            self.ensemble_model = joblib.load(load_path)
            print(f"Ensemble model loaded from {load_path}")
        else:
            print(f"Ensemble model not found at {load_path}")


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for multimodal approach.
    """
    
    def __init__(self):
        self.feature_transformers = {}
        self.interaction_features = []
    
    def create_polynomial_features(self, X, degree: int = 2, feature_subset: List[str] = None):
        """Create polynomial features for selected columns."""
        from sklearn.preprocessing import PolynomialFeatures
        
        if feature_subset is None:
            # Use numerical features only
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            feature_subset = numerical_cols[:5]  # Limit to avoid explosion
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        X_subset = X[feature_subset]
        X_poly = poly.fit_transform(X_subset)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(feature_subset)
        
        # Add to original dataframe
        X_enhanced = X.copy()
        for i, name in enumerate(feature_names):
            if name not in X.columns:  # Avoid duplicates
                X_enhanced[f'poly_{name}'] = X_poly[:, i]
        
        self.feature_transformers['polynomial'] = poly
        return X_enhanced
    
    def create_clustering_features(self, X, n_clusters: int = 5):
        """Create clustering-based features."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Scale features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster features
        X_enhanced = X.copy()
        X_enhanced['cluster_label'] = cluster_labels
        
        # Add distance to each cluster center
        distances = kmeans.transform(X_scaled)
        for i in range(n_clusters):
            X_enhanced[f'dist_to_cluster_{i}'] = distances[:, i]
        
        self.feature_transformers['clustering'] = {'kmeans': kmeans, 'scaler': scaler}
        return X_enhanced
    
    def create_statistical_features(self, X):
        """Create statistical aggregation features."""
        X_enhanced = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            # Row-wise statistics
            X_enhanced['row_mean'] = X[numerical_cols].mean(axis=1)
            X_enhanced['row_std'] = X[numerical_cols].std(axis=1)
            X_enhanced['row_min'] = X[numerical_cols].min(axis=1)
            X_enhanced['row_max'] = X[numerical_cols].max(axis=1)
            X_enhanced['row_range'] = X_enhanced['row_max'] - X_enhanced['row_min']
            X_enhanced['row_skew'] = X[numerical_cols].skew(axis=1)
        
        return X_enhanced


def create_multimodal_pipeline(X_train, X_test, y_train, y_test):
    """
    Create and evaluate a complete multimodal ML pipeline.
    """
    print("Creating Multimodal ML Pipeline...")
    print("=" * 50)
    
    # Initialize multimodal ensemble
    ensemble = MultimodalEnsemble()
    
    # Load pre-trained models
    print("Loading pre-trained models...")
    ensemble.load_traditional_models()
    ensemble.load_transformer_models(input_dim=X_train.shape[1])
    
    # Advanced feature engineering
    print("Applying advanced feature engineering...")
    feature_engineer = AdvancedFeatureEngineering()
    
    # Create enhanced features
    X_train_enhanced = feature_engineer.create_polynomial_features(X_train)
    X_train_enhanced = feature_engineer.create_clustering_features(X_train_enhanced)
    X_train_enhanced = feature_engineer.create_statistical_features(X_train_enhanced)
    
    # Apply same transformations to test set
    X_test_enhanced = X_test.copy()
    if 'polynomial' in feature_engineer.feature_transformers:
        poly = feature_engineer.feature_transformers['polynomial']
        # Apply polynomial features to test set (implementation needed)
    
    print(f"Enhanced training features shape: {X_train_enhanced.shape}")
    
    # Train ensemble models
    ensemble.train_ensemble(X_train, y_train, ensemble_type='stacking')
    
    # Evaluate all models
    print("\nEvaluating all models...")
    results = ensemble.compare_all_models(X_test, y_test)
    
    # Print results
    print("\nModel Performance Comparison:")
    print("-" * 40)
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:<25}: {accuracy:.4f}")
    
    # Cross-validation
    print("\nPerforming cross-validation on ensemble...")
    cv_scores = ensemble.cross_validate_ensemble(X_train, y_train)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save ensemble model
    ensemble.save_ensemble()
    
    # Create visualizations
    ensemble.plot_model_comparison(results)
    
    return ensemble, results