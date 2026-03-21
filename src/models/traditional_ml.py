import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib
import os

class TraditionalMLModels:
    def __init__(self, save_dir='../models/saved'):
        """Initialize the traditional ML models class.
        
        Args:
            save_dir (str): Directory to save trained models
        """
        self.models = {
            'lightgbm': None,
            'xgboost': None,
            'svm': None
        }
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model with optimized parameters."""
        print("Training LightGBM...")
        
        # Use pre-configured model if provided, otherwise create with good defaults
        if self.models['lightgbm'] is None:
            self.models['lightgbm'] = LGBMClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                objective='multiclass',
                num_class=4,
                verbose=-1
            )
        
        # Train the model
        self.models['lightgbm'].fit(X_train, y_train)
        
        # Return train score as approximation
        train_score = self.models['lightgbm'].score(X_train, y_train)
        print(f"LightGBM training accuracy: {train_score:.4f}")
        return train_score
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model with optimized parameters."""
        print("Training XGBoost...")
        
        # Use pre-configured model if provided, otherwise create with good defaults
        if self.models['xgboost'] is None:
            self.models['xgboost'] = XGBClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                objective='multi:softmax',
                num_class=4,
                eval_metric='mlogloss',
                verbosity=0
            )
        
        # Train the model
        self.models['xgboost'].fit(X_train, y_train)
        
        # Return train score as approximation
        train_score = self.models['xgboost'].score(X_train, y_train)
        print(f"XGBoost training accuracy: {train_score:.4f}")
        return train_score
    
    def train_svm(self, X_train, y_train):
        """Train SVM model with optimized parameters."""
        print("Training SVM...")
        
        # Use pre-configured model if provided, otherwise create with good defaults
        if self.models['svm'] is None:
            self.models['svm'] = SVC(
                random_state=42,
                probability=True,
                C=10.0,
                kernel='rbf',
                gamma='scale',
                decision_function_shape='ovr',
                verbose=False
            )
        
        # Train the model
        self.models['svm'].fit(X_train, y_train)
        
        # Return train score as approximation
        train_score = self.models['svm'].score(X_train, y_train)
        print(f"SVM training accuracy: {train_score:.4f}")
        return train_score
    
    def train_all_models(self, X_train, y_train):
        """Train all models and return their cross-validation scores."""
        scores = {}
        scores['lightgbm'] = self.train_lightgbm(X_train, y_train)
        scores['xgboost'] = self.train_xgboost(X_train, y_train)
        scores['svm'] = self.train_svm(X_train, y_train)
        return scores
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a specific model on test data."""
        model = self.models[model_name]
        if model is None:
            print(f"Model '{model_name}' is not trained yet.")
            return None

        y_pred = model.predict(X_test)
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{model_name} Test Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}")

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': probabilities,
            'classification_report': report,
            'confusion_matrix': cm
        }

    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models on test data."""
        results = {}
        for name in self.models:
            if self.models[name] is not None:
                results[name] = self.evaluate_model(name, X_test, y_test)
        return results
    
    def save_models(self):
        """Save all trained models to disk."""
        for model_name, model in self.models.items():
            if model is not None:
                save_path = os.path.join(self.save_dir, f"{model_name}_model.joblib")
                joblib.dump(model, save_path)
                print(f"Saved {model_name} model to {save_path}")
    
    def load_models(self):
        """Load all saved models from disk."""
        for model_name in self.models:
            model_path = os.path.join(self.save_dir, f"{model_name}_model.joblib")
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model from {model_path}")
            else:
                print(f"No saved model found for {model_name}")
