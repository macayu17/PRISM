"""
Medical Transformer Models for Parkinson's Disease Classification
Implements three specialized medical transformers:
1. PubMedBERT - Encoder-only model
2. BioMistral-7B - Decoder-only model
3. Clinical-T5 - Encoder-Decoder model

Optimized for GPU training with CUDA support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
import joblib
import os
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BitsAndBytesConfig
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MedicalTabularDataset(Dataset):
    """Enhanced dataset for tabular medical data with feature descriptions."""
    
    def __init__(self, X, y, feature_names=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Feature descriptions for medical context
        self.feature_descriptions = {
            'age': 'patient age in years',
            'SEX': 'biological sex (0=Female, 1=Male)',
            'EDUCYRS': 'years of education',
            'BMI': 'body mass index',
            'fampd': 'family history of Parkinsons disease',
            'sym_tremor': 'tremor severity score',
            'sym_rigid': 'rigidity severity score',
            'sym_brady': 'bradykinesia severity score',
            'sym_posins': 'postural instability score',
            'moca': 'Montreal Cognitive Assessment score',
            'gds': 'Geriatric Depression Scale score',
            'stai': 'State-Trait Anxiety Inventory score',
            'rem': 'REM sleep behavior disorder check',
            'ess': 'Epworth Sleepiness Scale',
            'upsit': 'University of Pennsylvania Smell Identification Test score',
            'upsit_pctl': 'smell identification percentile',
            'updrs1_score': 'UPDRS Part I (non-motor experiences of daily living)',
            'updrs2_score': 'UPDRS Part II (motor experiences of daily living)',
            'updrs3_score': 'UPDRS Part III (motor examination)',
            'updrs4_score': 'UPDRS Part IV (motor complications)',
            'updrs_totscore': 'Total UPDRS score',
            'mean_caudate': 'DatScan mean caudate uptake ratio',
            'mean_putamen': 'DatScan mean putamen uptake ratio',
            'abeta': 'CSF Amyloid beta 1-42 level',
            'tau': 'CSF Total Tau level',
            'ptau': 'CSF Phosphorylated Tau level'
        }
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_text_description(self, idx):
        """Convert features to medical text description."""
        sample = self.X[idx].numpy()
        descriptions = []
        
        for i, (name, val) in enumerate(zip(self.feature_names, sample)):
            base_name = name.split('_')[0] if '_' in name else name
            desc = self.feature_descriptions.get(base_name, f"{name}")
            descriptions.append(f"{desc}: {val:.2f}")
        
        return "Patient clinical data: " + ", ".join(descriptions)


class PubMedBERTClassifier(nn.Module):
    """
    PubMedBERT - Encoder-only transformer model
    Pretrained on PubMed abstracts, optimized for medical text understanding
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1, 
                 freeze_bert: bool = True):
        super(PubMedBERTClassifier, self).__init__()
        
        self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        print(f"Loading PubMedBERT from {self.model_name}")
        
        # Load pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)
        
        # Optionally freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("PubMedBERT parameters frozen")
        else:
            # Fine-tune last 4 layers
            for param in list(self.bert.parameters())[:-48]:
                param.requires_grad = False
            print("PubMedBERT last 4 layers unfrozen for fine-tuning")
        
        self.hidden_size = self.bert.config.hidden_size  # 768 for base model
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, self.hidden_size)
        )
        
        # Multi-head attention for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, text_input=None):
        batch_size = x.size(0)
        
        # Project tabular features
        tabular_features = self.feature_projection(x)  # [batch, 768]
        tabular_features = tabular_features.unsqueeze(1)  # [batch, 1, 768]
        
        if text_input is not None:
            # Process text with BERT
            inputs = self.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(x.device) for k, v in inputs.items()}
            
            # Get BERT embeddings
            outputs = self.bert(**inputs)
            text_features = outputs.last_hidden_state  # [batch, seq_len, 768]
            
            # Concatenate features
            combined = torch.cat([tabular_features, text_features], dim=1)
            
            # Apply attention
            attended, _ = self.attention(combined, combined, combined)
            pooled = attended.mean(dim=1)  # Average pooling
        else:
            # Use only tabular features
            pooled = tabular_features.squeeze(1)
        
        # Combine original and attended features
        combined_features = torch.cat([pooled, tabular_features.squeeze(1)], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output


class BioMistralClassifier(nn.Module):
    """BioGPT (decoder-only) hybrid with optional partial fine-tuning."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantization: bool = False,
        train_decoder_layers: int = 4
    ):
        super(BioMistralClassifier, self).__init__()
        
        # Use a smaller medical language model that's publicly available
        # BioGPT or similar decoder-only model
        self.model_name = "microsoft/biogpt"
        print(f"Loading BioGPT (Decoder-only) from {self.model_name}")
        
        try:
            # Load without quantization and device_map to avoid accelerate dependency
            # Quantization disabled by default to avoid bitsandbytes and accelerate issues
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use standard precision
            )
            print("Model loaded successfully (no quantization)")
                
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            
            # Freeze decoder parameters (we may selectively unfreeze later)
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.hidden_size = self.model.config.hidden_size
            
        except Exception as e:
            print(f"Error loading BioGPT: {e}")
            print("Falling back to GPT2 architecture")
            # Fallback to a simpler model
            from transformers import GPT2Model, GPT2Tokenizer
            self.model_name = "gpt2"
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2Model.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.hidden_size = self.model.config.hidden_size
            
            for param in self.model.parameters():
                param.requires_grad = False
        
        self._unfreeze_decoder_layers(train_decoder_layers)
        
        # Feature encoder (larger capacity)
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.hidden_size),
            nn.GELU()
        )
        
        # Cross-attention layer (restore 8 heads)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head with more width/depth
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2 if dropout > 0 else 0.0),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, text_input=None):
        # Encode tabular features
        feature_embeddings = self.feature_encoder(x)  # [batch, hidden_size]
        feature_embeddings = feature_embeddings.unsqueeze(1)  # [batch, 1, hidden_size]
        
        if text_input is not None:
            # Tokenize text
            inputs = self.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(x.device) for k, v in inputs.items()}
            
            # Get decoder outputs (no torch.no_grad — allow gradients for unfrozen layers)
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Cross-attention between features and text
            attended, _ = self.cross_attention(
                feature_embeddings,
                hidden_states,
                hidden_states
            )
            pooled = attended.squeeze(1)
        else:
            pooled = feature_embeddings.squeeze(1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

    def _unfreeze_decoder_layers(self, num_layers: int):
        """Optionally unfreeze the last decoder layers to grow capacity."""
        if num_layers <= 0:
            print("BioGPT remains fully frozen (train_decoder_layers=0)")
            return

        # Debug: print model structure to diagnose layer discovery
        print(f"  [DEBUG] Model type: {type(self.model).__name__}")
        print(f"  [DEBUG] Top-level children: {[n for n, _ in self.model.named_children()]}")

        layers = None
        layernorm = None
        head = None

        # Strategy 1: Try known attribute paths
        if hasattr(self.model, 'biogpt'):
            layers = getattr(self.model.biogpt, 'layers', None)
            layernorm = getattr(self.model.biogpt, 'layer_norm', None)
            head = getattr(self.model, 'output_projection', None)
        elif hasattr(self.model, 'transformer'):
            layers = getattr(self.model.transformer, 'h', None)
            layernorm = getattr(self.model.transformer, 'ln_f', None)
            head = getattr(self.model, 'lm_head', None)

        # Strategy 2: Search all named modules for a large ModuleList (transformer layers)
        if layers is None:
            print("  [DEBUG] Known paths failed, searching named_modules...")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 6:
                    layers = module
                    print(f"  [DEBUG] Found transformer layers at '{name}' ({len(module)} layers)")
                    break

        # Strategy 3: Search for layernorm and head if not found yet
        if layers is not None and layernorm is None:
            for name, module in self.model.named_modules():
                if ('layer_norm' in name or 'ln_f' in name) and not any(c in name for c in ['.0.', '.1.', '.2.']):
                    layernorm = module
                    break
        if layers is not None and head is None:
            for name, module in self.model.named_modules():
                if 'output_projection' in name or 'lm_head' in name:
                    head = module
                    break

        if layers is None:
            print("Warning: could not locate transformer layers to unfreeze")
            print("  All named modules:")
            for name, module in self.model.named_children():
                print(f"    {name}: {type(module).__name__}")
            return

        total_layers = len(layers)
        num_layers = min(num_layers, total_layers)

        print(f"Unfreezing last {num_layers}/{total_layers} layers of BioGPT/GPT2...")

        # Unfreeze last N layers
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze final layer norm
        if layernorm is not None:
            for param in layernorm.parameters():
                param.requires_grad = True

        # Unfreeze head
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Unfroze last {num_layers} decoder layers plus norm/head for BioGPT")
        print(f"  Backbone trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")


class ClinicalT5Classifier(nn.Module):
    """
    Clinical-T5 - Encoder-Decoder transformer model
    T5 model fine-tuned on clinical notes and medical literature
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1,
                 freeze_encoder: bool = True):
        super(ClinicalT5Classifier, self).__init__()
        
        # Use T5 or Flan-T5 as base model
        self.model_name = "google/flan-t5-base"
        print(f"Loading Clinical T5 (Encoder-Decoder) from {self.model_name}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Freeze encoder/decoder parameters
        if freeze_encoder:
            for param in self.t5.encoder.parameters():
                param.requires_grad = False
            for param in self.t5.decoder.parameters():
                param.requires_grad = False
            print("T5 encoder and decoder frozen")
        else:
            # Freeze all first, then selectively unfreeze last blocks
            for param in self.t5.parameters():
                param.requires_grad = False
            # Unfreeze last 4 encoder blocks
            for block in self.t5.encoder.block[-4:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Unfreeze last 2 decoder blocks
            for block in self.t5.decoder.block[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Unfreeze final layer norms
            if hasattr(self.t5.encoder, 'final_layer_norm'):
                for param in self.t5.encoder.final_layer_norm.parameters():
                    param.requires_grad = True
            if hasattr(self.t5.decoder, 'final_layer_norm'):
                for param in self.t5.decoder.final_layer_norm.parameters():
                    param.requires_grad = True
            trainable = sum(p.numel() for p in self.t5.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.t5.parameters())
            print(f"T5 partially unfrozen: {trainable:,}/{total:,} params trainable ({100*trainable/total:.1f}%)")
        
        self.hidden_size = self.t5.config.d_model  # 768 for base
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, self.hidden_size)
        )
        
        # Encoder-decoder attention mechanism
        self.encoder_decoder_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, text_input=None):
        # Project features
        feature_embeddings = self.feature_projection(x)
        feature_embeddings = feature_embeddings.unsqueeze(1)
        
        if text_input is not None:
            # Prepare input text
            input_texts = [f"classify patient: {text}" for text in text_input]
            
            # Tokenize
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(x.device) for k, v in inputs.items()}
            
            # Create dummy decoder inputs
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1),
                dtype=torch.long,
                device=x.device
            )
            
            # Get encoder and decoder outputs (allow gradients for unfrozen layers)
            encoder_outputs = self.t5.encoder(**inputs)
            decoder_outputs = self.t5.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state
            )
            
            encoder_hidden = encoder_outputs.last_hidden_state
            decoder_hidden = decoder_outputs.last_hidden_state
            
            # Combine encoder and decoder information
            combined = torch.cat([encoder_hidden, decoder_hidden], dim=1)
            
            # Attend to features
            attended, _ = self.encoder_decoder_attention(
                feature_embeddings,
                combined,
                combined
            )
            
            pooled = attended.squeeze(1)
        else:
            pooled = feature_embeddings.squeeze(1)
        
        # Combine with original features
        combined_features = torch.cat([pooled, feature_embeddings.squeeze(1)], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output


class MedicalTransformerTrainer:
    """
    Trainer class for medical transformer models with GPU optimization
    """
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.models = {}
        self.training_histories = {}
        self.evaluation_results = {}
    
    def create_model(self, model_type: str, input_dim: int, num_classes: int, **kwargs):
        """Create a medical transformer model."""
        if model_type == 'pubmedbert':
            model = PubMedBERTClassifier(input_dim, num_classes, **kwargs)
        elif model_type == 'biomistral':
            model = BioMistralClassifier(input_dim, num_classes, **kwargs)
        elif model_type == 'clinical_t5':
            model = ClinicalT5Classifier(input_dim, num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def train_model(self, model, train_loader, val_loader, model_name: str,
                   epochs: int = 50, lr: float = 0.001, weight_decay: float = 1e-4,
                   class_weights: torch.Tensor = None, use_amp: bool = True):
        """
        Train model with mixed precision and gradient accumulation for GPU efficiency
        """
        print(f"\nTraining {model_name}")
        print("=" * 70)
        
        # Loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device.type == 'cuda' else None
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 15
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_steps = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_loss /= train_steps
            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(all_targets, all_preds)
            val_f1 = f1_score(all_targets, all_preds, average='weighted')
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Step scheduler
            scheduler.step()
            
            # Print progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Accuracy: {val_accuracy:.4f}")
                print(f"  Val F1: {val_f1:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_accuracy
                patience_counter = 0
                # Save best model
                self.save_model(model, f"{model_name}_best")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    print(f"Best validation accuracy: {best_val_acc:.4f}")
                    break
        
        self.training_histories[model_name] = history
        return history
    
    def evaluate_model(self, model, test_loader, model_name: str):
        """Comprehensive model evaluation."""
        print(f"\nEvaluating {model_name}")
        print("=" * 70)
        
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_x)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # ROC-AUC for multiclass
        try:
            y_bin = label_binarize(all_targets, classes=range(len(np.unique(all_targets))))
            roc_auc = roc_auc_score(y_bin, all_probs, average='weighted', multi_class='ovr')
        except:
            roc_auc = 0.0
        
        cm = confusion_matrix(all_targets, all_preds)
        report = classification_report(all_targets, all_preds)
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_preds,
            'probabilities': all_probs,
            'targets': all_targets
        }
        
        print(f"\nResults for {model_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"\nClassification Report:\n{report}")
        
        self.evaluation_results[model_name] = results
        return results
    
    def save_model(self, model, model_name: str, save_dir: str = "./models/saved"):
        """Save model with error handling."""
        try:
            # Use absolute path
            if not os.path.isabs(save_dir):
                save_dir = os.path.abspath(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"[OK] Model saved to {model_path}")
        except Exception as e:
            print(f"[ERR] Error saving model: {e}")
            import traceback
            traceback.print_exc()
    
    def load_model(self, model_type: str, model_name: str, input_dim: int,
                   num_classes: int, save_dir: str = "./models/saved", **kwargs):
        """Load saved model."""
        # Use absolute path
        if not os.path.isabs(save_dir):
            save_dir = os.path.abspath(save_dir)
        model = self.create_model(model_type, input_dim, num_classes, **kwargs)
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def plot_training_curves(self, model_name: str, save_dir: str = "./notebooks"):
        """Plot training curves."""
        if model_name not in self.training_histories:
            print(f"No training history found for {model_name}")
            return
        
        # Use absolute path
        if not os.path.isabs(save_dir):
            save_dir = os.path.abspath(save_dir)
        
        history = self.training_histories[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', 
                       color='green', linewidth=2)
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score curve
        axes[1, 0].plot(history['val_f1'], label='Val F1 Score', 
                       color='orange', linewidth=2)
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(history['learning_rates'], label='Learning Rate', 
                       color='red', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{model_name}_training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved for {model_name}")
    
    def plot_confusion_matrix(self, model_name: str, class_names: List[str] = None,
                            save_dir: str = "./notebooks"):
        """Plot confusion matrix."""
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        # Use absolute path
        if not os.path.isabs(save_dir):
            save_dir = os.path.abspath(save_dir)
        
        cm = self.evaluation_results[model_name]['confusion_matrix']
        
        if class_names is None:
            class_names = ['HC', 'PD', 'SWEDD', 'PRODROMAL']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved for {model_name}")
    
    def save_evaluation_results(self, save_dir: str = "../notebooks"):
        """Save all evaluation results to file."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_file = os.path.join(save_dir, f'evaluation_results_{timestamp}.txt')
        
        with open(results_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Medical Transformer Models - Evaluation Results\n")
            f.write("=" * 70 + "\n\n")
            
            for model_name, results in self.evaluation_results.items():
                f.write(f"\n{model_name}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"F1 Score: {results['f1_score']:.4f}\n")
                f.write(f"ROC-AUC: {results['roc_auc']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(results['classification_report'])
                f.write("\n" + "=" * 70 + "\n")
        
        print(f"\nEvaluation results saved to {results_file}")
