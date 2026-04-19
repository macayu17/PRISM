import torch
import sys
import os
from pathlib import Path
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from models.medical_transformers import (
    BioMistralClassifier as BioGPTForTabular,
    ClinicalT5Classifier as ClinicalT5ForTabular,
    PubMedBERTClassifier as PubMedBERTForTabular,
)

def print_memory(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] VRAM Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print(f"[{stage}] CPU only")

def main():
    print("Testing model initialization memory usage...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    input_dim = 24
    num_classes = 4
    
    # 1. PubMedBERT
    print("\n--- Testing PubMedBERT ---")
    print_memory("Start")
    try:
        model = PubMedBERTForTabular(input_dim, num_classes, dropout=0.15, freeze_bert=False).to(device)
        print("PubMedBERT loaded successfully")
        print_memory("Loaded")
        del model
        torch.cuda.empty_cache()
        print_memory("Cleared")
    except Exception as e:
        print(f"PubMedBERT Failed: {e}")

    # 2. BioGPT
    print("\n--- Testing BioGPT ---")
    print_memory("Start")
    try:
        model = BioGPTForTabular(input_dim, num_classes, dropout=0.15, train_decoder_layers=6).to(device)
        print("BioGPT loaded successfully")
        print_memory("Loaded")
        del model
        torch.cuda.empty_cache()
        print_memory("Cleared")
    except Exception as e:
        print(f"BioGPT Failed: {e}")

    # 3. Clinical-T5
    print("\n--- Testing Clinical-T5 ---")
    print_memory("Start")
    try:
        model = ClinicalT5ForTabular(input_dim, num_classes, dropout=0.15, freeze_encoder=False).to(device)
        print("Clinical-T5 loaded successfully")
        print_memory("Loaded")
        del model
        torch.cuda.empty_cache()
        print_memory("Cleared")
    except Exception as e:
        print(f"Clinical-T5 Failed: {e}")

if __name__ == "__main__":
    main()
