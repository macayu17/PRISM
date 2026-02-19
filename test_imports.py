import time
print("Starting imports...", flush=True)

t0 = time.time()
import sys
print(f"sys imported: {time.time()-t0:.2f}s", flush=True)

t0 = time.time()
import os
print(f"os imported: {time.time()-t0:.2f}s", flush=True)

t0 = time.time()
import numpy as np
print(f"numpy imported: {time.time()-t0:.2f}s", flush=True)

t0 = time.time()
import torch
print(f"torch imported: {time.time()-t0:.2f}s", flush=True)

print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

t0 = time.time()
from transformers import AutoModel, AutoTokenizer
print(f"transformers imported: {time.time()-t0:.2f}s", flush=True)
