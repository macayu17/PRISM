import time

print("Starting imports...", flush=True)

t0 = time.time()
import sys

print(
    f"sys imported: {time.time() - t0:.2f}s | version={sys.version.split()[0]}",
    flush=True,
)

t0 = time.time()
import os

print(f"os imported: {time.time() - t0:.2f}s | cwd={os.getcwd()}", flush=True)

t0 = time.time()
import numpy as np

print(f"numpy imported: {time.time() - t0:.2f}s | version={np.__version__}", flush=True)

t0 = time.time()
import torch

print(
    f"torch imported: {time.time() - t0:.2f}s | version={torch.__version__}",
    flush=True,
)

print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

t0 = time.time()
from transformers import AutoModel, AutoTokenizer

print(
    "transformers imported: "
    f"{time.time() - t0:.2f}s | "
    f"AutoModel={AutoModel.__name__} | AutoTokenizer={AutoTokenizer.__name__}",
    flush=True,
)
