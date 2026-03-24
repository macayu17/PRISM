@echo off
setlocal

set PYTHONUNBUFFERED=1
set TOKENIZERS_PARALLELISM=false
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python check_a4000_ready.py
if errorlevel 1 (
    echo.
    echo Preflight failed. Fix the issues above before starting training.
    exit /b 1
)

python src\train_model_suite.py train --run-name a4000_full --gpu-profile rtx-a4000 --epochs 30 --patience 8 --traditional-trials 4 --transformer-trials 4 %*
exit /b %errorlevel%
