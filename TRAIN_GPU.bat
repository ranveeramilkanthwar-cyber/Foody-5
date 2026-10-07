@echo off
title FoodSafe AI - GPU Training (RTX 5050 + torch 2.12.0 cu128)
echo ============================================================
echo   FoodSafe AI - M: Drive Training Launcher
echo   Python : M:\Projects\ML\ChatBot\venv
echo   Torch  : 2.12.0.dev+cu128 (RTX 5050 VERIFIED)
echo   Data   : M:\food-safety-ai\dataset\
echo   Model  : M:\food-safety-ai\models\best_model.pt
echo ============================================================
echo.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set VENV_PY=M:\Projects\ML\ChatBot\venv\Scripts\python.exe

echo [1] Checking GPU...
%VENV_PY% -W ignore -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

echo.
echo [2] Starting training (5 epochs, 5000 images, 15 Indian food categories)...
echo     Log: M:\food-safety-ai\data\training_history.json
echo     Model will save to: M:\food-safety-ai\models\best_model.pt
echo.
%VENV_PY% "M:\food-safety-ai\train_gpu.py"
echo.
echo [3] Training done! Starting dashboard server...
%VENV_PY% "M:\food-safety-ai\server.py"
pause
