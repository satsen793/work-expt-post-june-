#!/bin/bash
# Lightning AI quick setup using python venv + pip (no conda)
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# Try CUDA 12.1 wheels first; fallback to 12.4 if needed
if python - <<'PY'
import sys
try:
    import torch
    ok=True
except Exception:
    ok=False
sys.exit(0 if ok else 1)
PY
then
    echo "Torch already present."
else
    echo "Installing PyTorch (CUDA 12.1 wheels)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    (echo "Retrying with CUDA 12.4 wheels..." && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124)
fi

pip install numpy matplotlib scipy

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

echo "Venv ready. Use: source .venv/bin/activate"
