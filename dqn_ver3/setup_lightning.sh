#!/bin/bash
# Lightning AI setup script
set -euo pipefail

echo "Setting up DQN environment on Lightning AI..."

# Create conda env if not exists
if ! conda env list | grep -q "dqn_gpu"; then
    echo "Creating conda environment..."
    conda create -n dqn_gpu python=3.10 -y
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dqn_gpu

# Install packages
echo "Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib scipy

# Verify GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "Setup complete!"
