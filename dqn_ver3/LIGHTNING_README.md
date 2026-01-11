# Lightning AI - Complete DQN Training Guide

## Quick Start (1 hour budget)

### 1. Transfer code to Lightning AI

From Windows PowerShell:
```powershell
# Create tarball
cd C:\Users\HP\Videos
tar -czf dqn_ver3.tar.gz --exclude='.venv*' --exclude='__pycache__' --exclude='*.pyc' --exclude='logs' --exclude='figures' dqn_ver3

# Upload to Lightning AI
scp -i C:\Users\HP\.ssh\lightning_rsa dqn_ver3.tar.gz ssh.lightning.ai:~/
```

### 2. SSH into Lightning AI

```powershell
ssh -i C:\Users\HP\.ssh\lightning_rsa ssh.lightning.ai
```

### 3. Extract and setup

```bash
tar -xzf dqn_ver3.tar.gz
cd dqn_ver3
chmod +x *.sh scripts/*.sh
bash setup_lightning.sh
```

### 4. Run training (choose one)

**Option A: Full multi-seed run with all figures (~20-25 min)**
```bash
bash run_lightning_multi.sh
```

**Option B: Quick single-seed test (~4-5 min)**
```bash
bash run_lightning_single.sh
```

### 5. Download results

From Windows PowerShell:
```powershell
# Download logs and figures
scp -i C:\Users\HP\.ssh\lightning_rsa -r ssh.lightning.ai:~/dqn_ver3/logs ./
scp -i C:\Users\HP\.ssh\lightning_rsa -r ssh.lightning.ai:~/dqn_ver3/figures ./
```

## Expected Runtime on H200 GPU

- Single seed (100 episodes): ~4-5 minutes
- 5 seeds (100 episodes each): ~20-25 minutes
- Total with setup: ~30 minutes (well within 1 hour)

## Outputs Generated

### JSON/CSV
- `logs/multiseed_summary.json` - Statistical summary (mean±SD, CI)
- `logs/multiseed_episodes.csv` - Per-episode metrics across seeds
- `logs/dqn_steps.csv` - Per-step logs (for calibration)

### Figures (ready for paper)
- `figures/learning_curve_moving_avg_reward.png` - Training dynamics
- `figures/post_content_gain_by_modality.png` - Content effectiveness
- `figures/variance_across_seeds.png` - Stability analysis
- `figures/compute_vs_reward.png` - Efficiency trade-off

## Troubleshooting

If conda activation fails:
```bash
# Use pip in base environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib scipy
```

Quick GPU check:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Custom runs

```bash
# Different number of episodes
EPISODES=150 bash run_lightning_single.sh

# Different seeds
python scripts/run_multiseed.py --seeds 0 1 2 3 4 5 6 7 --episodes 80
```
