# DQN Training for Adaptive Learning Simulator

This is my self-contained simulator plus a baseline DQN (with prioritized replay) for adaptive learning. The spec documents (`spec_*.md`) describe the design in detail.

I mostly run this in two places:
- Local Windows (CPU, quick smoke tests)
- Lightning AI (GPU, production runs)

---

## Quick Start — Local Windows (CPU)

```powershell
# From the repo root
py -3.10 -m venv .venv
./.venv/Scripts/Activate.ps1

# Minimal deps (CPU)
python -m pip install -U pip
pip install numpy matplotlib scipy
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Smoke test: 3 episodes, no warmup
python train_dqn.py --seed 0 --steps 120 --episodes 3 --start-steps 0
```

---

## Production Run — Lightning AI (GPU)

```bash
# In Lightning AI terminal
git clone https://github.com/satsen793/dqn_ver3.git
cd dqn_ver3

python3 -m venv .venv
source .venv/bin/activate

# CUDA 12.1 wheels + essentials
python -m pip install -U pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install numpy matplotlib scipy

# (Optional) verify GPU
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

# Recommended production run (30k steps per seed)
# Note: episodes=300 just provides headroom; --total-steps enforces the exact 30k per seed.
python scripts/run_multiseed.py --seeds 0 1 2 3 4 --episodes 300 --total-steps 30000

# Outputs (see below for details)
ls logs/; ls figures/
```

If you prefer explicit per-seed runs first, then aggregate:

```bash
for s in 0 1 2 3 4; do
  python train_dqn.py --seed "$s" --total-steps 30000 --episodes 300 \
    --out-csv "logs/seed_${s}.csv" --out-json "logs/seed_${s}.json";
done
python scripts/run_multiseed.py --seeds 0 1 2 3 4 --episodes 300 --total-steps 30000
```

---

## What gets saved

- Logs
  - `logs/multiseed_summary.json` — mean/SD/95% CI, medians, per-seed timing
  - `logs/multiseed_episodes.csv` — per-episode metrics across all seeds (18 columns)
- Figures
  - `figures/learning_curve_moving_avg_reward.png`
  - `figures/post_content_gain_by_modality.png`
  - `figures/variance_across_seeds.png`
  - `figures/compute_vs_reward.png`

I keep only the aggregated files in git. Per-seed CSV/JSON are ignored to avoid repo bloat.

---

## Git: save results back to GitHub

```bash
# Stage aggregated outputs
git add logs/multiseed_episodes.csv logs/multiseed_summary.json figures/*.png

# Commit and push
git commit -m "Add aggregated multiseed results and figures"
git push origin main
```

---

## Design cheatsheet (for quick reference)

- State: 30 mastery + frustration + response time = 32 dims
- Actions: 90 question (30×3) + 180 content (30×6) = 270
- IRT 3PL: a∈[0.5, 2.0], b∈[-2.0, 2.0], c∈[0.1, 0.25] by difficulty band
- Reward: correctness (+), mastery_gain (+), frustration (−), post_content (+), engagement (+)
- Termination: mastery ≥ 0.8 or step cap
- Blueprint mix: strict difficulty masking (20/60/20) in exploration and greedy choice

---

## Optional: single-seed CSV/JSON

```bash
python train_dqn.py \
  --seed 0 --steps 140 --episodes 50 \
  --out-csv logs/single_seed_ep.csv \
  --out-json logs/single_seed_results.json
```

CSV columns: episode, return, ttm, total_steps, final_mastery, cumulative_reward, question_accuracy, content_rate, blueprint_adherence, post_content_gain, post_content_gain_video, post_content_gain_PPT, post_content_gain_text, post_content_gain_blog, post_content_gain_article, post_content_gain_handout, mean_frustration.
