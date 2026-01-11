# CRITICAL FIXES REQUIRED - Action Items

**Priority:** 🔴 BLOCKING  
**Status:** Must complete before running experiments  
**Estimated Total Time:** 12-16 hours

---

## DECISION REQUIRED (30 minutes)

### **Issue:** Template-Spec Structural Mismatch

The Elsevier template (`Elsevier_Template.tex`) is written as if PPO is the primary/only method:
- Line 64-69: "PPO learning curve"
- Line 78-85: "PPO performance summary"
- Line 93: "PPO" in figure caption
- Line 109: "for PPO" in table caption

However, all 4 spec files (`spec_*.md`) are identical except algorithm-specific sections, suggesting equal treatment of all 4 algorithms.

### **Decision Options:**

#### **Option A: PPO-Primary Paper** (4 hours to implement)
**Structure:**
- Keep template mostly as-is
- Add comparison tables for other algorithms vs PPO baseline
- PPO gets detailed figures; others get summary statistics

**Required Changes:**
1. Add `tab:dqn_vs_ppo` comparison table
2. Add `tab:pets_vs_ppo` comparison table  
3. Add `tab:mbpo_vs_ppo` comparison table
4. Each table includes: TTM Δ, reward Δ, p-value, Cohen's d

**Pros:** Minimal template changes, clear narrative  
**Cons:** Devalues other algorithms, may not match spec intent

---

#### **Option B: All-4-Equal Paper** (8 hours to implement)
**Structure:**
- Restructure template to present all 4 algorithms equally
- Each algorithm gets its own performance summary
- Comparative analysis in separate section

**Required Changes:**
1. Rename `fig:learning_curve` → `fig:learning_curves_all` (show all 4)
2. Expand `tab:perf_summary` to have 4 columns (DQN, PPO, PETS, MBPO)
3. Add algorithm-specific subsections under Results
4. Change captions: "for PPO" → "for all methods"

**Pros:** Matches spec intent, comprehensive comparison  
**Cons:** More work, longer paper

---

### **Recommendation:** **Option B (All-4-Equal)**

**Reasoning:**
1. Specs are written algorithm-agnostically (identical across projects)
2. `shared_config.py` enforces fair comparison (same seeds, episodes, MDP)
3. Research question is "which RL method works best?" not "does PPO work?"
4. Equal treatment provides more value to readers

**Action:** Meet with co-authors, make decision, document in `TEMPLATE_STRUCTURE.md`

---

## FIX #1: Implement AUC Metric (1 hour)

### **Issue:** Table 4 (`tab:dqn_ppo_perf`) requires "AUC (reward) @10k steps" but none of the training scripts compute it.

### **Implementation:**

Add to all 4 training scripts (`train_dqn.py`, `pets_train.py`, `ppo_train.py`, `train_mbpo.py`):

```python
def compute_auc_at_steps(returns, steps_per_episode, target_steps=10000):
    """
    Compute area under reward curve up to target_steps.
    
    Args:
        returns: List[float] - reward per episode
        steps_per_episode: List[int] - steps taken in each episode
        target_steps: int - compute AUC up to this step count
    
    Returns:
        float - AUC value (trapezoidal integration)
    """
    cumulative_steps = np.cumsum(steps_per_episode)
    
    # Find episodes that fit within target_steps
    valid_episodes = cumulative_steps <= target_steps
    
    if not valid_episodes.any():
        return 0.0
    
    # Truncate to target
    episode_indices = np.where(valid_episodes)[0]
    truncated_returns = returns[episode_indices]
    truncated_steps = cumulative_steps[episode_indices]
    
    # Trapezoidal integration
    auc = np.trapz(truncated_returns, truncated_steps)
    
    return float(auc)
```

### **Integration Points:**

**DQN** (`train_dqn.py`):
```python
# After line 1035 (in run_training function):
auc_10k = compute_auc_at_steps(
    returns=episode_returns,
    steps_per_episode=[em["total_steps"] for em in episode_metrics],
    target_steps=10_000
)
results["auc_10k"] = auc_10k
```

**PETS** (`pets_train.py`):
```python
# After line 888 (in main function):
all_seed_auc_10k = []
for seed_returns in all_seed_returns:
    auc = compute_auc_at_steps(seed_returns, [140]*len(seed_returns), 10_000)
    all_seed_auc_10k.append(auc)
# Export: mean ± SD
```

**PPO** (`ppo_train.py`):
```python
# After line 850 (in training loop):
auc_10k = compute_auc_at_steps(all_returns, [len(ep_log) for ep_log in all_episode_logs], 10_000)
```

**MBPO** (`train_mbpo.py`):
```python
# After line 950 (in agent training):
# Similar integration as DQN
```

### **Testing:**
```python
# Test with dummy data
test_returns = [10, 15, 20, 25]
test_steps = [100, 100, 100, 100]
auc = compute_auc_at_steps(test_returns, test_steps, 300)
expected = 0.5 * (10+15)*100 + 0.5 * (15+20)*100 + 0.5 * (20+25)*100  # Should match
assert abs(auc - expected) < 1e-6
```

---

## FIX #2: Add Step-Based Checkpoints (3 hours)

### **Issue:** Table 5 (`tab:budget_perf`) requires metrics at 10k, 25k, 50k steps, but training is episode-based.

### **Implementation Strategy:**

Modify training loops to track cumulative steps and save metrics at checkpoints.

### **Example for DQN** (`train_dqn.py`):

```python
# Add after line 785 (before episode loop):
CHECKPOINTS = [10_000, 25_000, 50_000]
checkpoint_idx = 0
global_steps = 0
checkpoint_metrics = {cp: {} for cp in CHECKPOINTS}

# Inside episode loop (after line 850):
global_steps += step_count  # step_count from current episode

# Check if we passed a checkpoint
while checkpoint_idx < len(CHECKPOINTS) and global_steps >= CHECKPOINTS[checkpoint_idx]:
    cp = CHECKPOINTS[checkpoint_idx]
    
    # Compute metrics at this checkpoint
    checkpoint_metrics[cp] = {
        "global_steps": global_steps,
        "episodes_completed": episode_count,
        "cumulative_reward_mean": np.mean(episode_returns[-20:]),  # last 20 episodes
        "time_to_mastery_mean": np.mean([ttm for ttm in time_to_mastery if ttm is not None]),
        "blueprint_adherence": np.mean([em["blueprint_adherence"] for em in episode_metrics[-20:]]),
        "post_content_gain": np.mean([em["post_content_gain"] for em in episode_metrics[-20:]]),
    }
    
    checkpoint_idx += 1

# After training completes, export:
results["checkpoint_metrics"] = checkpoint_metrics
```

### **Checkpoint JSON Format:**

```json
{
  "checkpoint_metrics": {
    "10000": {
      "global_steps": 10050,
      "episodes_completed": 72,
      "cumulative_reward_mean": 45.3,
      "time_to_mastery_mean": 85.2,
      "blueprint_adherence": 92.5,
      "post_content_gain": 0.08
    },
    "25000": { ... },
    "50000": { ... }
  }
}
```

### **LaTeX Table Generation:**

Create `generate_budget_table.py`:
```python
import json

def generate_budget_table(dqn_json, ppo_json, pets_json, mbpo_json):
    checkpoints = [10_000, 25_000, 50_000]
    
    latex = r"""\begin{tabular}{lccc}
\toprule
Method & 10k steps & 25k steps & 50k steps \\
\midrule
"""
    
    for algo, json_path in [("DQN", dqn_json), ("PPO", ppo_json), ("PETS", pets_json), ("MBPO", mbpo_json)]:
        with open(json_path) as f:
            data = json.load(f)
        
        row = f"{algo}"
        for cp in checkpoints:
            reward = data["checkpoint_metrics"][str(cp)]["cumulative_reward_mean"]
            row += f" & {reward:.2f}"
        row += r" \\"
        latex += row + "\n"
    
    latex += r"""\bottomrule
\end{tabular}"""
    
    with open("results/table_budget_perf.tex", "w") as f:
        f.write(latex)
```

---

## FIX #3: Clarify Calibration Scope (30 minutes)

### **Issue:** Template Fig 3 (`fig:calibration`) expects calibration data, but only PETS/MBPO maintain predictive state models. DQN/PPO don't predict mastery probabilities.

### **Solution:** Add footnote to template

**Insert after line 109** in `Elsevier_Template.tex`:

```latex
\subsection{Trustworthiness of mastery estimates}
The calibration curve in Figure~\ref{fig:calibration} compares predicted mastery with empirical correctness; proximity to the diagonal indicates better calibration.

% ADD THIS NOTE:
\textbf{Note:} Calibration is reported for model-based methods (PETS, MBPO) only, as they explicitly maintain predictive state estimates. Model-free baselines (DQN, PPO) do not output calibrated mastery probabilities and are thus omitted from this analysis.

% ------------------- Figure: Calibration curve -------------------
\begin{figure}[t]
  \centering
  \includegraphics[width=0.7\linewidth]{results/calibration.png}
  \caption{Calibration of predicted mastery vs. observed correctness (PETS and MBPO).}
  \label{fig:calibration}
\end{figure}
```

**Alternative:** Add calibration to DQN/PPO by tracking per-LO correctness rates:

```python
# In DQN/PPO training scripts:
class SimpleMasteryEstimator:
    def __init__(self, num_los=30, window=10):
        self.num_los = num_los
        self.window = window
        self.correctness_history = [[] for _ in range(num_los)]
    
    def update(self, lo, correct):
        self.correctness_history[lo].append(float(correct))
        if len(self.correctness_history[lo]) > self.window:
            self.correctness_history[lo].pop(0)
    
    def estimate_mastery(self, lo):
        if not self.correctness_history[lo]:
            return 0.5  # default
        return np.mean(self.correctness_history[lo])
    
    def get_calibration_data(self):
        predicted = []
        actual = []
        for lo in range(self.num_los):
            if self.correctness_history[lo]:
                pred = self.estimate_mastery(lo)
                act = self.correctness_history[lo][-1]  # most recent
                predicted.append(pred)
                actual.append(act)
        return predicted, actual
```

---

## FIX #4: Standardize Wall-Clock Timing (1 hour)

### **Issue:** PETS exports `wall_clock_mean_s`, but DQN/PPO don't explicitly track/export timing.

### **Implementation:**

Add to **all 4 scripts**:

```python
import time

# At start of training:
training_start_time = time.time()
episode_durations = []

# Inside episode loop:
episode_start = time.time()
# ... run episode ...
episode_durations.append(time.time() - episode_start)

# After training:
total_wall_clock = time.time() - training_start_time

results["timing"] = {
    "total_wall_clock_s": total_wall_clock,
    "mean_episode_duration_s": np.mean(episode_durations),
    "std_episode_duration_s": np.std(episode_durations),
}
```

### **Integration:**

**DQN** (`train_dqn.py`):
- Add timing dict to results (after line 1050)

**PPO** (`ppo_train.py`):
- Add timing dict to results (after line 900)

**MBPO** (`train_mbpo.py`):
- Add timing dict to results (after line 990)

**PETS** (`pets_train.py`):
- Already has timing; ensure format matches others

---

## FIX #5: Standardize Bootstrap CI (1 hour)

### **Issue:** DQN has robust bootstrap code (lines 1013-1022), but PETS/PPO compute manually.

### **Solution:** Extract to shared utility

Create `stats_utils.py`:

```python
import numpy as np
from typing import List, Callable, Tuple

def bootstrap_ci(
    data: List[float],
    statistic_fn: Callable = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Sample data
        statistic_fn: Function to compute statistic (default: mean)
        confidence: CI level (default: 0.95)
        n_bootstrap: Number of bootstrap samples (default: 1000)
    
    Returns:
        (lower_bound, upper_bound)
    """
    stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stats.append(statistic_fn(sample))
    
    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100
    
    lower = np.percentile(stats, lower_pct)
    upper = np.percentile(stats, upper_pct)
    
    return float(lower), float(upper)
```

### **Usage in all scripts:**

```python
from stats_utils import bootstrap_ci

# Compute CI for time-to-mastery
ttm_ci_lower, ttm_ci_upper = bootstrap_ci(time_to_mastery_across_seeds)

# Compute CI for cumulative reward
reward_ci_lower, reward_ci_upper = bootstrap_ci(cumulative_rewards_across_seeds)
```

---

## FIX #6: Unify JSON Output Schema (1 hour)

### **Issue:** Each script exports different JSON keys, making cross-algorithm comparison difficult.

### **Solution:** Define canonical schema in `OUTPUT_SCHEMA.md`

```markdown
# Canonical Output Schema

All training scripts must export JSON with the following structure:

```json
{
  "metadata": {
    "algorithm": "DQN|PPO|PETS|MBPO",
    "seed": 0,
    "episodes": 295,
    "total_steps": 29500,
    "timestamp": "2026-01-11T12:00:00Z"
  },
  "episode_metrics": [
    {
      "episode": 1,
      "total_steps": 105,
      "final_mastery": 0.45,
      "cumulative_reward": 52.3,
      "question_accuracy": 0.62,
      "content_rate": 0.35,
      "blueprint_adherence": 89.5,
      "post_content_gain": 0.08,
      "mean_frustration": 0.25,
      "time_to_mastery": null
    }
    // ... 295 entries
  ],
  "seed_summary": {
    "cumulative_reward": {"mean": 55.2, "std": 3.1, "ci_lower": 54.0, "ci_upper": 56.4},
    "time_to_mastery": {"mean": 87.5, "std": 8.2, "ci_lower": 85.0, "ci_upper": 90.0},
    "blueprint_adherence": {"mean": 92.3, "std": 2.1},
    "post_content_gain": {"mean": 0.09, "std": 0.01}
  },
  "modality_gains": {
    "video": {"mean": 0.12, "std": 0.02, "count": 450},
    "PPT": {"mean": 0.10, "std": 0.02, "count": 380},
    "text": {"mean": 0.07, "std": 0.01, "count": 320},
    "blog": {"mean": 0.08, "std": 0.02, "count": 290},
    "article": {"mean": 0.07, "std": 0.01, "count": 310},
    "handout": {"mean": 0.07, "std": 0.01, "count": 300}
  },
  "calibration_data": {
    "predicted_mastery": [0.3, 0.4, ...],
    "empirical_correct": [0.0, 1.0, ...]
  },
  "timing": {
    "total_wall_clock_s": 7200.5,
    "mean_episode_duration_s": 24.3,
    "std_episode_duration_s": 3.2
  },
  "checkpoint_metrics": {
    "10000": { "cumulative_reward_mean": 45.3, ... },
    "25000": { ... },
    "50000": { ... }
  },
  "auc_10k": 450.2
}
```
```

### **Validation Script:**

Create `validate_output.py`:
```python
import json
import sys

REQUIRED_KEYS = [
    "metadata", "episode_metrics", "seed_summary", 
    "modality_gains", "timing", "checkpoint_metrics", "auc_10k"
]

def validate_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    missing = [k for k in REQUIRED_KEYS if k not in data]
    
    if missing:
        print(f"❌ Missing keys: {missing}")
        return False
    
    print(f"✅ {filepath} passes validation")
    return True

if __name__ == "__main__":
    validate_json(sys.argv[1])
```

---

## TESTING CHECKLIST

Before running full experiments, verify:

- [ ] **Decision:** Option A or B documented
- [ ] **Template:** Updated with chosen structure
- [ ] **AUC:** Function implemented and tested in all 4 scripts
- [ ] **Checkpoints:** Tracking at 10k/25k/50k in all 4 scripts
- [ ] **Calibration:** Note added to template OR simple estimator added to DQN/PPO
- [ ] **Timing:** Wall-clock exported in all 4 scripts
- [ ] **Bootstrap CI:** Shared utility imported in all 4 scripts
- [ ] **JSON Schema:** All scripts export canonical format
- [ ] **Validation:** `validate_output.py` passes for all 4 algorithms
- [ ] **Smoke Test:** 2 seeds × 20 episodes runs without errors
- [ ] **Output Files:** All expected JSON/PNG/TEX files generated
- [ ] **LaTeX Compilation:** Template compiles with generated tables

---

## ESTIMATED TIME BREAKDOWN

| Task | Time |
|------|------|
| Decision + meeting | 0.5h |
| Template update (Option B) | 6h |
| Implement AUC | 1h |
| Add checkpoints | 3h |
| Clarify calibration | 0.5h |
| Standardize timing | 1h |
| Unify Bootstrap CI | 1h |
| Define JSON schema | 0.5h |
| Create validation script | 0.5h |
| Testing & debugging | 2h |
| **TOTAL** | **16h** |

---

## NEXT STEPS AFTER FIXES

1. **Run Smoke Test** (2 hours)
   ```bash
   python train_dqn.py --seed 0 --episodes 20 --out-json results/dqn_smoke_s0.json
   python pets_train.py --seed 0 --steps 20
   python ppo_train.py --seed 0 --episodes 20
   python train_mbpo.py --seed 0 --episodes 20
   ```

2. **Validate Outputs**
   ```bash
   python validate_output.py results/dqn_smoke_s0.json
   python validate_output.py results/pets_smoke_s0.json
   python validate_output.py results/ppo_smoke_s0.json
   python validate_output.py results/mbpo_smoke_s0.json
   ```

3. **Generate Test Figures**
   ```bash
   python generate_figures.py --input results/
   ```

4. **Compile LaTeX**
   ```bash
   pdflatex pets_ver3/Elsevier_Template.tex
   ```

5. **If smoke test passes → Run full experiments**

---

**End of Critical Fixes Document**  
**Prepared by:** AI Audit System  
**Date:** 2026-01-11
