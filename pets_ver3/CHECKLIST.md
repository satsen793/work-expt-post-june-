# Paper Requirements Checklist

## Required by Elsevier Template

### Figures (All in results/ after running scripts)

- [x] **Figure 1**: `learning_curve.png`
  - Shows PPO learning curve (batch mean reward across seeds)
  - Shaded band for ±1 SD
  - Generated from: `learning_curve_data.json`

- [x] **Figure 2**: `modality_gains.png`
  - Post-content gain by modality (video, PPT, text, blog, article, handout)
  - Mean with SD error bars
  - Generated from: `modality_gains.json`

- [x] **Figure 3**: `calibration.png`
  - Calibration of predicted mastery vs. observed correctness
  - Diagonal line = perfect calibration
  - Generated from: `calibration_data.json`

- [x] **Figure 4**: `variance_bands_all.png`
  - Variance of cumulative reward across random seeds
  - Shows stability/reproducibility
  - Generated from: `variance_data.json`

- [x] **Figure 5**: `time_to_mastery_all.png`
  - Time-to-mastery comparison (bar chart with CI)
  - Lower is better
  - Generated from: `performance_summary.json`

- [x] **Figure 6**: `compute_vs_reward.png`
  - Compute-reward trade-off
  - Wall-clock time vs final reward
  - Generated from: `performance_summary.json`

### Tables (LaTeX fragments in results/)

- [x] **Table 1**: `table_ppo_perf.tex`
  - Performance summary (mean±SD across seeds)
  - Metrics: time-to-mastery, cumulative reward, question accuracy, blueprint adherence, mean frustration, final mastery, wall-clock time
  - Auto-generated LaTeX code for direct inclusion

- [x] **Table 2**: `table_modality.tex`
  - Post-content gain by modality
  - Mean gain and std dev for each modality
  - Auto-generated LaTeX code

## Required Metrics (From spec_evaluation.md)

### Primary Metrics

- [x] **Time-to-Mastery**: Steps to reach 0.8 mean mastery
  - Tracked in: `time_to_mastery` field
  - Reported: Mean ± SD, 95% CI

- [x] **Cumulative Reward**: Total reward per episode
  - Tracked in: `cumulative_reward` field
  - Reported: Mean ± SD, 95% CI

- [x] **Post-Content Gain**: Mastery improvement after content
  - Tracked per modality in: `modality_gains` field
  - Reported: Mean ± SD by modality

- [x] **Blueprint Adherence**: Deviation from 20-60-20 target
  - Tracked in: `blueprint_adherence` field
  - Reported: Percentage (100% = perfect)

- [x] **Policy Stability**: Reward variance across seeds
  - Tracked in: seed-level returns
  - Reported: Variance plot

### Secondary Metrics

- [x] **Question Accuracy**: % correct answers
  - Tracked in: `question_accuracy` field

- [x] **Final Mastery Level**: Mean mastery at episode end
  - Tracked in: `final_mastery` field

- [x] **Mean Frustration**: Average frustration over episode
  - Tracked in: `mean_frustration` field

- [x] **Content Recommendation Rate**: % content actions
  - Tracked in: `content_count` / `total_steps`

- [x] **Calibration Data**: Predicted vs actual correctness
  - Tracked in: `calibration_predicted`, `calibration_actual`

## Statistical Validation (From spec_evaluation.md)

- [x] **Random Seeds**: 5 independent seeds
  - Implemented: (42, 1337, 7, 21, 2024)

- [x] **Paired Design**: Same seeds across runs
  - Implemented: Fixed seed list in TRAIN_CONFIG

- [x] **Bootstrap CI**: 1,000 iterations
  - Implemented in: main() function

- [x] **Mean ± SD Reporting**: All metrics
  - Implemented in: export_results_for_paper()

- [x] **95% Confidence Intervals**: Bootstrap + parametric
  - Implemented in: main() function

## MDP Components (From spec_overview.md)

- [x] **State Space**: 32 dimensions
  - 30 mastery levels + 1 frustration + 1 response_time
  - All values in [0, 1]

- [x] **Action Space**: 270 discrete actions
  - 90 question actions (30 LOs × 3 difficulties)
  - 180 content actions (30 LOs × 6 modalities)

- [x] **Reward Function**: Multi-objective
  - Questions: α·correct + β·Δm - γ_f·f - blueprint_penalty
  - Content: 2.0·Δm + 0.5·(-frustration_delta)
  - Weights: α=1.0, β=0.5, γ_f=0.3

- [x] **Episode Termination**: 3 conditions
  - Mastery threshold (≥0.8)
  - Step limit (140)
  - Critical frustration (≥0.95)

## PETS Algorithm (From spec_pets.md)

- [x] **Ensemble Dynamics**: 5 models
  - Architecture: 3-layer MLP (512 hidden units)
  - Output: Gaussian next-state + reward

- [x] **Factorized CEM**: Multi-discrete actions
  - Components: gate (2), difficulty (3), LO (30), modality (6)
  - Logits per component per horizon step

- [x] **MPC Planning**: Replanning every step
  - Horizon H = 10
  - Iterations J = 5
  - Candidates N = 500
  - Elite fraction = 0.1

- [x] **Hyperparameters**:
  - Learning rate: 1e-3
  - Train epochs: 50
  - Batch size: 256
  - Discount γ: 0.99
  - Update rate η: 0.5

## Simulator Specifications (From spec_simulator.md)

- [x] **Learner Initialization**: Beta(2, 5) distribution
- [x] **IRT Model**: 3-parameter logistic
  - Parameters: a (discrimination), b (difficulty), c (guessing)
- [x] **Question Bank**: 600 questions
  - 20 per LO with 20-60-20 difficulty split
- [x] **Content Repository**: 180 items
  - 6 modalities per LO
  - Modality-specific effectiveness and engagement
- [x] **Mastery Updates**: Stochastic with diminishing returns
- [x] **Frustration Dynamics**: Increases on failure, decreases on success
- [x] **Fail-Streak Remediation**: Content trigger after 3 failures

## Data Export (For Paper)

### JSON Files (results/)
- [x] `learning_curve_data.json` - Episode rewards with mean/std
- [x] `performance_summary.json` - All aggregate metrics
- [x] `modality_gains.json` - Per-modality statistics
- [x] `calibration_data.json` - Predicted vs actual mastery
- [x] `variance_data.json` - Seed-level returns

### LaTeX Tables (results/)
- [x] `table_ppo_perf.tex` - Performance summary table
- [x] `table_modality.tex` - Modality gains table

### PNG Figures (results/)
- [x] `learning_curve.png` - Training dynamics
- [x] `modality_gains.png` - Content effectiveness
- [x] `calibration.png` - Mastery calibration
- [x] `variance_bands_all.png` - Stability analysis
- [x] `time_to_mastery_all.png` - Efficiency comparison
- [x] `compute_vs_reward.png` - Trade-off analysis

## Usage Verification

### Step 1: Run Training
```bash
python pets_train.py
```
Expected output:
- Console: Training progress for 5 seeds × 500 episodes
- Files: All JSON and LaTeX files in results/

### Step 2: Generate Figures
```bash
python generate_figures.py
```
Expected output:
- Console: Figure generation progress
- Files: All 6 PNG figures in results/

### Step 3: Verify Output
```bash
ls -la results/
```
Should contain:
- 5 JSON files
- 2 TEX files
- 6 PNG files

### Step 4: Include in Paper
In `Elsevier_Template.tex`:
```latex
\includegraphics[width=0.9\linewidth]{results/learning_curve.png}
\input{results/table_ppo_perf.tex}
```

## Final Verification

- [x] All spec files reviewed (overview, pets, methodology, simulator, evaluation)
- [x] Elsevier template requirements identified
- [x] Code implements all required metrics
- [x] Export functions generate all required outputs
- [x] Visualization script creates all figures
- [x] Documentation provides complete mapping
- [x] Hyperparameters match specifications
- [x] Statistical validation implemented
- [x] 1:1 correspondence verified

## Status: ✅ COMPLETE

All requirements from .md files and Elsevier template have been implemented in pets_train.py with comprehensive data export and visualization capabilities.
