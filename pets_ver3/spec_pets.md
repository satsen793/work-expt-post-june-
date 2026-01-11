# PETS (Probabilistic Ensembles with Trajectory Sampling) - Full Specification

## Overview

PETS is a **model-based** reinforcement learning algorithm that learns an ensemble of probabilistic dynamics models and uses Model Predictive Control (MPC) with the Cross-Entropy Method (CEM) for planning.

## Algorithm Type
- **Category:** Model-Based RL
- **Planning Method:** MPC with CEM optimization
- **Action Space:** Adapted for multi-discrete (factorized categorical)
- **Uncertainty Handling:** Ensemble disagreement

---

## Core Components

### 1. Probabilistic Dynamics Ensemble

Learn E models of environment dynamics:
```
p_θᵢ(s', r | s, a)  for i = 1, ..., E
```

**Model Architecture (per ensemble member):**
```
Input: [state, action]  → concatenated vector
    ↓
Dense(512, ReLU)
    ↓
Dense(512, ReLU)
    ↓
Dense(512, ReLU)
    ↓
Output: [next_state_mean, next_state_logvar, reward]
```

**Probabilistic Output:**
- Next state: Gaussian `N(μ_θᵢ(s,a), Σ_θᵢ(s,a))`
- Reward: Deterministic or Gaussian (usually deterministic for this domain)

### 2. Action Factorization for Discrete Actions

Unlike continuous PETS, we use **factorized categorical actions**:

```
a_t = (d_t, ℓ_t, m_t)
```

Where:
- `d_t ∈ {0,1,2}`: Difficulty (Easy, Medium, Hard)
- `ℓ_t ∈ {0,...,29}`: Learning Outcome index
- `m_t ∈ {0,...,5}`: Modality (or question/content indicator)

**Joint Distribution:**
```
π(a_t | s_t) = π_d(d_t | s_t) · π_ℓ(ℓ_t | s_t) · π_m(m_t | s_t)
```

**Independence Assumption:** For planning efficiency, assume components are conditionally independent given state.

---

## MPC with Factorized Categorical CEM

### Planning Horizon

At each environment step t, plan a length-H action sequence:
```
{a_t, a_{t+1}, ..., a_{t+H-1}}
```

Execute only `a_t`, observe real next state, replan at t+1.

**Typical H:** 5-15 steps

### CEM Planning Variables

Maintain categorical logits for each action component at each horizon step:

```
{φ_h^(d), φ_h^(ℓ), φ_h^(m)}  for h = 0, ..., H-1
```

Each `φ_h^(component)` is a vector of unnormalized logits.

**Example:**
- `φ_h^(d)`: shape (3,) for 3 difficulty levels
- `φ_h^(ℓ)`: shape (30,) for 30 LOs
- `φ_h^(m)`: shape (6,) for 6 modalities

### CEM Iterations

For J iterations (typically J=5-10):

#### Step 1: Sample N Candidate Sequences

For each candidate n=1,...,N:
- For each horizon h=0,...,H-1:
  ```
  d_h^(n) ~ Categorical(softmax(φ_h^(d)))
  ℓ_h^(n) ~ Categorical(softmax(φ_h^(ℓ)))
  m_h^(n) ~ Categorical(softmax(φ_h^(m)))
  ```

**Typical N:** 500-1000 candidates per iteration

#### Step 2: Evaluate Each Candidate Sequence

For each sampled sequence:
```
J̃^(n) = Σ_{h=0}^{H-1} γ^h · r̂_h
```

Where predicted reward and next state come from ensemble rollout:
1. Start from current real state `s_0 = s_t`
2. For each step h:
   - Randomly select ensemble member i ~ Uniform{1,...,E}
   - Sample next state: `s_{h+1}^(n) ~ p_θᵢ(· | s_h^(n), a_h^(n))`
   - Compute reward: `r_h = R(s_h^(n), a_h^(n))` (known reward function)

**Uncertainty Penalty (Optional):**
```
J̃^(n) = Σ_{h=0}^{H-1} γ^h · (r̂_h - λ_u · σ_h)
```

Where `σ_h` is ensemble disagreement (std dev across ensemble predictions).

#### Step 3: Select Elite Sequences

Choose top K sequences by return:
```
Elite = {n : J̃^(n) in top K}
```

**Typical K:** 10% of N (e.g., K=50 if N=500)

#### Step 4: Update Logits with Elite Frequencies

For each component and horizon step, compute elite frequency:
```
f_h^(d)(x) = (1/K) · Σ_{n∈Elite} 1[d_h^(n) = x]
```

Update logits (exponential moving average):
```
φ_h^(d) ← (1-η)·φ_h^(d) + η·log(f_h^(d) + ε)
```

Where:
- `η ∈ (0,1]`: Update rate (typical: 0.5-1.0)
- `ε > 0`: Smoothing constant (e.g., 1e-6)

Repeat for `φ_h^(ℓ)` and `φ_h^(m)`.

---

## Complete PETS Algorithm

```
Algorithm: PETS with Factorized Categorical CEM

Parameters:
    - Ensemble size E = 5
    - Planning horizon H = 10
    - CEM iterations J = 5
    - Candidates per iteration N = 500
    - Elite count K = 50
    - CEM update rate η = 0.5
    - Smoothing ε = 1e-6
    - Discount γ = 0.99
    - Initial exploration episodes: 5

Initialize:
    - Dynamics ensemble {p_θᵢ}_{i=1}^E with random weights
    - Dataset D ← ∅

# Phase 1: Random exploration to seed dataset
For episode = 1 to initial_exploration:
    s ← env.reset()
    For t = 1 to T:
        a ← random action
        s', r, done ← env.step(a)
        D.add((s, a, r, s'))
        s ← s'
        If done: break

# Phase 2: Iterative model learning + MPC
For episode = initial_exploration+1 to max_episodes:
    
    # Train ensemble on dataset
    For epoch = 1 to model_train_epochs:
        For batch in minibatches(D):
            For i = 1 to E:
                # Compute NLL loss for ensemble member i
                loss_i ← -log p_θᵢ(s', r | s, a)
                θᵢ ← θᵢ - α·∇loss_i
    
    # Collect episode using MPC
    s ← env.reset()
    For t = 1 to T:
        
        # MPC planning with factorized CEM
        Initialize logits {φ_h^(d), φ_h^(ℓ), φ_h^(m)}_{h=0}^{H-1} ← 0
        
        For j = 1 to J:  # CEM iterations
            # Sample N candidate sequences
            sequences ← []
            returns ← []
            For n = 1 to N:
                s_0 ← s
                J̃ ← 0
                For h = 0 to H-1:
                    # Sample action components
                    d_h ~ Cat(softmax(φ_h^(d)))
                    ℓ_h ~ Cat(softmax(φ_h^(ℓ)))
                    m_h ~ Cat(softmax(φ_h^(m)))
                    a_h ← (d_h, ℓ_h, m_h)
                    
                    # Predict next state (random ensemble member)
                    i ~ Uniform{1,...,E}
                    s_{h+1} ~ p_θᵢ(· | s_h, a_h)
                    
                    # Accumulate return
                    r_h ← R(s_h, a_h)
                    J̃ ← J̃ + γ^h · r_h
                    
                    s_h ← s_{h+1}
                
                sequences.append({d_0,...,d_{H-1}, ℓ_0,..., m_{H-1}})
                returns.append(J̃)
            
            # Select elites
            Elite ← indices of top K returns
            
            # Update logits based on elite frequencies
            For h = 0 to H-1:
                For each component c ∈ {d, ℓ, m}:
                    f_h^(c)(x) ← frequency of x in elite sequences at step h
                    φ_h^(c) ← (1-η)·φ_h^(c) + η·log(f_h^(c) + ε)
        
        # Execute first action from best sequence
        n* ← argmax_n returns[n]
        a_t ← first action from sequences[n*]
        
        # Real environment step
        s', r, done ← env.step(a_t)
        D.add((s, a_t, r, s'))
        
        s ← s'
        If done: break
```

---

## Dynamics Model Details

### Network Architecture (per ensemble member)

```python
class EnsembleMember(nn.Module):
    def __init__(self, state_dim=32, action_dim=270, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # Output: next_state_mean, next_state_logvar, reward
        self.state_mean = nn.Linear(hidden, state_dim)
        self.state_logvar = nn.Linear(hidden, state_dim)
        self.reward_head = nn.Linear(hidden, 1)
    
    def forward(self, state, action):
        # Action encoding: one-hot or embedding
        action_onehot = F.one_hot(action, num_classes=270).float()
        x = torch.cat([state, action_onehot], dim=-1)
        
        features = self.net(x)
        
        next_state_mean = self.state_mean(features)
        next_state_logvar = self.state_logvar(features)
        reward = self.reward_head(features)
        
        return next_state_mean, next_state_logvar, reward
```

### Loss Function

Negative log-likelihood for Gaussian predictions:

```
L_i = -Σ_batch log N(s' | μ_θᵢ(s,a), Σ_θᵢ(s,a)) + (r - r̂_θᵢ(s,a))²
```

Simplified:
```
L_i = Σ (
    0.5·log_var + 0.5·((s' - mean)² / exp(log_var))
) + MSE(r, r̂)
```

### Training Procedure

```python
def train_ensemble(ensemble, dataset, epochs=50, batch_size=256):
    for epoch in range(epochs):
        indices = np.random.permutation(len(dataset))
        
        for start in range(0, len(dataset), batch_size):
            batch_indices = indices[start:start+batch_size]
            states, actions, rewards, next_states = dataset[batch_indices]
            
            for i, model in enumerate(ensemble):
                # Forward pass
                pred_mean, pred_logvar, pred_reward = model(states, actions)
                
                # NLL loss for next state
                inv_var = torch.exp(-pred_logvar)
                state_loss = 0.5 * (
                    pred_logvar + 
                    (next_states - pred_mean)**2 * inv_var
                ).mean()
                
                # MSE loss for reward
                reward_loss = F.mse_loss(pred_reward, rewards)
                
                total_loss = state_loss + reward_loss
                
                # Backprop
                optimizer[i].zero_grad()
                total_loss.backward()
                optimizer[i].step()
```

---

## Action Encoding for Dynamics Model

### Option 1: Flattened One-Hot (Simple)

```python
action_id = flatten_action(difficulty, lo, modality)
action_onehot = F.one_hot(action_id, num_classes=270)
```

**Pros:** Simple, no learned parameters
**Cons:** High-dimensional (270-dim vector)

### Option 2: Factorized One-Hot (Recommended)

```python
d_onehot = F.one_hot(difficulty, num_classes=3)    # (3,)
l_onehot = F.one_hot(lo, num_classes=30)           # (30,)
m_onehot = F.one_hot(modality, num_classes=6)      # (6,)
action_encoding = torch.cat([d_onehot, l_onehot, m_onehot])  # (39,)
```

**Pros:** Lower-dimensional, preserves structure
**Cons:** Assumes independence (reasonable for planning)

### Option 3: Learned Embeddings

```python
self.difficulty_embed = nn.Embedding(3, 16)
self.lo_embed = nn.Embedding(30, 32)
self.modality_embed = nn.Embedding(6, 16)

d_emb = self.difficulty_embed(difficulty)
l_emb = self.lo_embed(lo)
m_emb = self.modality_embed(modality)
action_encoding = torch.cat([d_emb, l_emb, m_emb])  # (64,)
```

**Pros:** Learnable semantic structure
**Cons:** More parameters, requires tuning

---

## Hyperparameters

### Model Learning
```python
MODEL_CONFIG = {
    'ensemble_size': 5,
    'hidden_dim': 512,
    'learning_rate': 1e-3,
    'train_epochs': 50,
    'batch_size': 256,
    'weight_decay': 1e-5,
    'initial_exploration_episodes': 5
}
```

### MPC Planning
```python
MPC_CONFIG = {
    'horizon': 10,
    'cem_iterations': 5,
    'num_candidates': 500,
    'num_elites': 50,
    'cem_update_rate': 0.5,
    'smoothing_epsilon': 1e-6,
    'discount_gamma': 0.99,
    'uncertainty_penalty': 0.0  # Optional
}
```

### Episode Settings
```python
EPISODE_CONFIG = {
    'max_episodes': 500,
    'max_steps_per_episode': 140
}
```

---

## Uncertainty Quantification

### Ensemble Disagreement

Measure prediction uncertainty as ensemble variance:

```python
def ensemble_uncertainty(ensemble, state, action):
    predictions = []
    for model in ensemble:
        pred_mean, _, _ = model(state, action)
        predictions.append(pred_mean)
    
    predictions = torch.stack(predictions)  # (E, state_dim)
    uncertainty = predictions.std(dim=0).mean()  # Scalar
    return uncertainty
```

### Uncertainty-Aware Planning (Optional)

Penalize uncertain trajectories:
```
J̃^(n) = Σ_{h=0}^{H-1} γ^h · (r̂_h - λ_u · σ_h)
```

Typical `λ_u`: 0.1-1.0

---

## Expected Performance

### Sample Efficiency
- **Early Learning (0-10 episodes):** Very strong due to planning
- **Time-to-Mastery:** 75 ± 10 steps (20-30% better than model-free)
- **Compute Cost:** Higher per step (MPC overhead)

### Stability
- **Across Seeds:** Moderate variance (ensemble helps)
- **Convergence:** Smooth if model quality is good

### Typical Metrics (Simulated Data)
- **Time-to-Mastery:** 75 ± 10 steps
- **Cumulative Reward:** 95 ± 12
- **Blueprint Adherence:** 91% (9 ± 3 pp deviation)
- **Wall-Clock Time:** ~60-90 minutes/seed (CPU)

---

## Comparison Expectations

### vs Model-Free (DQN, PPO)
- **Sample Efficiency:** PETS ~30% fewer steps
- **Early Performance:** PETS much stronger (0-20 episodes)
- **Asymptotic Performance:** Similar final reward
- **Compute:** PETS 2-3x slower per episode

### vs MBPO
- **Sample Efficiency:** PETS better early, MBPO better mid-late
- **Stability:** MBPO typically more stable
- **Compute:** PETS higher (CEM overhead)

---

## Implementation Checklist

### Core Components
- [ ] Probabilistic ensemble dynamics models (E=5)
- [ ] Factorized categorical CEM planner
- [ ] MPC loop with replanning every step
- [ ] Dataset collection and storage
- [ ] Ensemble training procedure

### Planning
- [ ] Logit initialization (zeros or warm-start)
- [ ] Candidate sampling (categorical)
- [ ] Elite selection (top-K)
- [ ] Logit update with smoothing

### Evaluation
- [ ] Time-to-mastery tracking
- [ ] Blueprint adherence
- [ ] Compute time logging
- [ ] Model calibration metrics

---

## Debugging Checklist

### Common Issues

1. **Poor Model Quality**
   - Check dataset diversity (enough exploration?)
   - Verify loss convergence
   - Inspect ensemble disagreement (should be moderate)

2. **CEM Not Converging**
   - Increase J (iterations) or N (candidates)
   - Tune η (update rate)
   - Check logit initialization

3. **Planning Too Slow**
   - Reduce N or H
   - Batch ensemble predictions
   - Use GPU for dynamics models

4. **High Variance Across Seeds**
   - Increase ensemble size E
   - More initial exploration episodes
   - Check model overfitting

### Monitoring Metrics
- Model train/val loss
- Ensemble disagreement
- Planned vs actual returns
- CEM convergence (elite return improvement per iteration)
- State prediction error

---

## Code Template

```python
import torch
import torch.nn as nn
import numpy as np

class PETSAgent:
    def __init__(self, state_dim, action_dim, config):
        self.ensemble = [
            EnsembleMember(state_dim, action_dim, config['hidden_dim'])
            for _ in range(config['ensemble_size'])
        ]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            for model in self.ensemble
        ]
        
        self.dataset = []
        self.config = config
        
        # Action space structure
        self.num_difficulties = 3
        self.num_los = 30
        self.num_modalities = 6
    
    def train_dynamics_models(self, epochs=50):
        # Train each ensemble member on dataset
        for epoch in range(epochs):
            # Sample minibatches and update each model
            pass
    
    def plan_action_cem(self, state):
        H = self.config['horizon']
        J = self.config['cem_iterations']
        N = self.config['num_candidates']
        K = self.config['num_elites']
        
        # Initialize logits (zeros)
        logits_d = [np.zeros(self.num_difficulties) for _ in range(H)]
        logits_l = [np.zeros(self.num_los) for _ in range(H)]
        logits_m = [np.zeros(self.num_modalities) for _ in range(H)]
        
        for iteration in range(J):
            # Sample N candidates
            sequences = []
            returns = []
            
            for n in range(N):
                seq_return = self.evaluate_sequence(
                    state, logits_d, logits_l, logits_m
                )
                returns.append(seq_return)
            
            # Select elites
            elite_indices = np.argsort(returns)[-K:]
            
            # Update logits based on elite frequencies
            # ...
        
        # Return best first action
        # ...
    
    def evaluate_sequence(self, start_state, logits_d, logits_l, logits_m):
        # Sample action sequence from logits
        # Rollout through ensemble
        # Return predicted cumulative reward
        pass
    
    def collect_episode(self):
        state = self.env.reset()
        episode_data = []
        
        for t in range(self.config['max_steps']):
            action = self.plan_action_cem(state)
            next_state, reward, done = self.env.step(action)
            
            episode_data.append((state, action, reward, next_state))
            self.dataset.append((state, action, reward, next_state))
            
            state = next_state
            if done:
                break
        
        return episode_data
```

---

## Deliverables for Developer

1. **Ensemble Dynamics Models** (5 members)
2. **CEM Planner** for factorized categorical actions
3. **MPC Loop** with replanning
4. **Dataset Management** (collection + storage)
5. **Training Script** for ensemble
6. **Evaluation Script** with metrics
7. **Config File** with all hyperparameters
