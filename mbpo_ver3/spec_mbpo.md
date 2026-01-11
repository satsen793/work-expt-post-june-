# MBPO (Model-Based Policy Optimization) - Full Specification

## Overview

MBPO is a **hybrid model-based + model-free** RL algorithm that combines short model rollouts with off-policy policy optimization (Soft Actor-Critic adapted for discrete actions).

## Algorithm Type
- **Category:** Model-Based RL (hybrid)
- **Inner Optimizer:** Discrete SAC (factorized categorical policy)
- **Model Usage:** Short rollouts for sample augmentation
- **Action Space:** Multi-discrete (factorized)

---

## Core Architecture

### Three Main Components

1. **Dynamics Ensemble** (same as PETS)
   - E=5 probabilistic models: `p_θᵢ(s', r | s, a)`
   - Trained on real experience

2. **Factorized Discrete SAC Agent**
   - Actor: Categorical policies for each action component
   - Critics: Q-functions over joint discrete actions
   - Temperature: Entropy regularization

3. **Replay Mixing**
   - Real buffer: Stores environment transitions
   - Model buffer: Stores synthetic transitions from rollouts
   - Training: Mix both buffers with ratio ρ

---

## Factorized Discrete Action Space

### Action Components

```
a = (d, ℓ, m)
```

Where:
- `d ∈ {0,1,2}`: Difficulty
- `ℓ ∈ {0,...,29}`: Learning Outcome
- `m ∈ {0,...,5}`: Modality (or gate variable)

### Factorized Policy

```
π_ω(a|s) = π_ωd(d|s) · π_ωℓ(ℓ|s) · π_ωm(m|s)
```

Each component is a categorical distribution parameterized by a neural network.

**Joint Log-Probability:**
```
log π_ω(a|s) = log π_ωd(d|s) + log π_ωℓ(ℓ|s) + log π_ωm(m|s)
```

**Entropy Decomposition:**
```
H(π_ω) = H(π_ωd) + H(π_ωℓ) + H(π_ωm)
```

---

## Discrete SAC Formulation

### Soft Q-Function

Learn action-value function with entropy regularization:

```
Q(s,a) = E_π[Σ γ^t (r_t + α·H(π(·|s_t))) | s_0=s, a_0=a]
```

Where α is the temperature parameter.

### Soft Bellman Target

```
y(s,a,r,s') = r + γ(1-done)·V(s')
```

Where soft value:
```
V(s') = E_{a'~π}[min_{i=1,2} Q_φᵢ(s',a') - α log π(a'|s')]
```

**For discrete actions:**
```
V(s') = Σ_a' π(a'|s')[min_{i=1,2} Q_φᵢ(s',a') - α log π(a'|s')]
```

This sum can be computed by enumeration (if |A| small) or approximated by sampling.

### Critic Loss (Twin Q-Networks)

```
L_Q(φᵢ) = E_{(s,a,r,s')~B}[(Q_φᵢ(s,a) - y)²]
```

Where y uses target networks Q_φ̄₁, Q_φ̄₂.

### Actor Loss (Policy Improvement)

```
L_π(ω) = E_{s~B, a~π_ω}[α log π_ω(a|s) - min_{i=1,2} Q_φᵢ(s,a)]
```

**For factorized discrete policy:**
- Sample each component independently
- Joint log-prob is sum of component log-probs
- Entropy bonus naturally decomposes

### Temperature Update (Optional Auto-Tuning)

Target entropy per component:
```
H_target^(d) = -log(1/3) ≈ 1.10
H_target^(ℓ) = -log(1/30) ≈ 3.40
H_target^(m) = -log(1/6) ≈ 1.79
```

Loss:
```
L_α(α) = -α·(H_current - H_target)
```

---

## Model Rollout Procedure

### Short Rollouts from Real States

1. **Sample starting state** s₀ from real replay buffer
2. **Rollout for K steps** (K=1,3,5 typical):
   ```
   For k = 0 to K-1:
       d_k ~ π_ωd(·|s_k)
       ℓ_k ~ π_ωℓ(·|s_k)
       m_k ~ π_ωm(·|s_k)
       a_k ← (d_k, ℓ_k, m_k)
       
       i ~ Uniform{1,...,E}  # Random ensemble member
       s_{k+1} ~ p_θᵢ(·|s_k, a_k)
       r_k ← R(s_k, a_k)  # Known reward function
       
       Store (s_k, a_k, r_k, s_{k+1}) in model buffer
   ```

### Rollout Hyperparameters

```python
ROLLOUT_CONFIG = {
    'rollout_length': 1,        # K (increase to 3-5 for more model reliance)
    'rollout_batch_size': 400,  # M (number of rollouts per training iter)
    'rollout_freq': 1,          # How often to refresh model buffer (every N env steps)
    'model_retain_epochs': 5    # How many epochs to keep synthetic data
}
```

---

## Complete MBPO Algorithm

```
Algorithm: MBPO with Factorized Discrete SAC

Parameters:
    - Ensemble size E = 5
    - Rollout length K = 1
    - Rollout batch size M = 400
    - Mixing ratio ρ = 0.5 (50% real, 50% model)
    - Real collection steps T_real = 1 (collect per step)
    - Dynamics update steps U = 1000 (every new episode)
    - Policy update steps G = 1 (per env step after warmup)
    - Warmup steps = 5000

Initialize:
    - Dynamics ensemble {p_θᵢ}_{i=1}^E
    - Factorized SAC: actor π_ω, critics Q_φ₁, Q_φ₂, target critics Q_φ̄₁, Q_φ̄₂
    - Real buffer B_real ← ∅
    - Model buffer B_model ← ∅
    - Temperature α (or learnable α)

# Phase 1: Warmup (collect initial data)
For step = 1 to warmup_steps:
    s ← env state
    a ~ π_ω(·|s) or random
    s', r, done ← env.step(a)
    B_real.add((s, a, r, s', done))
    If done: s ← env.reset()

# Phase 2: Main training loop
For iteration = 1 to max_iterations:
    
    # (1) Collect real experience
    For t = 1 to T_real:
        s ← current env state
        Sample a = (d,ℓ,m):
            d ~ π_ωd(·|s)
            ℓ ~ π_ωℓ(·|s)
            m ~ π_ωm(·|s)
        
        s', r, done ← env.step(a)
        B_real.add((s, a, r, s', done))
        
        s ← s'
        If done: s ← env.reset()
    
    # (2) Update dynamics ensemble on real buffer
    If iteration % dynamics_update_freq == 0:
        For u = 1 to U:
            batch ← sample(B_real)
            For i = 1 to E:
                # Train model i on (s,a) → (s',r)
                loss_i ← -log p_θᵢ(s', r | s, a)
                θᵢ ← θᵢ - α_model·∇loss_i
    
    # (3) Generate model rollouts (synthetic transitions)
    For m = 1 to M:
        s_0 ~ sample_state(B_real)
        s ← s_0
        
        For k = 0 to K-1:
            # Sample action from current policy
            d ~ π_ωd(·|s)
            ℓ ~ π_ωℓ(·|s)
            m ~ π_ωm(·|s)
            a ← (d, ℓ, m)
            
            # Predict next state (random ensemble)
            i ~ Uniform{1,...,E}
            s' ~ p_θᵢ(·|s, a)
            r ← R(s, a)  # Known reward
            
            B_model.add((s, a, r, s', False))  # done=False for synthetic
            s ← s'
    
    # (4) Policy optimization (Discrete SAC) with mixed replay
    For g = 1 to G:
        # Sample minibatch with mixing ratio ρ
        batch_real ~ sample(B_real, size=ρ·batch_size)
        batch_model ~ sample(B_model, size=(1-ρ)·batch_size)
        batch ← concatenate(batch_real, batch_model)
        
        (s, a, r, s', done) ← batch
        
        # Critic update
        With torch.no_grad():
            # Compute soft value target
            If enumerate_actions:
                V_target = Σ_a' π(a'|s')[min(Q_φ̄₁(s',a'), Q_φ̄₂(s',a')) - α log π(a'|s')]
            Else:
                # Sample-based approximation
                a'_samples ~ π(·|s') [N samples]
                V_target ≈ mean[min(Q_φ̄₁(s',a'), Q_φ̄₂(s',a')) - α log π(a'|s')]
            
            y ← r + γ(1-done)·V_target
        
        L_Q1 ← MSE(Q_φ₁(s,a), y)
        L_Q2 ← MSE(Q_φ₂(s,a), y)
        
        Update φ₁, φ₂ via gradient descent
        
        # Actor update
        Sample a_new = (d,ℓ,m):
            d ~ π_ωd(·|s)
            ℓ ~ π_ωℓ(·|s)
            m ~ π_ωm(·|s)
        
        log_prob ← log π_ω(a_new|s)  # Sum of component log-probs
        q_value ← min(Q_φ₁(s, a_new), Q_φ₂(s, a_new))
        
        L_π ← mean[α·log_prob - q_value]
        
        Update ω via gradient descent
        
        # Temperature update (if auto-tuned)
        If learnable_α:
            H_current ← mean[-log π_ω(a_new|s)]
            L_α ← -α·(H_current - H_target)
            Update α
        
        # Soft update target networks
        φ̄₁ ← τ·φ₁ + (1-τ)·φ̄₁
        φ̄₂ ← τ·φ₂ + (1-τ)·φ̄₂
```

---

## Network Architectures

### Actor: Factorized Categorical Policy

```python
class FactorizedActor(nn.Module):
    def __init__(self, state_dim=32, hidden=256):
        super().__init__()
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Component heads
        self.difficulty_head = nn.Linear(hidden, 3)
        self.lo_head = nn.Linear(hidden, 30)
        self.modality_head = nn.Linear(hidden, 6)
    
    def forward(self, state):
        features = self.shared(state)
        
        logits_d = self.difficulty_head(features)
        logits_l = self.lo_head(features)
        logits_m = self.modality_head(features)
        
        return logits_d, logits_l, logits_m
    
    def sample_action(self, state):
        logits_d, logits_l, logits_m = self.forward(state)
        
        d_dist = Categorical(logits=logits_d)
        l_dist = Categorical(logits=logits_l)
        m_dist = Categorical(logits=logits_m)
        
        d = d_dist.sample()
        l = l_dist.sample()
        m = m_dist.sample()
        
        log_prob = (d_dist.log_prob(d) + 
                    l_dist.log_prob(l) + 
                    m_dist.log_prob(m))
        
        return (d, l, m), log_prob
    
    def log_prob(self, state, action):
        d, l, m = action
        logits_d, logits_l, logits_m = self.forward(state)
        
        log_prob = (F.log_softmax(logits_d, dim=-1)[..., d] +
                    F.log_softmax(logits_l, dim=-1)[..., l] +
                    F.log_softmax(logits_m, dim=-1)[..., m])
        return log_prob
```

### Critic: Q-Network for Joint Actions

**Option 1: Enumerate All Actions (if small)**
```python
class QNetwork(nn.Module):
    def __init__(self, state_dim=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3*30*6)  # 540 Q-values
        )
    
    def forward(self, state):
        return self.net(state).view(-1, 3, 30, 6)
    
    def get_q_value(self, state, action):
        d, l, m = action
        q_all = self.forward(state)
        return q_all[..., d, l, m]
```

**Option 2: Concatenate State + Action Encoding**
```python
class QNetwork(nn.Module):
    def __init__(self, state_dim=32, action_dim=39, hidden=256):
        super().__init__()
        # action_dim = 3+30+6 for factorized one-hot
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, state, action):
        # action: factorized one-hot (3+30+6=39 dims)
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)
```

---

## Hyperparameters

### SAC Agent
```python
SAC_CONFIG = {
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'alpha_lr': 3e-4,
    'discount_gamma': 0.99,
    'target_update_tau': 0.005,
    'initial_temperature': 0.2,
    'auto_tune_temperature': True,
    'target_entropy_scale': 1.0,  # Multiplier for -log(1/|A_component|)
}
```

### Model Training
```python
MODEL_CONFIG = {
    'ensemble_size': 5,
    'hidden_dim': 512,
    'model_lr': 1e-3,
    'model_train_epochs': 1000,
    'model_train_freq': 250,  # Train every N environment steps
    'batch_size': 256
}
```

### Rollout & Mixing
```python
ROLLOUT_CONFIG = {
    'rollout_length': 1,        # K
    'rollout_batch_size': 400,  # M
    'rollout_freq': 1,
    'model_buffer_size': 100000,
    'real_ratio': 0.5           # ρ
}
```

### Training
```python
TRAINING_CONFIG = {
    'warmup_steps': 5000,
    'max_episodes': 500,
    'max_steps_per_episode': 140,
    'updates_per_step': 1,
    'batch_size': 256
}
```

---

## Action Space Handling

### Flattening Factorized Actions

If using joint Q-network (Option 1):
```python
def flatten_action(d, l, m):
    return d * (30 * 6) + l * 6 + m

def unflatten_action(action_id):
    m = action_id % 6
    l = (action_id // 6) % 30
    d = action_id // (30 * 6)
    return d, l, m
```

### Factorized One-Hot Encoding

For critic Option 2:
```python
def encode_action(d, l, m):
    d_onehot = F.one_hot(d, num_classes=3)
    l_onehot = F.one_hot(l, num_classes=30)
    m_onehot = F.one_hot(m, num_classes=6)
    return torch.cat([d_onehot, l_onehot, m_onehot], dim=-1)
```

---

## Soft Value Target Computation

### Enumeration (Small Action Space)

```python
def compute_soft_value_target(critic1, critic2, actor, next_states, alpha):
    # Enumerate all actions
    all_actions = generate_all_actions()  # (540, 3) -> (d,l,m) tuples
    
    probs = []
    q_values = []
    
    for a in all_actions:
        log_prob = actor.log_prob(next_states, a)
        q1 = critic1.get_q_value(next_states, a)
        q2 = critic2.get_q_value(next_states, a)
        
        probs.append(torch.exp(log_prob))
        q_values.append(torch.min(q1, q2) - alpha * log_prob)
    
    probs = torch.stack(probs, dim=-1)
    q_values = torch.stack(q_values, dim=-1)
    
    return (probs * q_values).sum(dim=-1)
```

### Sampling-Based (Large Action Space)

```python
def compute_soft_value_target_sampled(critic1, critic2, actor, next_states, 
                                       alpha, num_samples=10):
    q_samples = []
    
    for _ in range(num_samples):
        a, log_prob = actor.sample_action(next_states)
        
        q1 = critic1.get_q_value(next_states, a)
        q2 = critic2.get_q_value(next_states, a)
        
        q_samples.append(torch.min(q1, q2) - alpha * log_prob)
    
    return torch.stack(q_samples).mean(dim=0)
```

---

## Expected Performance

### Sample Efficiency
- **vs Model-Free:** 20-35% fewer steps to mastery
- **vs PETS:** Similar early, better asymptotic
- **Rollout Length Impact:** K=1 conservative, K=3-5 more aggressive

### Stability
- **Across Seeds:** High (best among all algorithms)
- **Convergence:** Smooth, monotonic improvement
- **Model Quality:** Less sensitive than PETS (short rollouts)

### Typical Metrics (Simulated Data)
- **Time-to-Mastery:** 70 ± 6 steps (S=20 seeds)
- **Cumulative Reward:** 98 ± 8
- **Blueprint Adherence:** 93% (7 ± 2 pp deviation)
- **Wall-Clock Time:** ~40-50 minutes/seed (CPU)

---

## Comparison Summary

| Metric | Rule-Based | DQN | PPO | PETS | MBPO |
|--------|------------|-----|-----|------|------|
| Time-to-Mastery | 120±15 | 95±12 | 87±8 | 75±10 | **70±6** |
| Cumulative Reward | 70±8 | 85±15 | 92±10 | 95±12 | **98±8** |
| Blueprint Adherence | 78% | 92% | 94% | 91% | **93%** |
| Stability (Variance) | Low | High | Medium | Medium | **Low** |
| Compute (min/seed) | <1 | ~45 | ~30 | ~80 | ~45 |

---

## Implementation Checklist

### Core Components
- [ ] Dynamics ensemble (reuse from PETS)
- [ ] Factorized discrete SAC agent
- [ ] Twin Q-networks with target networks
- [ ] Replay buffers (real + model)
- [ ] Mixing mechanism (ρ ratio)

### Rollout Procedure
- [ ] Sample starting states from real buffer
- [ ] K-step rollouts with policy sampling
- [ ] Store in model buffer
- [ ] Periodic refresh

### SAC Updates
- [ ] Critic loss with soft targets
- [ ] Actor loss with factorized log-probs
- [ ] Temperature auto-tuning (optional)
- [ ] Soft target network updates

### Evaluation
- [ ] Time-to-mastery
- [ ] Post-content gain
- [ ] Blueprint adherence
- [ ] Seed variance analysis

---

## Debugging Checklist

### Common Issues

1. **Actor Not Learning**
   - Check critic convergence first
   - Verify log-prob computation (sum of components)
   - Inspect entropy (should be positive)
   - Tune temperature α

2. **Model Overfitting**
   - Reduce rollout length K
   - Increase model training regularization
   - Use ensemble disagreement as early stop

3. **Unstable Training**
   - Reduce critic learning rate
   - Increase target update tau
   - Check gradient norms

4. **Poor Sample Efficiency**
   - Increase rollout batch size M
   - Tune mixing ratio ρ
   - Verify model quality (train longer)

### Monitoring Metrics
- Critic loss (Q1, Q2)
- Actor loss
- Temperature α
- Entropy (per component)
- Model prediction error
- Real vs synthetic return gap

---

## Code Template

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class MBPOAgent:
    def __init__(self, state_dim, config):
        # Dynamics ensemble (reuse PETS)
        self.ensemble = [...]
        
        # SAC components
        self.actor = FactorizedActor(state_dim, config['hidden'])
        self.critic1 = QNetwork(state_dim, config['hidden'])
        self.critic2 = QNetwork(state_dim, config['hidden'])
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        self.actor_optimizer = torch.optim.Adam(...)
        self.critic1_optimizer = torch.optim.Adam(...)
        self.critic2_optimizer = torch.optim.Adam(...)
        
        # Temperature
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], ...)
        
        # Buffers
        self.real_buffer = ReplayBuffer(capacity=100000)
        self.model_buffer = ReplayBuffer(capacity=100000)
        
        self.config = config
    
    def generate_model_rollouts(self, num_rollouts, rollout_length):
        # Sample starting states from real buffer
        # Rollout K steps with policy + dynamics
        # Store in model buffer
        pass
    
    def update_sac(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Critic update
        with torch.no_grad():
            soft_value_target = self.compute_soft_value_target(next_states)
            y = rewards + self.config['gamma'] * (1-dones) * soft_value_target
        
        q1_loss = F.mse_loss(self.critic1(states, actions), y)
        q2_loss = F.mse_loss(self.critic2(states, actions), y)
        
        # ... backprop
        
        # Actor update
        new_actions, log_probs = self.actor.sample_action(states)
        q_values = torch.min(
            self.critic1(states, new_actions),
            self.critic2(states, new_actions)
        )
        
        actor_loss = (self.alpha * log_probs - q_values).mean()
        
        # ... backprop
        
        # Temperature update
        # ... 
        
        # Soft update targets
        self.soft_update(self.critic1, self.critic1_target, tau=0.005)
        self.soft_update(self.critic2, self.critic2_target, tau=0.005)
```

---

## Deliverables for Developer

1. **Dynamics Ensemble** (reuse from PETS)
2. **Factorized Discrete SAC** (actor + twin critics)
3. **Rollout Generator** (short K-step rollouts)
4. **Replay Mixing Mechanism** (ρ ratio)
5. **Training Loop** with model updates + policy updates
6. **Evaluation Script** with all metrics
7. **Config File** with hyperparameters
8. **Logging** (TensorBoard/WandB)
