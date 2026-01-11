# DQN with Prioritized Experience Replay (PER) - Full Specification

## Overview

Deep Q-Network (DQN) is a **value-based, model-free** RL algorithm that learns an approximation to the optimal action-value function Q*(s,a) using a neural network.

## Algorithm Type
- **Category:** Model-Free, Off-Policy, Value-Based
- **Policy:** Implicit (greedy/ε-greedy over learned Q-values)
- **Action Space:** Discrete (unified gated action space)

---

## Mathematical Foundation

### Optimal Action-Value Function

DQN learns Q_φ(s,a) to approximate:

```
Q*(s,a) = E[r + γ max_{a'} Q*(s', a') | s, a]
```

### One-Step TD Loss

```
L_DQN(φ) = E_{(s,a,r,s')~D} [w_i · (r + γ(1-1_term)·max_{a'} Q_φ̄(s',a') - Q_φ(s,a))²]
```

Where:
- `D`: Replay buffer
- `φ`: Current Q-network parameters
- `φ̄`: Target network parameters (updated slowly)
- `1_term`: Terminal state indicator (1 if terminal, 0 otherwise)
- `w_i`: Importance sampling weight from PER

### Bellman Target

```
y_i = r + γ(1 - 1_term) · max_{a'} Q_φ̄(s', a')
```

For terminal states: `y_i = r` (no bootstrapping)

---

## Key Components

### 1. Q-Network Architecture

**Input:** State vector s_t (dimension: 32 for K=30 LOs + 2 engagement vars)

**Output:** Q-values for all actions (dimension: 270)
- Actions 0-89: Question actions (30 LOs × 3 difficulties)
- Actions 90-269: Content actions (30 LOs × 6 modalities)

**Recommended Architecture:**
```
Input Layer: (32,)
    ↓
Dense(256, ReLU)
    ↓
Dense(256, ReLU)
    ↓
Dense(128, ReLU)
    ↓
Output Layer: (270,) [Q-values]
```

**Alternative (Dueling DQN):**
```
Shared:
    Dense(256, ReLU)
    Dense(256, ReLU)
    
Value Stream:           Advantage Stream:
Dense(128, ReLU)        Dense(128, ReLU)
Dense(1)                Dense(270)

Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
```

### 2. Target Network (φ̄)

- **Purpose:** Stabilize training by providing consistent targets
- **Update Rule:** Soft update (Polyak averaging):
  ```
  φ̄ ← τ·φ + (1-τ)·φ̄
  ```
  Typical τ = 0.005 (update every step) or hard copy every N steps

### 3. Replay Buffer (D)

Standard experience replay:
- **Capacity:** 100,000 transitions
- **Storage:** (s, a, r, s', done) tuples
- **Sampling:** Uniform random or prioritized

### 4. Prioritized Experience Replay (PER)

**Core Idea:** Sample transitions with probability proportional to TD error magnitude.

**Priority Assignment:**
```
p_i = |δ_i|^α_PER + ε_small
```

Where:
- `δ_i = r + γ max_{a'} Q_φ̄(s',a') - Q_φ(s,a)` (TD error)
- `α_PER`: Priority exponent (0 = uniform, 1 = full prioritization)
- `ε_small`: Small constant (e.g., 1e-6) to ensure non-zero probability

**Sampling Probability:**
```
P(i) = p_i / Σ_j p_j
```

**Importance Sampling Weights (Bias Correction):**
```
w_i = (N · P(i))^{-β_PER}
```

Then normalize: `w_i ← w_i / max_j w_j`

Where:
- `N`: Buffer size
- `β_PER`: Importance-sampling exponent (annealed from β_start to 1.0)
- Corrects bias introduced by non-uniform sampling

**Hyperparameters:**
- `α_PER`: 0.6 (priority exponent)
- `β_PER`: 0.4 → 1.0 (annealed over training)
- `ε_small`: 1e-6

---

## Action Selection

### Training: ε-Greedy Exploration

```
a_t = {
    argmax_{a∈A} Q_φ(s_t, a)    with probability 1-ε
    Uniform(A)                   with probability ε
}
```

**ε-Decay Schedule:**
```
ε(t) = ε_end + (ε_start - ε_end) · exp(-t / ε_decay)
```

Typical values:
- `ε_start`: 1.0
- `ε_end`: 0.01
- `ε_decay`: 10,000 steps

### Evaluation: Greedy Policy

```
a_t = argmax_{a∈A} Q_φ(s_t, a)
```

### Action Masking (Optional)

If certain actions are invalid in state s:
```
Q_masked(s, a) = {
    Q_φ(s, a)    if a is valid
    -∞           if a is invalid
}
```

Then: `a_t = argmax_{a∈A} Q_masked(s_t, a)`

---

## Training Algorithm

### Complete DQN-PER Algorithm

```
Algorithm: DQN with Prioritized Experience Replay

Parameters:
    - Learning rate α = 3e-4
    - Discount γ = 0.99
    - Replay capacity N = 100,000
    - Batch size B = 64
    - Target update τ = 0.005 (soft) or C = 1000 steps (hard)
    - ε_start = 1.0, ε_end = 0.01, ε_decay = 10,000
    - α_PER = 0.6, β_PER = 0.4→1.0
    - Training start: 1,000 steps (prefill buffer)

Initialize:
    - Q-network Q_φ with random weights φ
    - Target network Q_φ̄ ← φ
    - Replay buffer D ← ∅
    - Priority tree for PER
    
For episode = 1 to num_episodes:
    s ← env.reset()
    For t = 1 to T:
        # Action selection
        If random() < ε(t):
            a ← random action from A
        Else:
            a ← argmax_a Q_φ(s, a)
        
        # Environment interaction
        s', r, done ← env.step(a)
        
        # Compute initial priority (using current Q-values)
        δ ← |r + γ(1-done)·max_{a'} Q_φ̄(s',a') - Q_φ(s,a)|
        p ← δ^α_PER + ε_small
        
        # Store transition
        D.add((s, a, r, s', done), priority=p)
        
        # Training step
        If len(D) ≥ training_start AND t % update_freq == 0:
            # Sample prioritized batch
            batch, indices, weights ← D.sample(B, β_PER)
            
            # Compute targets
            targets ← []
            For (s_i, a_i, r_i, s'_i, done_i) in batch:
                y_i ← r_i + γ(1-done_i)·max_{a'} Q_φ̄(s'_i, a')
                targets.append(y_i)
            
            # Compute TD errors
            Q_pred ← Q_φ(s_i, a_i) for all i
            δ_batch ← targets - Q_pred
            
            # Weighted loss
            loss ← Σ_i weights[i] · δ_batch[i]²
            
            # Update Q-network
            φ ← φ - α · ∇_φ loss
            
            # Update priorities in buffer
            For i, idx in enumerate(indices):
                D.update_priority(idx, |δ_batch[i]|^α_PER + ε_small)
            
            # Update target network (soft update)
            φ̄ ← τ·φ + (1-τ)·φ̄
        
        # Next state
        s ← s'
        If done: break
    
    # Anneal β_PER and ε
    β_PER ← min(1.0, β_PER + Δβ)
    ε ← ε_end + (ε_start - ε_end)·exp(-episode/ε_decay)
```

---

## Hyperparameters (Complete List)

### Network Architecture
```python
NETWORK = {
    'hidden_layers': [256, 256, 128],
    'activation': 'ReLU',
    'output_dim': 270,  # |A|
    'dueling': False    # Set True for Dueling DQN
}
```

### Training
```python
TRAINING = {
    'learning_rate': 3e-4,
    'optimizer': 'Adam',
    'batch_size': 64,
    'discount_gamma': 0.99,
    'training_start': 1000,     # Prefill buffer
    'update_frequency': 4,       # Train every N steps
    'target_update_tau': 0.005,  # Soft update rate
    'max_episodes': 500,
    'max_steps_per_episode': 140
}
```

### Exploration
```python
EXPLORATION = {
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 10000
}
```

### Replay Buffer
```python
REPLAY = {
    'capacity': 100000,
    'alpha_per': 0.6,           # Priority exponent
    'beta_per_start': 0.4,
    'beta_per_end': 1.0,
    'beta_anneal_steps': 50000,
    'epsilon_small': 1e-6
}
```

---

## Implementation Considerations

### Discrete Action Space Handling

**Gated Action Flattening:**
```python
# Question actions: (LO, difficulty)
def question_to_action_id(lo_idx, difficulty):
    # difficulty: 0=Easy, 1=Medium, 2=Hard
    return lo_idx * 3 + difficulty

# Content actions: (LO, modality)
def content_to_action_id(lo_idx, modality):
    # modality: 0=video, 1=PPT, ..., 5=handout
    return 90 + lo_idx * 6 + modality
```

### Episode Termination

**Handling Terminal States:**
```python
if done:
    target = reward  # No bootstrapping
else:
    target = reward + gamma * max_a Q_target(next_state, a)
```

### Numerical Stability

**Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
```

**Huber Loss (Optional):**
```python
def huber_loss(delta, kappa=1.0):
    if |delta| <= kappa:
        return 0.5 * delta^2
    else:
        return kappa * (|delta| - 0.5*kappa)
```

---

## Expected Performance

### Learning Curve Characteristics
- **Early Phase (0-5k steps):** Exploration-dominated, high variance
- **Mid Phase (5k-20k steps):** Rapid improvement as Q-values stabilize
- **Late Phase (20k+ steps):** Convergence with occasional instability

### Comparison Expectations
- **vs Rule-Based:** ~20-30% better cumulative reward
- **vs PPO:** Similar final performance, higher training variance
- **vs PETS/MBPO:** Lower sample efficiency (needs more steps)

### Typical Metrics (Simulated Data)
- **Time-to-Mastery:** 95 ± 12 steps (mean ± SD over 20 seeds)
- **Cumulative Reward:** 85 ± 15
- **Blueprint Deviation:** 8 ± 3 percentage points
- **Training Time:** ~45 minutes on CPU (single seed)

---

## Statistical Reporting

### Paired Comparison Protocol
Run DQN on same S=20 seeds as all other algorithms.

**Report:**
```
DQN-PER Performance (S=20 seeds):
- Time-to-Mastery: 95 ± 12 steps [CI: 89-101]
- Cumulative Reward: 85 ± 15 [CI: 78-92]
- Blueprint Adherence: 92% (deviation 8 ± 3 pp)

vs PPO:
- Paired t-test: p=0.042 (PPO faster by 8 steps)
- Cohen's d: 0.52 (medium effect)
- Wilcoxon: p=0.038 (confirms)

vs MBPO:
- MBPO achieves target mastery 25% faster
- Effect size d=0.85 (large)
```

---

## Debugging Checklist

### Common Issues
1. **Q-values Explode:** Check target network updates, reduce learning rate
2. **No Learning:** Verify reward scaling, check ε-decay schedule
3. **Instability:** Increase target update frequency, use gradient clipping
4. **Slow Convergence:** Increase batch size, tune PER hyperparameters

### Monitoring During Training
- Q-value statistics (mean, std, max)
- TD error magnitude
- Epsilon value
- Replay buffer diversity
- Gradient norms

---

## Code Structure Template

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim=32, action_dim=270, hidden=[256,256,128]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def add(self, transition, priority):
        # Implementation with sum-tree for efficient sampling
        pass
    
    def sample(self, batch_size, beta):
        # Return (batch, indices, weights)
        pass
    
    def update_priorities(self, indices, priorities):
        # Update priorities for sampled transitions
        pass

class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                     lr=config['learning_rate'])
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config['replay_capacity'],
            alpha=config['alpha_per']
        )
        
        self.gamma = config['discount_gamma']
        self.epsilon = config['epsilon_start']
        # ... other config
    
    def select_action(self, state, eval_mode=False):
        if eval_mode or np.random.rand() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
                return q_values.argmax().item()
        return np.random.randint(self.action_dim)
    
    def train_step(self, batch_size, beta_per):
        batch, indices, weights = self.replay_buffer.sample(batch_size, beta_per)
        
        # Compute loss with importance sampling weights
        # Update priorities
        # Soft update target network
        pass
    
    def update_target_network(self, tau=0.005):
        for target_param, param in zip(self.target_network.parameters(),
                                        self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + 
                                    (1-tau) * target_param.data)
```

---

## References for Implementation

- **Original DQN Paper:** Mnih et al. (2015) "Human-level control through deep RL"
- **Prioritized Replay:** Schaul et al. (2016) "Prioritized Experience Replay"
- **Double DQN:** van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"
- **Dueling DQN:** Wang et al. (2016) "Dueling Network Architectures"

---

## Deliverables for Developer

1. **DQN Agent Class** with PER
2. **Replay Buffer** with priority sampling
3. **Training Loop** with ε-annealing and β-annealing
4. **Evaluation Script** for metrics collection
5. **Config File** with all hyperparameters
6. **Logging** for TensorBoard/WandB
