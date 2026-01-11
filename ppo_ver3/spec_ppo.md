# Proximal Policy Optimization (PPO) for Discrete Actions - Full Specification

## Overview

PPO is a **policy-gradient, model-free** RL algorithm that learns an explicit stochastic policy over discrete actions. It uses a clipped surrogate objective to ensure stable, monotonic improvement.

## Algorithm Type
- **Category:** Model-Free, On-Policy, Policy-Gradient
- **Policy:** Explicit categorical distribution π_θ(a|s)
- **Action Space:** Discrete (unified gated action space, 270 actions)

---

## Mathematical Foundation

### Policy Representation

Stochastic categorical policy over unified discrete action space:

```
π_θ(a|s) = Categorical(logits_θ(s))
```

Output: Probability distribution over all 270 actions (90 questions + 180 content)

### Value Function

Learn state-value function V_ν(s) to reduce variance:

```
V_ν(s) ≈ E_π[Σ γ^t r_t | s_0 = s]
```

### Advantage Estimation (GAE)

Generalized Advantage Estimation for variance reduction:

```
Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

Where TD residual:
```
δ_t = r_t + γV_ν(s_{t+1})(1-done_t) - V_ν(s_t)
```

Typical λ=0.95 (balances bias-variance tradeoff)

---

## PPO Objective Function

### Clipped Surrogate Objective

```
L_CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

Where probability ratio:
```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

**Intuition:** Clip prevents too large policy updates (conservative optimization)

### Value Function Loss

```
L_VF(ν) = E_t[(V_ν(s_t) - R̂_t)²]
```

Where R̂_t is the return-to-go (discounted cumulative reward from t)

Optional: Clip value updates similar to policy

### Entropy Bonus

Encourage exploration:
```
H(π_θ) = E_s[Σ_a -π_θ(a|s) log π_θ(a|s)]
```

### Combined Objective

```
L_PPO(θ, ν) = L_CLIP(θ) - λ_V·L_VF(ν) + λ_H·H(π_θ)
```

Typical weights:
- `λ_V = 0.5` (value loss weight)
- `λ_H = 0.01` (entropy coefficient)

---

## PPO Algorithm (Complete)

### Hyperparameters

```python
CONFIG = {
    # Network
    'state_dim': 32,
    'action_dim': 270,
    'hidden_layers': [256, 256],
    
    # Training
    'learning_rate': 3e-4,
    'discount_gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    
    # Rollout
    'rollout_steps': 2048,      # Collect this many steps before update
    'num_envs': 1,              # Parallel environments (can be > 1)
    'epochs_per_update': 10,    # K epochs on same batch
    'minibatch_size': 64,
    
    # Episodes
    'max_episodes': 500,
    'max_steps_per_episode': 140
}
```

### Algorithm Pseudocode

```
Algorithm: Proximal Policy Optimization (PPO) - Discrete

Initialize:
    - Policy network π_θ: (state) → logits over 270 actions
    - Value network V_ν: (state) → scalar value
    - Optimizer (Adam with lr=3e-4)
    - Rollout buffer B

For iteration = 1 to num_iterations:
    
    # Phase 1: Collect Rollout (on-policy data)
    B ← ∅
    s ← env.reset()
    For t = 1 to rollout_steps:
        # Sample action from current policy
        a ~ π_θ(·|s)
        logprob_old ← log π_θ(a|s)
        value ← V_ν(s)
        
        # Environment step
        s', r, done ← env.step(a)
        
        # Store transition
        B.add((s, a, r, s', done, logprob_old, value))
        
        s ← s'
        If done: s ← env.reset()
    
    # Phase 2: Compute Advantages (GAE)
    advantages ← []
    returns ← []
    For each trajectory in B:
        Compute GAE advantages Â using λ=0.95, γ=0.99
        Compute returns R̂ = Â + V_ν(s)
        advantages.append(Â)
        returns.append(R̂)
    
    # Normalize advantages (optional but recommended)
    advantages ← (advantages - mean) / (std + 1e-8)
    
    # Phase 3: Policy Update (K epochs on same data)
    For epoch = 1 to K (default K=10):
        # Shuffle and create minibatches
        indices ← shuffle(0, 1, ..., rollout_steps-1)
        For minibatch in split(indices, minibatch_size):
            
            # Get old data
            states, actions, old_logprobs, advantages, returns ← B[minibatch]
            
            # Forward pass with current policy
            logits ← π_θ(states)
            new_logprobs ← log π_θ(actions | states)
            values ← V_ν(states)
            entropy ← -Σ π_θ(a|s) log π_θ(a|s)
            
            # Compute ratio and clipped objective
            ratio ← exp(new_logprobs - old_logprobs)
            surr1 ← ratio * advantages
            surr2 ← clip(ratio, 1-ε, 1+ε) * advantages
            policy_loss ← -mean(min(surr1, surr2))
            
            # Value loss (MSE)
            value_loss ← mean((values - returns)²)
            
            # Entropy bonus
            entropy_loss ← -mean(entropy)
            
            # Combined loss
            loss ← policy_loss + λ_V·value_loss + λ_H·entropy_loss
            
            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters, max_norm=0.5)
            optimizer.step()
    
    # Logging
    Log(mean_reward, mean_advantage, policy_loss, value_loss, etc.)
```

---

## Network Architectures

### Shared Feature Extractor (Recommended)

```python
class SharedNetwork(nn.Module):
    def __init__(self, state_dim=32, hidden=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        self.shared = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.shared(state)

class ActorCritic(nn.Module):
    def __init__(self, state_dim=32, action_dim=270, hidden=[256,256]):
        super().__init__()
        self.shared = SharedNetwork(state_dim, hidden)
        
        # Policy head (actor)
        self.policy_head = nn.Linear(hidden[-1], action_dim)
        
        # Value head (critic)
        self.value_head = nn.Linear(hidden[-1], 1)
    
    def forward(self, state):
        features = self.shared(state)
        logits = self.policy_head(features)  # (batch, 270)
        value = self.value_head(features)     # (batch, 1)
        return logits, value.squeeze(-1)
    
    def get_action_and_value(self, state, action=None):
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value
```

### Separate Networks (Alternative)

```python
class PolicyNetwork(nn.Module):
    # Maps state → action logits
    pass

class ValueNetwork(nn.Module):
    # Maps state → value estimate
    pass
```

---

## Advantage Computation (GAE)

### Generalized Advantage Estimation

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: (T,) array of rewards
    values: (T,) array of V(s_t)
    dones: (T,) array of done flags
    
    Returns:
        advantages: (T,) array
        returns: (T,) array
    """
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = 0  # or V(s_{T+1}) if not terminal
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t + 1]
        
        # TD residual
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        
        # GAE recursion
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    
    returns = advantages + values
    return advantages, returns
```

---

## Clipping Mechanisms

### Policy Ratio Clipping

```python
def ppo_clip_loss(ratio, advantage, epsilon=0.2):
    """
    ratio: π_new(a|s) / π_old(a|s)
    advantage: Â_t
    """
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage
    return -torch.min(surr1, surr2).mean()
```

### Value Clipping (Optional)

```python
def value_clip_loss(value, value_old, returns, epsilon=0.2):
    """Clip value updates similar to policy"""
    value_clipped = value_old + torch.clamp(
        value - value_old, -epsilon, epsilon
    )
    loss_unclipped = (value - returns) ** 2
    loss_clipped = (value_clipped - returns) ** 2
    return torch.max(loss_unclipped, loss_clipped).mean()
```

---

## Action Selection

### Training (Stochastic)

```python
def select_action(state, policy_network):
    logits, value = policy_network(state)
    probs = Categorical(logits=logits)
    action = probs.sample()
    log_prob = probs.log_prob(action)
    return action.item(), log_prob.item(), value.item()
```

### Evaluation (Deterministic - Optional)

```python
def select_action_eval(state, policy_network):
    logits, _ = policy_network(state)
    action = logits.argmax(dim=-1)
    return action.item()
```

**Note:** For stochastic evaluation (more realistic), still sample from π_θ(·|s)

---

## Episode Termination Handling

### Finite-Horizon Returns

When computing returns and advantages:
- If `done=True` at step t: next_value = 0 (terminal)
- If episode reaches max_steps but not "naturally done": may bootstrap from V(s_T)

### Implementation

```python
for t in range(len(episode)):
    if dones[t]:
        next_value = 0
    else:
        next_value = values[t+1] if t < len(episode)-1 else bootstrap_value
    
    delta[t] = rewards[t] + gamma * next_value - values[t]
```

---

## Exploration via Entropy

### Entropy Calculation

```python
def entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()
```

### Entropy Coefficient Annealing (Optional)

```python
def update_entropy_coef(initial=0.01, final=0.001, steps=50000, current_step=0):
    return final + (initial - final) * max(0, 1 - current_step / steps)
```

---

## Hyperparameter Tuning Guide

### Critical Hyperparameters

| Parameter | Typical Range | Impact |
|-----------|---------------|--------|
| `clip_epsilon` | 0.1 - 0.3 | Larger = more aggressive updates |
| `learning_rate` | 1e-4 - 3e-4 | Tune based on convergence |
| `gae_lambda` | 0.9 - 0.99 | Higher = more bias, less variance |
| `rollout_steps` | 1024 - 4096 | More = better gradient estimate |
| `epochs_per_update` | 3 - 15 | More = risk overfitting to old data |
| `entropy_coef` | 0.001 - 0.1 | Higher = more exploration |

### Recommended Starting Point (Education Domain)

```python
RECOMMENDED_CONFIG = {
    'learning_rate': 3e-4,
    'clip_epsilon': 0.2,
    'gae_lambda': 0.95,
    'discount_gamma': 0.99,
    'rollout_steps': 2048,
    'epochs_per_update': 10,
    'minibatch_size': 64,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5
}
```

---

## Comparison: PPO vs DQN

### When PPO Excels
1. **Action Space:** Better for large discrete spaces (270 actions here)
2. **Exploration:** Stochastic policy naturally explores multiple good actions
3. **Stability:** Clipping prevents destructive updates
4. **Convergence:** Often smoother learning curves than DQN

### When DQN Excels
1. **Sample Efficiency:** Reuses data via replay buffer
2. **Strong Shaping:** When Q-values have clear structure
3. **Deterministic Optimal:** If true optimal policy is deterministic

### Expected Relative Performance (This Domain)
- **Sample Efficiency:** PPO slightly better early, comparable late
- **Final Performance:** PPO ≈ DQN ± 5% cumulative reward
- **Stability:** PPO > DQN (lower variance across seeds)
- **Training Time:** PPO < DQN (fewer steps needed)

---

## Implementation Checklist

### Must-Have Components
- [ ] Actor-Critic network with shared features
- [ ] GAE advantage computation (λ=0.95)
- [ ] Clipped surrogate objective
- [ ] Entropy bonus
- [ ] Gradient clipping (max_norm=0.5)
- [ ] Advantage normalization
- [ ] Multiple epochs per rollout (K=10)

### Nice-to-Have
- [ ] Value clipping
- [ ] Learning rate annealing
- [ ] Entropy coefficient annealing
- [ ] Parallel environments (vectorized)
- [ ] Early stopping (KL divergence threshold)

---

## Expected Performance (Simulated Data)

### Convergence Metrics
- **Time-to-Mastery:** 87 ± 8 steps (mean ± SD, S=20 seeds)
- **Cumulative Reward:** 92 ± 10
- **Blueprint Adherence:** 94% (6 ± 2 pp deviation)
- **Training Time:** ~30 minutes/seed on CPU

### Learning Curve Shape
- **Phase 1 (0-10k steps):** Rapid improvement from exploration
- **Phase 2 (10k-30k steps):** Steady convergence
- **Phase 3 (30k+ steps):** Plateau with low variance

---

## Statistical Reporting Template

```
PPO Performance (S=20 seeds):
- Time-to-Mastery: 87 ± 8 steps [95% CI: 83-91]
- Cumulative Reward: 92 ± 10 [95% CI: 87-97]
- Blueprint Deviation: 6 ± 2 percentage points

Comparison vs DQN:
- Paired t-test: p=0.031 (PPO 8 steps faster)
- Cohen's d: 0.58 (medium effect, favors PPO)
- Reward variance: PPO 40% lower than DQN

Comparison vs MBPO:
- MBPO 15% faster time-to-mastery
- PPO more stable (lower seed variance)
- Both outperform rule-based by >30%
```

---

## Debugging Checklist

### Common Issues

1. **Policy Collapse (all actions → one action)**
   - Increase entropy coefficient
   - Check advantage normalization
   - Reduce learning rate

2. **Value Function Not Learning**
   - Increase value_loss_coef
   - Check return computation
   - Verify GAE implementation

3. **Slow Convergence**
   - Increase rollout_steps
   - Tune learning rate
   - Check reward scaling

4. **High Variance Across Seeds**
   - Increase rollout_steps
   - More epochs_per_update
   - Check initialization

### Monitoring Metrics
- Policy loss, value loss, entropy
- Mean/std of advantages
- KL divergence (old vs new policy)
- Explained variance: `1 - Var(V-R̂) / Var(R̂)`
- Gradient norms

---

## Code Template

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.actor_critic = ActorCritic(state_dim, action_dim, 
                                         config['hidden_layers'])
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), 
            lr=config['learning_rate']
        )
        
        self.gamma = config['discount_gamma']
        self.gae_lambda = config['gae_lambda']
        self.clip_epsilon = config['clip_epsilon']
        self.value_loss_coef = config['value_loss_coef']
        self.entropy_coef = config['entropy_coef']
        self.max_grad_norm = config['max_grad_norm']
    
    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, _, value = self.actor_critic.get_action_and_value(
                torch.FloatTensor(state)
            )
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones):
        # Implementation as shown above
        pass
    
    def update(self, rollout_buffer, epochs=10, minibatch_size=64):
        # Extract data
        states, actions, old_logprobs, advantages, returns = rollout_buffer
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Get current policy predictions
                _, new_logprobs, entropy, values = self.actor_critic.get_action_and_value(
                    states[mb_indices], 
                    actions[mb_indices]
                )
                
                # Compute losses
                ratio = (new_logprobs - old_logprobs[mb_indices]).exp()
                
                surr1 = ratio * advantages[mb_indices]
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 
                                    1+self.clip_epsilon) * advantages[mb_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = ((values - returns[mb_indices]) ** 2).mean()
                entropy_loss = -entropy.mean()
                
                loss = (policy_loss + 
                        self.value_loss_coef * value_loss + 
                        self.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 
                                         self.max_grad_norm)
                self.optimizer.step()
```

---

## Deliverables for Developer

1. **ActorCritic Network** with shared backbone
2. **PPO Agent Class** with update method
3. **GAE Computation** function
4. **Rollout Collection** loop
5. **Training Script** with logging
6. **Config File** with all hyperparameters
7. **Evaluation Script** for metrics
