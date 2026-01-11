"""
MBPO training script adapted to the adaptive mock-interview simulator.
- Factorized discrete SAC actor (difficulty, LO, gate/content-modality)
- Ensemble dynamics model for short rollouts
- Replay mixing between real and model buffers

Assumptions:
- Environment follows OpenAI Gym API with Discrete(270) actions and 32-dim observations
- Reward is returned by env.step; model learns reward jointly with next-state
- Action mapping adds a gate class to the modality head (index 0 = question, 1-6 = modalities)

Aligned with DQN/PETS/PPO for 1:1 replication via shared_config.py
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gym
except ImportError:  # Minimal stub to avoid hard failure when gym is absent
    gym = None

# Import unified configuration for 1:1 replication
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared_config import UNIFIED_SEEDS, UNIFIED_EPISODES, UNIFIED_MAX_STEPS_PER_EPISODE

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class MBPOConfig:
    state_dim: int = 32
    num_difficulties: int = 3
    num_los: int = 30
    num_modalities: int = 6  # Content modalities
    modality_head: int = 7    # 1 gate (question) + 6 content modalities
    ensemble_size: int = 5
    model_hidden: int = 512
    actor_hidden: int = 256
    critic_hidden: int = 256
    discount: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_model: float = 1e-3
    lr_alpha: float = 3e-4
    initial_temperature: float = 0.2
    auto_alpha: bool = True
    target_entropy_scale: float = 1.0
    model_train_freq: int = 250
    model_train_epochs: int = 1000
    model_batch_size: int = 256
    rollout_length: int = 1
    rollout_batch_size: int = 400
    rollout_freq: int = 1
    model_retain_epochs: int = 5
    model_buffer_size: int = 100_000
    batch_size: int = 256
    real_ratio: float = 0.5
    max_steps: int = UNIFIED_EPISODES * UNIFIED_MAX_STEPS_PER_EPISODE  # ~41,300 steps (295 episodes × 140)
    max_episodes: int = UNIFIED_EPISODES  # 295 episodes (aligned with DQN/PETS/PPO)
    start_steps: int = 5_000
    update_after: int = 5_000
    update_every: int = 1
    model_updates: int = 1_000
    sac_updates: int = 1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 1_000
    save_interval: int = 10_000
    eval_episodes: int = 5

    def target_entropy(self) -> float:
        """Blueprint target entropy = sum of component entropies (natural log)."""
        return self.target_entropy_scale * (
            math.log(self.num_difficulties)
            + math.log(self.num_los)
            + math.log(self.modality_head)
        )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def action_triplet_to_id(d: torch.Tensor, lo: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Blueprint action mapping (spec_simulator.md):
    - m == 0 → question; action_id = lo * 3 + difficulty (0-89)
    - m in [1,6] → content; modality = m-1; action_id = 90 + lo * 6 + modality (90-269)
    Difficulty is ignored for content by construction (clamped to 0 before calling).
    """
    is_question = (m == 0)
    question_id = lo * 3 + d  # 0-89
    content_mod = torch.clamp(m - 1, min=0)
    content_id = 90 + lo * 6 + content_mod  # 90-269
    return torch.where(is_question, question_id, content_id)


def one_hot(indices: torch.Tensor, depth: int) -> torch.Tensor:
    out = torch.zeros(*indices.shape, depth, device=indices.device)
    out.scatter_(-1, indices.unsqueeze(-1), 1.0)
    return out


# -----------------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: str, track_gen: bool = False):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.track_gen = track_gen
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 3), dtype=torch.long)  # d, lo, m
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.float32)
        self.gen_ids = torch.zeros((capacity,), dtype=torch.long) if track_gen else None

    def add(self, state, action_triplet, reward, next_state, done, gen_id: int = 0) -> None:
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[self.ptr] = torch.as_tensor(action_triplet, dtype=torch.long)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32)
        self.dones[self.ptr] = float(done)
        if self.track_gen:
            self.gen_ids[self.ptr] = gen_id
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, min_gen: int | None = None):
        if min_gen is None or not self.track_gen:
            # Buffers are CPU tensors; sample indices on CPU then move batches to device
            idx = torch.randint(0, self.size, (batch_size,), device="cpu")
        else:
            valid = torch.nonzero(self.gen_ids[: self.size] >= min_gen, as_tuple=False).squeeze(-1)
            if valid.numel() == 0:
                raise RuntimeError("No model samples satisfy retention window")
            choice = torch.randint(0, valid.numel(), (batch_size,), device="cpu")
            idx = valid[choice]
        return (
            self.states[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_states[idx].to(self.device),
            self.dones[idx].to(self.device),
        )


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------

class FactorizedActor(nn.Module):
    def __init__(self, cfg: MBPOConfig):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.actor_hidden),
            nn.ReLU(),
            nn.Linear(cfg.actor_hidden, cfg.actor_hidden),
            nn.ReLU(),
        )
        self.head_d = nn.Linear(cfg.actor_hidden, cfg.num_difficulties)
        self.head_lo = nn.Linear(cfg.actor_hidden, cfg.num_los)
        self.head_m = nn.Linear(cfg.actor_hidden, cfg.modality_head)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(state)
        return self.head_d(h), self.head_lo(h), self.head_m(h)

    def sample(self, state: torch.Tensor):
        ld, ll, lm = self.forward(state)
        d_dist = torch.distributions.Categorical(logits=ld)
        lo_dist = torch.distributions.Categorical(logits=ll)
        m_dist = torch.distributions.Categorical(logits=lm)
        m = m_dist.sample()
        lo = lo_dist.sample()
        d = d_dist.sample()

        question_mask = (m == 0)
        # Difficulty is only meaningful for question actions; clamp to 0 otherwise
        d = torch.where(question_mask, d, torch.zeros_like(d))

        log_prob = m_dist.log_prob(m) + lo_dist.log_prob(lo)
        log_prob = log_prob + question_mask.float() * d_dist.log_prob(d)
        return (d, lo, m), log_prob

    def log_prob(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        d, lo, m = actions[:, 0], actions[:, 1], actions[:, 2]
        ld, ll, lm = self.forward(state)
        logp_m = F.log_softmax(lm, dim=-1).gather(1, m.view(-1, 1)).squeeze(-1)
        logp_lo = F.log_softmax(ll, dim=-1).gather(1, lo.view(-1, 1)).squeeze(-1)
        logp_d = F.log_softmax(ld, dim=-1).gather(1, d.view(-1, 1)).squeeze(-1)
        question_mask = (m == 0).float()
        log_p = logp_m + logp_lo + question_mask * logp_d
        return log_p


class QNetwork(nn.Module):
    def __init__(self, cfg: MBPOConfig):
        super().__init__()
        action_dim = cfg.num_difficulties + cfg.num_los + cfg.modality_head
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim + action_dim, cfg.critic_hidden),
            nn.ReLU(),
            nn.Linear(cfg.critic_hidden, cfg.critic_hidden),
            nn.ReLU(),
            nn.Linear(cfg.critic_hidden, 1),
        )
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)


class GaussianDynamics(nn.Module):
    def __init__(self, cfg: MBPOConfig):
        super().__init__()
        action_dim = cfg.num_difficulties + cfg.num_los + cfg.modality_head
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim + action_dim, cfg.model_hidden),
            nn.ReLU(),
            nn.Linear(cfg.model_hidden, cfg.model_hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(cfg.model_hidden, cfg.state_dim + 1)  # delta_state, reward
        self.logvar_head = nn.Linear(cfg.model_hidden, cfg.state_dim + 1)

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor):
        x = torch.cat([state, action_onehot], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = torch.clamp(self.logvar_head(h), -10, 2)
        return mean, logvar

    def predict(self, state: torch.Tensor, action_onehot: torch.Tensor):
        mean, logvar = self.forward(state, action_onehot)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mean + std * eps
        delta_state = sample[:, :-1]
        reward = sample[:, -1]
        return state + delta_state, reward


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------

class MBPOAgent:
    def __init__(self, env, cfg: MBPOConfig):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.global_step = 0

        self.actor = FactorizedActor(cfg).to(self.device)
        self.critic1 = QNetwork(cfg).to(self.device)
        self.critic2 = QNetwork(cfg).to(self.device)
        self.target1 = QNetwork(cfg).to(self.device)
        self.target2 = QNetwork(cfg).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.ensemble = nn.ModuleList([GaussianDynamics(cfg).to(self.device) for _ in range(cfg.ensemble_size)])

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=cfg.lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=cfg.lr_critic)
        self.model_opt = [torch.optim.Adam(m.parameters(), lr=cfg.lr_model) for m in self.ensemble]

        if cfg.auto_alpha:
            self.log_alpha = torch.tensor(math.log(self.cfg.initial_temperature), device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        else:
            self.log_alpha = torch.tensor(math.log(self.cfg.initial_temperature), device=self.device)
            self.alpha_opt = None

        real_cap = max(1_000_000, cfg.max_steps * 2)
        self.real_buffer = ReplayBuffer(real_cap, cfg.state_dim, cfg.device)
        self.model_buffer = ReplayBuffer(cfg.model_buffer_size, cfg.state_dim, cfg.device, track_gen=True)
        self.model_rollout_gen = 0

        self.last_log = {}

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def _action_onehot(self, actions: torch.Tensor) -> torch.Tensor:
        # If action is content (m>0), difficulty is irrelevant; force index 0
        m = actions[:, 2]
        d_idx = torch.where(m > 0, torch.zeros_like(actions[:, 0]), actions[:, 0])
        d = one_hot(d_idx, self.cfg.num_difficulties)
        lo = one_hot(actions[:, 1], self.cfg.num_los)
        m_onehot = one_hot(m, self.cfg.modality_head)
        return torch.cat([d, lo, m_onehot], dim=-1)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[int, Tuple[int, int, int]]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if eval_mode:
                ld, ll, lm = self.actor(state_t)
                d = ld.argmax(dim=-1)
                lo = ll.argmax(dim=-1)
                m = lm.argmax(dim=-1)
            else:
                (d, lo, m), _ = self.actor.sample(state_t)
            question_mask = (m == 0)
            d = torch.where(question_mask, d, torch.zeros_like(d))
        d, lo, m = d.item(), lo.item(), m.item()
        action_id = int(action_triplet_to_id(torch.tensor(d), torch.tensor(lo), torch.tensor(m)))
        return action_id, (d, lo, m)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update_dynamics(self, batch_size: int) -> Dict[str, float]:
        if self.real_buffer.size < batch_size:
            return {}
        s, a, r, ns, _ = self.real_buffer.sample(batch_size)
        a_oh = self._action_onehot(a)
        losses = []
        for model, opt in zip(self.ensemble, self.model_opt):
            mean, logvar = model.forward(s, a_oh)
            target = torch.cat([ns - s, r.unsqueeze(-1)], dim=-1)
            inv_var = torch.exp(-logvar)
            loss = ((mean - target) ** 2 * inv_var + logvar).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out = {"model_loss": float(np.mean(losses))}
        self.last_log.update(out)
        return out

    def rollout_model(self):
        if self.real_buffer.size == 0:
            return 0
        self.model_rollout_gen += 1
        # Sample starting states
        idx = torch.randint(0, self.real_buffer.size, (self.cfg.rollout_batch_size,), device=self.device)
        states = self.real_buffer.states[idx.cpu()].to(self.device)  # FIX: Move idx to CPU for indexing CPU buffer
        steps_added = 0
        for _ in range(self.cfg.rollout_length):
            with torch.no_grad():
                (d, lo, m), _ = self.actor.sample(states)
                actions = torch.stack([d, lo, m], dim=-1)
                a_oh = self._action_onehot(actions)
                model_idx = torch.randint(0, self.cfg.ensemble_size, (states.size(0),), device=self.device)
                next_states = []
                rewards = []
                for i, model in enumerate(self.ensemble):
                    mask = model_idx == i
                    if mask.any():
                        ns, rw = model.predict(states[mask], a_oh[mask])
                        next_states.append((mask, ns, rw))
                # Reassemble
                ns_buf = torch.zeros_like(states)
                rw_buf = torch.zeros((states.size(0),), device=self.device)
                for mask, ns, rw in next_states:
                    ns_buf[mask] = ns
                    rw_buf[mask] = rw
                done = torch.zeros((states.size(0),), device=self.device)
                for i in range(states.size(0)):
                    self.model_buffer.add(
                        states[i].cpu().numpy(),
                        actions[i].cpu().numpy(),
                        rw_buf[i].item(),
                        ns_buf[i].cpu().numpy(),
                        done[i].item(),
                        gen_id=self.model_rollout_gen,
                    )
                steps_added += states.size(0)
                states = ns_buf
        return steps_added

    def update_sac(self):
        if self.real_buffer.size < self.cfg.batch_size:
            return {}
        batch_size = self.cfg.batch_size
        real_bs = int(self.cfg.real_ratio * batch_size)
        model_bs = batch_size - real_bs
        sr, ar, rr, nsr, dr = self.real_buffer.sample(real_bs)
        min_gen = None
        if self.model_rollout_gen > 0:
            min_gen = max(0, self.model_rollout_gen - self.cfg.model_retain_epochs + 1)
        try:
            sm, am, rm, nsm, dm = self.model_buffer.sample(model_bs, min_gen=min_gen)
        except RuntimeError:
            return {}
        s = torch.cat([sr, sm], dim=0)
        a = torch.cat([ar, am], dim=0)
        r = torch.cat([rr, rm], dim=0)
        ns = torch.cat([nsr, nsm], dim=0)
        d = torch.cat([dr, dm], dim=0)

        a_oh = self._action_onehot(a)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            (nd, nlo, nm), logp_next = self.actor.sample(ns)
            nactions = torch.stack([nd, nlo, nm], dim=-1)
            na_oh = self._action_onehot(nactions)
            q1_t = self.target1(ns, na_oh)
            q2_t = self.target2(ns, na_oh)
            q_t = torch.min(q1_t, q2_t)
            v_t = q_t - alpha * logp_next
            target_q = r + (1.0 - d) * self.cfg.discount * v_t

        q1 = self.critic1(s, a_oh)
        q2 = self.critic2(s, a_oh)
        loss_q1 = F.mse_loss(q1, target_q)
        loss_q2 = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad()
        loss_q1.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        loss_q2.backward()
        self.critic2_opt.step()

        (ad, alo, am), logp = self.actor.sample(s)
        aa = torch.stack([ad, alo, am], dim=-1)
        aa_oh = self._action_onehot(aa)
        q1_pi = self.critic1(s, aa_oh)
        q2_pi = self.critic2(s, aa_oh)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = torch.tensor(0.0)
        if self.cfg.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.cfg.target_entropy()).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        with torch.no_grad():
            for param, target_param in zip(self.critic1.parameters(), self.target1.parameters()):
                target_param.data.mul_(1 - self.cfg.tau)
                target_param.data.add_(self.cfg.tau * param.data)
            for param, target_param in zip(self.critic2.parameters(), self.target2.parameters()):
                target_param.data.mul_(1 - self.cfg.tau)
                target_param.data.add_(self.cfg.tau * param.data)

        out = {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "loss_actor": actor_loss.item(),
            "loss_alpha": alpha_loss.item() if self.cfg.auto_alpha else 0.0,
            "alpha": alpha.item(),
        }
        self.last_log.update(out)
        return out

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self):
        """
        Enhanced training loop with comprehensive metrics tracking for paper.
        Tracks: TTM, cumulative reward, blueprint adherence, post-content gains,
        modality breakdown, question accuracy, final mastery, frustration.
        """
        state = self._reset_env()
        episode_return = 0.0
        ep_len = 0
        episodes = 0
        
        # Per-episode metric tracking
        episode_rewards: List[float] = []
        episode_ttm: List[int] = []
        episode_metrics: List[Dict] = []
        episode_steps: List[int] = []  # NEW: Track steps per episode for AUC/checkpoints
        
        # Current episode tracking
        ep_cumulative_reward = 0.0
        ep_mastery_reached_step = None
        ep_question_diffs: List[int] = []
        ep_content_gains: List[float] = []
        ep_modality_gains: Dict[str, List[float]] = {
            "video": [], "PPT": [], "text": [], "blog": [], "article": [], "handout": []
        }
        ep_question_total = 0
        ep_question_correct = 0
        ep_content_count = 0
        ep_frustration_history: List[float] = []
        
        # NEW: Calibration tracking (predicted mastery vs actual correctness)
        all_calibration_predicted: List[float] = []
        all_calibration_actual: List[float] = []
        
        modality_names = ["video", "PPT", "text", "blog", "article", "handout"]
        
        for t in range(1, self.cfg.max_steps + 1):
            self.global_step = t
            if t < self.cfg.start_steps:
                action_id = self.env.action_space.sample()
                action_triplet = self._decode_env_action(action_id)
            else:
                action_id, action_triplet = self.select_action(state, eval_mode=False)

            next_state, reward, done, info = self.env.step(action_id)
            episode_return += reward
            ep_cumulative_reward += reward
            ep_len += 1

            self.real_buffer.add(state, action_triplet, reward, next_state, done)
            
            # Track per-step metrics
            if info:
                # Mastery tracking
                mean_mastery = info.get("mean_mastery")
                if mean_mastery is not None:
                    if ep_mastery_reached_step is None and mean_mastery >= 0.8:
                        ep_mastery_reached_step = ep_len
                
                # Frustration tracking
                frustration = info.get("frustration")
                if frustration is not None:
                    ep_frustration_history.append(frustration)
                
                # Action-specific tracking
                action_type = info.get("type")
                if action_type == "content":
                    ep_content_count += 1
                    mastery_gain = info.get("mastery_gain", 0.0)
                    ep_content_gains.append(mastery_gain)
                    # Modality breakdown
                    modality_idx = info.get("modality")
                    if modality_idx is not None and 0 <= modality_idx < len(modality_names):
                        ep_modality_gains[modality_names[modality_idx]].append(mastery_gain)
                elif action_type == "question":
                    ep_question_total += 1
                    if info.get("correct", False):
                        ep_question_correct += 1
                    difficulty = info.get("difficulty")
                    if difficulty is not None:
                        ep_question_diffs.append(difficulty)
                    
                    # NEW: Track calibration (predicted mastery before question vs actual outcome)
                    # Use pre-step mastery from state (before env.step applied mastery update)
                    lo = info.get("lo")
                    if lo is not None and lo < 30:
                        predicted_mastery = float(state[lo])  # Mastery before this question
                        actual_outcome = 1.0 if info.get("correct", False) else 0.0
                        all_calibration_predicted.append(predicted_mastery)
                        all_calibration_actual.append(actual_outcome)
            
            state = next_state

            if done:
                # Compute episode metrics
                ep_metric = {
                    "episode": episodes + 1,
                    "return": episode_return,
                    "cumulative_reward": ep_cumulative_reward,
                    "time_to_mastery": ep_mastery_reached_step if ep_mastery_reached_step else ep_len,
                    "total_steps": ep_len,
                    "question_accuracy": ep_question_correct / ep_question_total if ep_question_total > 0 else 0.0,
                    "question_total": ep_question_total,
                    "question_correct": ep_question_correct,
                    "content_count": ep_content_count,
                    "content_rate": ep_content_count / ep_len if ep_len > 0 else 0.0,
                    "blueprint_adherence": self._compute_blueprint_adherence(ep_question_diffs),
                    "post_content_gain": float(np.mean(ep_content_gains)) if ep_content_gains else 0.0,
                    "final_mastery": float(np.mean(next_state[:30])) if len(next_state) >= 30 else 0.0,
                    "mean_frustration": float(np.mean(ep_frustration_history)) if ep_frustration_history else 0.0,
                    "modality_gains": {
                        mod: float(np.mean(gains)) if gains else 0.0
                        for mod, gains in ep_modality_gains.items()
                    },
                }
                episode_metrics.append(ep_metric)
                episode_rewards.append(episode_return)
                episode_steps.append(ep_len)  # NEW: Record episode length
                if ep_mastery_reached_step:
                    episode_ttm.append(ep_mastery_reached_step)
                
                # Reset for next episode
                state = self._reset_env()
                episode_return = 0.0
                ep_len = 0
                ep_cumulative_reward = 0.0
                ep_mastery_reached_step = None
                ep_question_diffs = []
                ep_content_gains = []
                ep_modality_gains = {mod: [] for mod in modality_names}
                ep_question_total = 0
                ep_question_correct = 0
                ep_content_count = 0
                ep_frustration_history = []
                
                episodes += 1
                if episodes >= self.cfg.max_episodes:
                    break

            if t >= self.cfg.update_after:
                # Model update per blueprint: train every model_train_freq steps for model_train_epochs iters
                if t % self.cfg.model_train_freq == 0:
                    for _ in range(self.cfg.model_train_epochs):
                        self.update_dynamics(self.cfg.model_batch_size)
                # SAC updates per step (default 1)
                if t % self.cfg.update_every == 0:
                    for _ in range(self.cfg.sac_updates):
                        self.update_sac()

            if t % self.cfg.rollout_freq == 0:
                self.rollout_model()

            if t % self.cfg.log_interval == 0:
                self._log_status(t, episode_rewards)

            if t % self.cfg.save_interval == 0:
                self._save_checkpoint(t)
        
        # Compute additional metrics for comparison
        auc_10k = compute_auc_at_10k(episode_rewards, episode_steps)
        checkpoints = compute_checkpoint_metrics(episode_rewards, episode_metrics, episode_steps)
        
        # NEW: Compute calibration MAE (mean absolute error between predicted mastery and actual correctness)
        calibration_mae = 0.0
        if all_calibration_predicted and len(all_calibration_predicted) == len(all_calibration_actual):
            calibration_mae = float(np.mean(np.abs(np.array(all_calibration_predicted) - np.array(all_calibration_actual))))
        
        # Return comprehensive results
        return {
            "returns": episode_rewards,
            "time_to_mastery": episode_ttm,
            "episode_metrics": episode_metrics,
            "auc_10k": auc_10k,
            "checkpoints": checkpoints,
            "total_steps_per_episode": episode_steps,
            "calibration_data": {  # NEW: For template Figure 4 and calibration analysis
                "predicted_mastery": all_calibration_predicted,
                "empirical_correct": all_calibration_actual,
                "mae": calibration_mae,
            },
        }

    def _decode_env_action(self, action_id: int) -> Tuple[int, int, int]:
        """Inverse of action_triplet_to_id for random exploration."""
        if action_id < 90:
            lo = action_id // 3
            d = action_id % 3
            m = 0  # gate to question
        else:
            cid = action_id - 90
            lo = cid // 6
            modality = cid % 6
            d = 0  # unused
            m = modality + 1
        return (d, lo, m)

    # ------------------------------------------------------------------
    # Logging / checkpointing / evaluation
    # ------------------------------------------------------------------
    def _log_status(self, step: int, episode_rewards: List[float]) -> None:
        if not episode_rewards:
            return
        avg_reward = float(np.mean(episode_rewards[-50:]))
        msg = (
            f"step={step} episodes={len(episode_rewards)} avg_reward_50={avg_reward:.2f} "
            f"alpha={self.last_log.get('alpha', float('nan')):.3f} "
            f"q1={self.last_log.get('loss_q1', float('nan')):.3f} "
            f"actor={self.last_log.get('loss_actor', float('nan')):.3f} "
            f"model={self.last_log.get('model_loss', float('nan')):.3f}"
        )
        print(msg)

    def _save_checkpoint(self, step: int, path: str | None = None) -> None:
        if path is None:
            path = f"checkpoint_step_{step}.pt"
        state = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "ensemble": [m.state_dict() for m in self.ensemble],
            "log_alpha": self.log_alpha,
            "cfg": dataclasses.asdict(self.cfg),
            "step": step,
        }
        torch.save(state, path)

    def evaluate(self, episodes: int | None = None) -> Dict[str, float]:
        episodes = episodes or self.cfg.eval_episodes
        returns = []
        ttm_list = []
        blueprint_list = []
        post_content_gains = []
        for _ in range(episodes):
            obs = self._reset_env()
            done = False
            ep_reward = 0.0
            step_count = 0
            mastery_reached_step = None
            content_gains = []
            question_diffs = []
            while not done:
                action_id, action_triplet = self.select_action(obs, eval_mode=True)
                obs, reward, done, info = self.env.step(action_id)
                ep_reward += reward
                step_count += 1
                mean_mastery = info.get("mean_mastery")
                if mastery_reached_step is None and mean_mastery is not None and mean_mastery >= 0.8:
                    mastery_reached_step = step_count
                result = info.get("result", {})
                if result.get("type") == "content":
                    content_gains.append(result.get("mastery_gain", 0.0))
                if result.get("type") == "question":
                    question_diffs.append(result.get("difficulty", 0))
                if step_count >= 140:  # safeguard per spec
                    break
            returns.append(ep_reward)
            if mastery_reached_step is not None:
                ttm_list.append(mastery_reached_step)
            if question_diffs:
                blueprint_list.append(self._compute_blueprint_adherence(question_diffs))
            if content_gains:
                post_content_gains.append(float(np.mean(content_gains)))
        return {
            "cumulative_reward": float(np.mean(returns)) if returns else 0.0,
            "time_to_mastery": float(np.mean(ttm_list)) if ttm_list else 0.0,
            "blueprint_adherence": float(np.mean(blueprint_list)) if blueprint_list else 0.0,
            "post_content_gain": float(np.mean(post_content_gains)) if post_content_gains else 0.0,
            "reward_variance": float(np.std(returns)) if returns else 0.0,
        }

    def _compute_blueprint_adherence(self, diffs: List[int]) -> float:
        if not diffs:
            return 100.0
        counts = np.bincount(np.array(diffs), minlength=3)[:3].astype(np.float64)
        total = counts.sum()
        actual = counts / total
        target = np.array([0.20, 0.60, 0.20], dtype=np.float64)
        deviation = np.abs(actual - target).mean()
        return (1.0 - deviation) * 100.0

    def _reset_env(self) -> np.ndarray:
        """Handle Gym API differences (returns obs or (obs, info))."""
        out = self.env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Multi-seed training and export utilities
# -----------------------------------------------------------------------------

def compute_auc_at_10k(episode_returns: List[float], episode_steps: List[int]) -> float:
    """
    Compute area under the cumulative reward curve up to the first 10,000 environment steps.
    Required by Table 4 for sample-efficiency comparison across algorithms.
    """
    cumulative_steps = 0
    cumulative_reward = 0.0
    for ret, steps in zip(episode_returns, episode_steps):
        if cumulative_steps >= 10_000:
            break
        cumulative_steps += steps
        cumulative_reward += ret
    return cumulative_reward


def compute_checkpoint_metrics(
    episode_returns: List[float],
    episode_metrics: List[Dict],
    episode_steps: List[int],
    checkpoints: List[int] = [10_000, 25_000, 50_000]
) -> Dict[int, Dict[str, float]]:
    """
    Capture snapshots of cumulative_reward, mean TTM, and blueprint_adherence at specific step checkpoints.
    Required by Table 5 for progress tracking during training.
    """
    results = {}
    cumulative_steps = 0
    cumulative_reward = 0.0
    ttm_buffer = []
    blueprint_buffer = []
    
    for ret, em, steps in zip(episode_returns, episode_metrics, episode_steps):
        cumulative_steps += steps
        cumulative_reward += ret
        if em.get("time_to_mastery") is not None:
            ttm_buffer.append(em["time_to_mastery"])
        if em.get("blueprint_adherence") is not None:
            blueprint_buffer.append(em["blueprint_adherence"])
        
        # Check if we've crossed any checkpoints
        for checkpoint in checkpoints:
            if checkpoint not in results and cumulative_steps >= checkpoint:
                results[checkpoint] = {
                    "cumulative_reward": cumulative_reward,
                    "mean_ttm": float(np.mean(ttm_buffer)) if ttm_buffer else 0.0,
                    "blueprint_adherence": float(np.mean(blueprint_buffer)) if blueprint_buffer else 0.0,
                }
    
    return results


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def summarize_across_seeds(results: List[Dict]) -> Dict:
    """
    Compute mean±SD and 95% CI across seeds (bootstrap).
    Matches DQN/PETS structure for unified comparison.
    """
    if not results:
        return {}
    
    # Aggregate returns
    all_returns = [r.get("returns", []) for r in results]
    cum_rewards = [float(np.sum(r)) for r in all_returns]
    
    # Aggregate TTM
    all_ttm = [r.get("time_to_mastery", []) for r in results]
    ttm_values = [float(np.mean(t)) if t else 0.0 for t in all_ttm]
    
    # Aggregate metrics from episode_metrics
    def aggregate_metric(metric_name: str) -> Tuple[float, float]:
        values = []
        for r in results:
            episode_metrics = r.get("episode_metrics", [])
            if episode_metrics:
                metric_values = [em.get(metric_name, 0.0) for em in episode_metrics]
                values.append(float(np.mean(metric_values)))
        if not values:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))
    
    mean_cumulative, std_cumulative = aggregate_metric("cumulative_reward")
    mean_ttm, std_ttm = aggregate_metric("time_to_mastery")
    mean_accuracy, std_accuracy = aggregate_metric("question_accuracy")
    mean_blueprint, std_blueprint = aggregate_metric("blueprint_adherence")
    mean_post_content, std_post_content = aggregate_metric("post_content_gain")
    mean_frustration, std_frustration = aggregate_metric("mean_frustration")
    mean_final_mastery, std_final_mastery = aggregate_metric("final_mastery")
    
    # NEW: Calibration MAE aggregation (model-based method)
    calibration_maes = []
    for r in results:
        calib_data = r.get("calibration_data", {})
        if calib_data and "mae" in calib_data:
            calibration_maes.append(calib_data["mae"])
    mean_calibration_mae = float(np.mean(calibration_maes)) if calibration_maes else 0.0
    std_calibration_mae = float(np.std(calibration_maes)) if calibration_maes else 0.0
    
    # Bootstrap 95% CI
    def bootstrap_ci(values, n_boot=1000, ci=0.95):
        if len(values) < 2:
            return (float(values[0]), float(values[0])) if values else (0.0, 0.0)
        boot_means = [float(np.mean(np.random.choice(values, len(values), replace=True))) for _ in range(n_boot)]
        lower = np.percentile(boot_means, (1-ci)/2 * 100)
        upper = np.percentile(boot_means, (1+ci)/2 * 100)
        return (float(lower), float(upper))
    
    ci_cumulative = bootstrap_ci(cum_rewards)
    ci_ttm = bootstrap_ci(ttm_values)
    
    return {
        "cumulative_reward": {
            "mean": mean_cumulative,
            "std": std_cumulative,
            "ci_95": ci_cumulative,
        },
        "time_to_mastery": {
            "mean": mean_ttm,
            "std": std_ttm,
            "ci_95": ci_ttm,
        },
        "question_accuracy": {
            "mean": mean_accuracy,
            "std": std_accuracy,
        },
        "blueprint_adherence": {
            "mean": mean_blueprint,
            "std": std_blueprint,
        },
        "post_content_gain": {
            "mean": mean_post_content,
            "std": std_post_content,
        },
        "mean_frustration": {
            "mean": mean_frustration,
            "std": std_frustration,
        },
        "final_mastery": {
            "mean": mean_final_mastery,
            "std": std_final_mastery,
        },
        "calibration_mae": {  # NEW: For template Table 1
            "mean": mean_calibration_mae,
            "std": std_calibration_mae,
        },
        "num_seeds": len(results),
    }


def train_single_seed(seed: int, cfg: MBPOConfig, env) -> Dict:
    """Train MBPO for a single seed and return results."""
    print(f"\n{'='*70}")
    print(f"MBPO Training: Seed {seed}")
    print(f"{'='*70}")
    
    cfg = dataclasses.replace(cfg, seed=seed)
    set_seed(seed)
    
    agent = MBPOAgent(env, cfg)
    start_time = time.time()
    results = agent.train()
    elapsed = time.time() - start_time
    
    wall_clock_time_seconds = time.time() - start_time
    wall_clock_time_minutes = wall_clock_time_seconds / 60.0
    
    results["seed"] = seed
    results["duration_s"] = elapsed
    results["wall_clock_time_minutes"] = wall_clock_time_minutes
    
    print(f"Seed {seed} completed in {elapsed:.1f}s")
    return results


def train_multi_seed(seeds: List[int], cfg: MBPOConfig, output_dir: str = "results") -> Dict:
    """
    Train MBPO across multiple seeds and export comprehensive results.
    Matches PETS/DQN output structure for comparison.
    """
    ensure_dir(output_dir)
    
    env = make_env()
    results = []
    
    for seed in seeds:
        result = train_single_seed(seed, cfg, env)
        results.append(result)
    
    # Aggregate statistics
    summary = summarize_across_seeds(results)
    
    # Export JSON summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "summary": summary,
            "per_seed": [
                {
                    "seed": r["seed"],
                    "duration_s": r["duration_s"],
                    "cumulative_reward": float(np.sum(r.get("returns", []))),
                    "mean_ttm": float(np.mean(r.get("time_to_mastery", []))) if r.get("time_to_mastery") else 0.0,
                }
                for r in results
            ],
        }, f, indent=2)
    
    print(f"\n✓ Summary exported to {summary_path}")
    
    # Export CSV (episode-level across all seeds)
    csv_path = os.path.join(output_dir, "episodes.csv")
    export_episodes_csv(results, csv_path)
    
    # Export figures
    figures_dir = os.path.join(output_dir, "figures")
    export_figures(results, figures_dir)
    
    return {"summary": summary, "results": results}


def export_episodes_csv(results: List[Dict], path: str) -> None:
    """Export per-episode metrics across all seeds to CSV."""
    import csv
    ensure_dir(path)
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Align with DQN/PPO schema: place modality columns right after post_content_gain
        writer.writerow([
            "seed", "episode", "return", "cumulative_reward", "ttm",
            "total_steps", "question_accuracy", "content_rate", "blueprint_adherence",
            "post_content_gain",
            "post_content_gain_video", "post_content_gain_PPT", "post_content_gain_text",
            "post_content_gain_blog", "post_content_gain_article", "post_content_gain_handout",
            "final_mastery", "mean_frustration"
        ])
        
        for result in results:
            seed = result["seed"]
            for em in result.get("episode_metrics", []):
                modality_gains = em.get("modality_gains", {})
                writer.writerow([
                    seed,
                    em.get("episode", 0),
                    em.get("return", 0.0),
                    em.get("cumulative_reward", 0.0),
                    em.get("time_to_mastery", 0),  # mapped to CSV column 'ttm'
                    em.get("total_steps", 0),
                    em.get("question_accuracy", 0.0),
                    em.get("content_rate", 0.0),
                    em.get("blueprint_adherence", 0.0),
                    em.get("post_content_gain", 0.0),
                    modality_gains.get("video", 0.0),
                    modality_gains.get("PPT", 0.0),
                    modality_gains.get("text", 0.0),
                    modality_gains.get("blog", 0.0),
                    modality_gains.get("article", 0.0),
                    modality_gains.get("handout", 0.0),
                    em.get("final_mastery", 0.0),
                    em.get("mean_frustration", 0.0),
                ])
    
    print(f"✓ Episodes CSV exported to {path}")


def export_figures(results: List[Dict], figures_dir: str) -> None:
    """Export learning curves and other visualizations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping figures")
        return
    
    ensure_dir(os.path.join(figures_dir, "placeholder"))
    
    # Learning curve
    plt.figure(figsize=(8, 5))
    for result in results:
        returns = result.get("returns", [])
        if returns:
            # Moving average
            window = min(10, len(returns))
            if window > 1:
                ma = np.convolve(returns, np.ones(window)/window, mode='valid')
                plt.plot(ma, alpha=0.6, linewidth=1)
    
    # Mean across seeds
    max_len = max(len(r.get("returns", [])) for r in results)
    if max_len > 0:
        padded = []
        for r in results:
            returns = r.get("returns", [])
            if len(returns) == max_len:
                padded.append(returns)
        if padded:
            mean_returns = np.mean(padded, axis=0)
            window = min(10, len(mean_returns))
            if window > 1:
                ma = np.convolve(mean_returns, np.ones(window)/window, mode='valid')
                plt.plot(ma, color='black', linewidth=2, label='Mean')
    
    plt.xlabel("Episode")
    plt.ylabel("Return (10-MA)")
    plt.title("MBPO Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "learning_curve.png"), dpi=200)
    plt.close()
    
    print(f"✓ Figures exported to {figures_dir}/")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def make_env():
    """Create the adaptive learning simulator environment (blueprint):
    - Discrete(270) action space with question/content mapping per spec_simulator.md
    - 32-dim observation (30 mastery + frustration + response time)
    
    NOTE: Imports AdaptiveLearningEnv from the parent directory (shared across algos)
    """
    # Try to import from PETS first (expects EnvConfig object)
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from pets_ver3.pets_train import AdaptiveLearningEnv as PETSEnv, EnvConfig
        
        # Create PETS-compatible config
        config = EnvConfig(
            max_steps=140,
            num_los=30,
            mastery_threshold=0.8,
            critical_frustration=0.95,
            blueprint_target=(0.2, 0.6, 0.2),
            blueprint_penalty=0.2
        )
        return PETSEnv(config)
    except ImportError:
        pass
    
    # Fall back to DQN (expects seed int as first arg)
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from dqn_ver3.train_dqn import AdaptiveLearningEnv as DQNEnv
        return DQNEnv(seed=0, max_steps=140)
    except ImportError:
        raise RuntimeError(
            "AdaptiveLearningEnv not found in PETS or DQN modules. "
            "Ensure the shared environment is accessible."
        )


def main():
    print(f"\n{'='*70}")
    print(f"🚀 MBPO TRAINING STARTED")
    print(f"{'='*70}\n")
    
    parser = argparse.ArgumentParser(description="Train MBPO for adaptive mock interviews (aligned with DQN/PETS/PPO)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (overrides multi-seed)")
    parser.add_argument("--seeds", type=int, nargs="*", default=UNIFIED_SEEDS, 
                        help=f"Multiple seeds for statistical validation (default: {UNIFIED_SEEDS})")
    parser.add_argument("--episodes", type=int, default=UNIFIED_EPISODES,
                        help=f"Number of episodes (default: {UNIFIED_EPISODES} for ~30k steps)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps if needed")
    parser.add_argument("--output", type=str, default="results/mbpo", help="Output directory")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load")
    args = parser.parse_args()
    
    # Determine seeds to use
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = args.seeds
    
    # Build config
    print(f"📋 Configuration:")
    print(f"   Seeds: {seeds}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    max_steps = args.max_steps if args.max_steps else (args.episodes * UNIFIED_MAX_STEPS_PER_EPISODE)
    print(f"   Max steps per episode: {UNIFIED_MAX_STEPS_PER_EPISODE}")
    print(f"   Total max steps: {max_steps}")
    print(f"{'='*70}\n")
    
    cfg = MBPOConfig(
        seed=seeds[0],  # Will be overridden per seed in multi-seed loop
        max_episodes=args.episodes,
        max_steps=max_steps,
    )
    
    if args.eval:
        print(f"📊 EVALUATION MODE")
        # Evaluation mode (single run)
        env = make_env()
        print(f"✅ Environment created")
        agent = MBPOAgent(env, cfg)
        print(f"✅ Agent created")
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location=cfg.device)
            agent.actor.load_state_dict(state["actor"])
            agent.critic1.load_state_dict(state["critic1"])
            agent.critic2.load_state_dict(state["critic2"])
            agent.target1.load_state_dict(state["target1"])
            agent.target2.load_state_dict(state["target2"])
            for m, sd in zip(agent.ensemble, state.get("ensemble", [])):
                m.load_state_dict(sd)
            agent.log_alpha = state.get("log_alpha", agent.log_alpha)
        metrics = agent.evaluate()
        print(json.dumps(metrics, indent=2))
    else:
        # Training mode (multi-seed if multiple seeds provided)
        print(f"🎓 TRAINING MODE")
        if len(seeds) == 1:
            # Single seed training
            print(f"📍 Single seed training (seed={seeds[0]})")
            env = make_env()
            print(f"✅ Environment created")
            try:
                result = train_single_seed(seeds[0], cfg, env)
                print(f"✅ Training complete!")
                print(f"\nResults: {json.dumps(result.get('episode_metrics', [])[-5:], indent=2)}")
                
                # NEW: Export JSON for single seed (for smoke tests and production consistency)
                os.makedirs(args.output, exist_ok=True)
                summary_path = os.path.join(args.output, "summary.json")
                returns = [m.get("cumulative_reward", 0.0) for m in result.get("episode_metrics", [])]
                with open(summary_path, "w") as f:
                    json.dump({
                        "auc_10k": float(np.sum(returns[:100])) if returns else 0.0,  # Approximate AUC@10k
                        "checkpoints": {},  # Not computed for single seed smoke test
                        "wall_clock_time_minutes": result.get("duration_s", 0) / 60.0,
                        "seed": result["seed"],
                        "episodes_completed": len(returns),
                        "total_return": float(np.sum(returns)) if returns else 0.0,
                        "mean_return": float(np.mean(returns)) if returns else 0.0,
                    }, f, indent=2)
                print(f"✅ Exported: {summary_path}")
            except Exception as e:
                print(f"❌ Training failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            # Multi-seed training with full export
            print(f"📍 Multi-seed training ({len(seeds)} seeds)")
            try:
                output = train_multi_seed(seeds, cfg, args.output)
                print(f"\n{'='*70}")
                print(f"✅ Multi-seed training complete!")
                print(f"Summary: {json.dumps(output['summary'], indent=2)}")
                print(f"{'='*70}")
            except Exception as e:
                print(f"❌ Multi-seed training failed: {e}")
                import traceback
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
