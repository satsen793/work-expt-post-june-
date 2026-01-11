"""
PETS training script for the adaptive mock interview simulator (discrete actions).
Based on spec_* files in this workspace.
Aligned with DQN_VER3 for 1:1 replication via shared_config.py
"""
from __future__ import annotations

import dataclasses
import math
import random
import time
import sys
import os
from typing import Callable, Dict, List, Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import unified configuration for 1:1 replication with DQN_VER3
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared_config import UNIFIED_SEEDS, UNIFIED_EPISODES, UNIFIED_MAX_STEPS_PER_EPISODE

# -----------------------
# Configurations
# -----------------------


@dataclasses.dataclass
class EnvConfig:
    max_steps: int = UNIFIED_MAX_STEPS_PER_EPISODE  # 140 steps (aligned with DQN)
    mastery_threshold: float = 0.8
    critical_frustration: float = 0.95
    num_los: int = 30
    num_questions: int = 600
    num_contents: int = 180
    blueprint_target: Tuple[float, float, float] = (0.2, 0.6, 0.2)  # Easy/Medium/Hard
    blueprint_penalty: float = 0.2


@dataclasses.dataclass
class ModelConfig:
    ensemble_size: int = 5
    hidden_dim: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    train_epochs: int = 50
    batch_size: int = 256
    logvar_clamp: Tuple[float, float] = (-10.0, 2.0)


@dataclasses.dataclass
class MPCConfig:
    horizon: int = 10
    iterations: int = 5
    candidates: int = 500
    elite_fraction: float = 0.1
    update_rate: float = 0.5
    gamma: float = 0.99
    smoothing_eps: float = 1e-6
    uncertainty_penalty: float = 0.0


@dataclasses.dataclass
class TrainConfig:
    total_episodes: int = UNIFIED_EPISODES  # 295 episodes for ~30k total steps (aligned with DQN)
    initial_exploration: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds: Tuple[int, ...] = tuple(UNIFIED_SEEDS)  # [0, 1, 2, 3, 4] (aligned with DQN)


ENV_CONFIG = EnvConfig()
MODEL_CONFIG = ModelConfig()
MPC_CONFIG = MPCConfig()
TRAIN_CONFIG = TrainConfig()

# -----------------------
# Environment
# -----------------------


class DiscreteActionSpace:
    """Simple wrapper for action_space.sample() compatibility with gym-based algorithms."""
    def __init__(self, n: int):
        self.n = n
    
    def sample(self):
        """Sample random action."""
        return np.random.randint(0, self.n)


class AdaptiveLearningEnv:
    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.num_los = config.num_los
        self.num_actions = 270  # 90 questions + 180 content
        self.action_space = DiscreteActionSpace(270)  # NEW: For gym-compatible algorithms (MBPO)
        self.max_steps = config.max_steps
        self.questions = self._load_questions()
        self.contents = self._load_contents()
        self.learner_state: Dict[str, np.ndarray] = {}
        self.step_count = 0
        self.episode_log: List[Dict] = []
        self.cumulative_reward = 0.0
        self.question_total = 0
        self.question_correct = 0
        self.content_count = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.learner_state = self._initialize_learner()
        self.step_count = 0
        self.diff_counts = [0, 0, 0]  # track question difficulty counts this episode
        self.episode_log = []
        self.cumulative_reward = 0.0
        self.question_total = 0
        self.question_correct = 0
        self.content_count = 0
        self.time_to_mastery = None
        self.calibration_predicted = []
        self.calibration_actual = []
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        action_dict = self._decode_action(action)
        if action_dict["type"] == "question" and self.learner_state.get("fail_streak", 0) >= 3:
            action_dict = self._remediate_content_action(action_dict["lo"])
        if action_dict["type"] == "question":
            result = self._execute_question(action_dict)
        else:
            result = self._execute_content(action_dict)

        reward = self._compute_reward(result)
        self.step_count += 1
        self.cumulative_reward += reward
        done, reason = self._is_terminal()
        obs = self._get_observation()
        info = {
            "result": result,
            "termination_reason": reason,
            "step": self.step_count,
            "mean_mastery": float(np.mean(self.learner_state["mastery"]))
        }
        self.episode_log.append({"step": self.step_count, "action": action, "reward": reward, "done": done, **info})
        return obs, reward, done, info

    # --- helpers ---
    def _initialize_learner(self) -> Dict[str, np.ndarray]:
        return {
            "mastery": np.random.beta(a=2, b=5, size=self.num_los),
            "ability": np.random.normal(0, 1),
            "frustration": 0.0,
            "response_time": 0.0,
            "fail_streak": 0,
            "engagement": 1.0,
        }

    def _load_questions(self) -> List[Dict]:
        questions = []
        difficulties = ["Easy", "Medium", "Hard"]
        # 20 questions per LO (approx 20-60-20 split)
        for lo in range(self.num_los):
            for idx in range(20):
                if idx < 4:
                    diff = "Easy"
                elif idx < 16:
                    diff = "Medium"
                    
                else:
                    diff = "Hard"
                a = np.random.uniform(0.5, 1.0) if diff == "Easy" else (
                    np.random.uniform(1.0, 1.5) if diff == "Medium" else np.random.uniform(1.5, 2.0)
                )
                b = np.random.uniform(-2.0, -0.5) if diff == "Easy" else (
                    np.random.uniform(-0.5, 0.5) if diff == "Medium" else np.random.uniform(0.5, 2.0)
                )
                c = np.random.uniform(0.1, 0.25)
                questions.append({
                    "id": f"q_{lo}_{idx}",
                    "learning_outcome": lo,
                    "difficulty": diff,
                    "irt_a": a,
                    "irt_b": b,
                    "irt_c": c,
                    "response_time_mean": 30,
                    "response_time_std": 10,
                })
        random.shuffle(questions)
        return questions

    def _load_contents(self) -> List[Dict]:
        modalities = ["video", "PPT", "text", "blog", "article", "handout"]
        contents = []
        for lo in range(self.num_los):
            for modality in modalities:
                duration_low, duration_high, eff_low, eff_high = self._content_params(modality)
                contents.append({
                    "id": f"c_{lo}_{modality}",
                    "learning_outcome": lo,
                    "modality": modality,
                    "duration": np.random.uniform(duration_low, duration_high),
                    "effectiveness": np.random.uniform(eff_low, eff_high),
                    "engagement_impact": self._modality_frustration(modality),
                })
        return contents

    def _modality_frustration(self, modality: str) -> float:
        table = {
            "video": -0.08,
            "PPT": -0.05,
            "text": 0.02,
            "blog": -0.03,
            "article": 0.0,
            "handout": 0.05,
        }
        return table[modality]

    def _content_params(self, modality: str) -> Tuple[float, float, float, float]:
        # modality-specific ranges per simulator spec
        table = {
            "video": (15, 25, 0.10, 0.15),
            "PPT": (10, 20, 0.08, 0.12),
            "text": (5, 10, 0.05, 0.08),
            "blog": (8, 15, 0.07, 0.10),
            "article": (10, 18, 0.06, 0.09),
            "handout": (5, 12, 0.05, 0.08),
        }
        return table[modality]

    def _decode_action(self, action_id: int) -> Dict:
        if action_id < 90:
            lo_index = action_id // 3
            diff_idx = action_id % 3
            return {
                "type": "question",
                "lo": lo_index,
                "difficulty_idx": diff_idx,
                "difficulty": ["Easy", "Medium", "Hard"][diff_idx],
            }
        else:
            content_id = action_id - 90
            lo_index = content_id // 6
            modality_idx = content_id % 6
            modalities = ["video", "PPT", "text", "blog", "article", "handout"]
            return {
                "type": "content",
                "lo": lo_index,
                "modality_idx": modality_idx,
                "modality": modalities[modality_idx],
            }

    def _sample_question(self, lo: int, difficulty: str) -> Dict:
        candidates = [q for q in self.questions if q["learning_outcome"] == lo and q["difficulty"] == difficulty]
        return random.choice(candidates)

    def _sample_content(self, lo: int, modality: str) -> Dict:
        candidates = [c for c in self.contents if c["learning_outcome"] == lo and c["modality"] == modality]
        return random.choice(candidates)

    def _remediate_content_action(self, lo: int) -> Dict:
        # Fail-streak remediation: route to engaging content (video) for the same LO
        modality = "video"
        return {
            "type": "content",
            "lo": lo,
            "modality_idx": 0,
            "modality": modality,
        }

    def _execute_question(self, action: Dict) -> Dict:
        lo = action["lo"]
        difficulty = action["difficulty"]
        question = self._sample_question(lo, difficulty)
        theta = self.learner_state["ability"]
        a = question["irt_a"]
        b = question["irt_b"]
        c = question["irt_c"]
        prob_correct = c + (1 - c) / (1 + math.exp(-a * (theta - b)))
        correct = np.random.rand() < prob_correct
        current_mastery = self.learner_state["mastery"][lo]
        
        # Track calibration: predicted mastery vs actual correctness
        self.calibration_predicted.append(float(current_mastery))
        self.calibration_actual.append(1.0 if correct else 0.0)

        if correct:
            gain = 0.05 * (1 - current_mastery)
            self.learner_state["mastery"][lo] = min(1.0, current_mastery + gain)
            self.learner_state["fail_streak"] = 0
            self.learner_state["frustration"] = max(0.0, self.learner_state["frustration"] - 0.05)
            self.learner_state["ability"] += 0.02
        else:
            self.learner_state["fail_streak"] += 1
            delta = 0.1
            if difficulty == "Hard" and current_mastery < 0.5:
                delta += 0.05
            self.learner_state["frustration"] = min(1.0, self.learner_state["frustration"] + delta)

        self.diff_counts[action["difficulty_idx"]] += 1
        self.question_total += 1
        if correct:
            self.question_correct += 1

        response_time = max(5.0, np.random.normal(question["response_time_mean"], question["response_time_std"]))
        self.learner_state["response_time"] = response_time / 120.0

        return {
            "type": "question",
            "correct": correct,
            "mastery_gain": self.learner_state["mastery"][lo] - current_mastery,
            "frustration": self.learner_state["frustration"],
            "fail_streak": self.learner_state["fail_streak"],
        }

    def _execute_content(self, action: Dict) -> Dict:
        lo = action["lo"]
        modality = action["modality"]
        content = self._sample_content(lo, modality)
        pre_mastery = self.learner_state["mastery"][lo]
        base_gain = content["effectiveness"]
        effective_gain = base_gain * (1 - pre_mastery)
        frustration_penalty = self.learner_state["frustration"] * 0.5
        effective_gain *= (1 - frustration_penalty)
        noise = np.random.normal(0, 0.02)
        final_gain = max(0.0, effective_gain + noise)

        self.learner_state["mastery"][lo] = min(1.0, pre_mastery + final_gain)
        post_mastery = self.learner_state["mastery"][lo]

        frustration_delta = content["engagement_impact"]
        self.learner_state["frustration"] = float(np.clip(self.learner_state["frustration"] + frustration_delta, 0, 1))
        self.learner_state["fail_streak"] = 0
        self.content_count += 1

        return {
            "type": "content",
            "mastery_gain": post_mastery - pre_mastery,
            "frustration_delta": frustration_delta,
            "frustration": self.learner_state["frustration"],
        }

    def _compute_reward(self, result: Dict) -> float:
        reward = 0.0
        if result["type"] == "question":
            if result["correct"]:
                reward += 1.0
            reward += 0.5 * result["mastery_gain"]
            reward -= 0.3 * result["frustration"]
            reward -= self._blueprint_penalty()
        else:
            reward += 2.0 * result["mastery_gain"]
            reward += 0.5 * (-result["frustration_delta"])
        return float(reward)

    def _blueprint_penalty(self) -> float:
        total_q = sum(self.diff_counts)
        if total_q == 0:
            return 0.0
        proportions = [c / total_q for c in self.diff_counts]
        target = self.cfg.blueprint_target
        deviation = sum(abs(p - t) for p, t in zip(proportions, target))
        return self.cfg.blueprint_penalty * deviation

    def _is_terminal(self) -> Tuple[bool, str | None]:
        mean_mastery = float(np.mean(self.learner_state["mastery"]))
        if mean_mastery >= self.cfg.mastery_threshold:
            if self.time_to_mastery is None:
                self.time_to_mastery = self.step_count
            return True, "mastery_achieved"
        if self.step_count >= self.cfg.max_steps:
            return True, "step_limit"
        if self.learner_state["frustration"] >= self.cfg.critical_frustration:
            return True, "critical_frustration"
        return False, None

    def _get_observation(self) -> np.ndarray:
        mastery = self.learner_state["mastery"]
        frustration = np.array([np.clip(self.learner_state["frustration"], 0, 1)])
        response_time = np.array([np.clip(self.learner_state["response_time"], 0, 1)])
        return np.concatenate([mastery, frustration, response_time]).astype(np.float32)

    def get_difficulty_proportions(self) -> List[float]:
        total_q = sum(self.diff_counts)
        if total_q == 0:
            return [0.0, 0.0, 0.0]
        return [c / total_q for c in self.diff_counts]

    def get_episode_metrics(self) -> Dict:
        if not self.episode_log:
            return {}
        
        # Compute post-content gains by modality
        modality_gains = self._compute_modality_gains()
        
        # Compute calibration data
        calibration_data = self._compute_calibration_data()
        
        # Get difficulty proportions
        diff_props = self.get_difficulty_proportions()
        
        return {
            "total_steps": self.step_count,
            "final_mastery": float(np.mean(self.learner_state["mastery"])),
            "cumulative_reward": float(self.cumulative_reward),
            "question_accuracy": float(self.question_correct / self.question_total) if self.question_total else 0.0,
            "question_total": self.question_total,
            "question_correct": self.question_correct,
            "content_count": self.content_count,
            "blueprint_adherence": self._compute_blueprint_adherence(),
            "blueprint_proportions": diff_props,
            "time_to_mastery": self.time_to_mastery,
            "mean_frustration": self._compute_mean_frustration(),
            "final_frustration": float(self.learner_state["frustration"]),
            "modality_gains": modality_gains,
            "calibration_data": calibration_data,
        }

    def _compute_blueprint_adherence(self) -> float:
        total_q = sum(self.diff_counts)
        if total_q == 0:
            return 1.0
        proportions = [c / total_q for c in self.diff_counts]
        target = self.cfg.blueprint_target
        deviation = sum(abs(p - t) for p, t in zip(proportions, target))
        return max(0.0, 1.0 - 0.5 * deviation)
    
    def _compute_mean_frustration(self) -> float:
        """Compute mean frustration over episode"""
        if not self.episode_log:
            return 0.0
        frustrations = [entry.get("result", {}).get("frustration", 0.0) for entry in self.episode_log if "result" in entry]
        return float(np.mean(frustrations)) if frustrations else 0.0
    
    def _compute_modality_gains(self) -> Dict[str, Dict[str, float]]:
        """Compute post-content gain by modality (mean and std)"""
        modalities = ["video", "PPT", "text", "blog", "article", "handout"]
        modality_gains = {mod: [] for mod in modalities}
        
        for entry in self.episode_log:
            result = entry.get("result", {})
            if result.get("type") == "content":
                # Find modality from action
                action = entry.get("action")
                if action >= 90:
                    content_id = action - 90
                    modality_idx = content_id % 6
                    modality = modalities[modality_idx]
                    gain = result.get("mastery_gain", 0.0)
                    modality_gains[modality].append(gain)
        
        # Compute statistics per modality
        stats = {}
        for mod, gains in modality_gains.items():
            if gains:
                stats[mod] = {
                    "mean": float(np.mean(gains)),
                    "std": float(np.std(gains)),
                    "count": len(gains)
                }
            else:
                stats[mod] = {"mean": 0.0, "std": 0.0, "count": 0}
        
        return stats
    
    def _compute_calibration_data(self) -> Dict[str, List[float]]:
        """Collect predicted mastery vs empirical correctness for calibration plots"""
        return {
            "predicted_mastery": self.calibration_predicted.copy(),
            "empirical_correct": self.calibration_actual.copy()
        }


# -----------------------
# Dynamics model ensemble
# -----------------------


class EnsembleMember(nn.Module):
    def __init__(self, state_dim: int = 32, action_dim: int = 270, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.state_mean = nn.Linear(hidden, state_dim)
        self.state_logvar = nn.Linear(hidden, state_dim)
        self.reward_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, action_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # action_id is (batch,) of ints
        action_onehot = F.one_hot(action_id.long(), num_classes=270).float()
        x = torch.cat([state, action_onehot], dim=-1)
        feat = self.net(x)
        next_state_mean = self.state_mean(feat)
        next_state_logvar = self.state_logvar(feat)
        reward = self.reward_head(feat).squeeze(-1)
        return next_state_mean, next_state_logvar, reward


# -----------------------
# Planner
# -----------------------


class FactorizedCategoricalCEM:
    def __init__(self, mpc_cfg: MPCConfig, ensemble: List[EnsembleMember], device: torch.device, get_diff_proportions: Callable[[], List[float]] | None = None):
        self.cfg = mpc_cfg
        self.ensemble = ensemble
        self.device = device
        self.d_sizes = {"gate": 2, "d": 3, "l": 30, "m": 6}
        self.blueprint_target = torch.tensor([0.2, 0.6, 0.2], device=device)
        self.mask_strength = 2.0  # higher -> stronger down-weight
        self.get_diff_proportions = get_diff_proportions
        self.last_diag: Dict = {}

    def plan(self, state: np.ndarray) -> int:
        horizon = self.cfg.horizon
        logits_gate = [torch.zeros(self.d_sizes["gate"], device=self.device) for _ in range(horizon)]
        logits_d = [torch.zeros(self.d_sizes["d"], device=self.device) for _ in range(horizon)]
        logits_l = [torch.zeros(self.d_sizes["l"], device=self.device) for _ in range(horizon)]
        logits_m = [torch.zeros(self.d_sizes["m"], device=self.device) for _ in range(horizon)]

        top_k = max(1, int(self.cfg.elite_fraction * self.cfg.candidates))
        best_seq: List[Tuple[int, int, int, int]] | None = None
        best_return = -1e9
        elite_means: List[float] = []
        disagreement_running: List[float] = []

        with torch.no_grad():
            for _ in range(self.cfg.iterations):
                sequences: List[List[Tuple[int, int, int, int]]] = []
                returns: List[float] = []
                disagreements_iter: List[float] = []
                for _ in range(self.cfg.candidates):
                    s = torch.tensor(state, device=self.device, dtype=torch.float32)
                    total_return = 0.0
                    seq: List[Tuple[int, int, int, int]] = []
                    for h in range(self.cfg.horizon):
                        gate = torch.distributions.Categorical(logits=logits_gate[h]).sample().item()
                        d_logits = logits_d[h] - self._difficulty_mask()
                        d = torch.distributions.Categorical(logits=d_logits).sample().item()
                        l = torch.distributions.Categorical(logits=logits_l[h]).sample().item()
                        m = torch.distributions.Categorical(logits=logits_m[h]).sample().item()
                        seq.append((gate, d, l, m))
                        action_id = flatten_action(gate, d, l, m)
                        # Ensemble predictions for disagreement (optional penalty)
                        preds_mean = []
                        preds_reward = []
                        for member in self.ensemble:
                            pm, plogvar, pr = member(
                                s.unsqueeze(0), torch.tensor([action_id], device=self.device)
                            )
                            preds_mean.append(pm)
                            preds_reward.append(pr)

                        disagreement = torch.stack(preds_mean).std(dim=0).mean().item()
                        disagreements_iter.append(disagreement)

                        # Sample next state from a random ensemble member (stochastic rollout)
                        member = random.choice(self.ensemble)
                        ns_mean, ns_logvar, r_pred = member(
                            s.unsqueeze(0), torch.tensor([action_id], device=self.device)
                        )
                        ns_logvar = ns_logvar.clamp(*MODEL_CONFIG.logvar_clamp)
                        std = torch.exp(0.5 * ns_logvar)
                        s = ns_mean + torch.randn_like(ns_mean) * std
                        s = s.squeeze(0)

                        penalized_reward = r_pred.item() - self.cfg.uncertainty_penalty * disagreement
                        total_return += (self.cfg.gamma ** h) * penalized_reward

                    sequences.append(seq)
                    returns.append(total_return)

                elite_idx = np.argsort(returns)[-top_k:]
                elite_returns = [returns[i] for i in elite_idx]
                elite_means.append(float(np.mean(elite_returns)))
                elites = [sequences[i] for i in elite_idx]
                for h in range(horizon):
                    freq_gate = torch.zeros(self.d_sizes["gate"], device=self.device)
                    freq_d = torch.zeros(self.d_sizes["d"], device=self.device)
                    freq_l = torch.zeros(self.d_sizes["l"], device=self.device)
                    freq_m = torch.zeros(self.d_sizes["m"], device=self.device)
                    for seq in elites:
                        gate, d, l, m = seq[h]
                        freq_gate[gate] += 1
                        freq_d[d] += 1
                        freq_l[l] += 1
                        freq_m[m] += 1
                    freq_gate = freq_gate / top_k
                    freq_d = freq_d / top_k
                    freq_l = freq_l / top_k
                    freq_m = freq_m / top_k
                    logits_gate[h] = (1 - self.cfg.update_rate) * logits_gate[h] + self.cfg.update_rate * torch.log(freq_gate + self.cfg.smoothing_eps)
                    logits_d[h] = (1 - self.cfg.update_rate) * logits_d[h] + self.cfg.update_rate * torch.log(freq_d + self.cfg.smoothing_eps)
                    logits_l[h] = (1 - self.cfg.update_rate) * logits_l[h] + self.cfg.update_rate * torch.log(freq_l + self.cfg.smoothing_eps)
                    logits_m[h] = (1 - self.cfg.update_rate) * logits_m[h] + self.cfg.update_rate * torch.log(freq_m + self.cfg.smoothing_eps)

                if disagreements_iter:
                    disagreement_running.append(float(np.mean(disagreements_iter)))

                best_idx = elite_idx[-1]
                if returns[best_idx] > best_return:
                    best_return = returns[best_idx]
                    best_seq = sequences[best_idx]

        if best_seq is None:
            return random.randint(0, 269)
        gate0, d0, l0, m0 = best_seq[0]
        # Save diagnostics
        self.last_diag = {
            "elite_mean_per_iter": elite_means,
            "disagreement_mean": float(np.mean(disagreement_running)) if disagreement_running else None,
        }
        return flatten_action(gate0, d0, l0, m0)

    def _difficulty_mask(self) -> torch.Tensor:
        if self.get_diff_proportions is None:
            return torch.zeros(self.d_sizes["d"], device=self.device)
        props = self.get_diff_proportions()
        props_t = torch.tensor(props, device=self.device)
        # Penalize over-used difficulties relative to target
        overuse = torch.relu(props_t - self.blueprint_target)
        return self.mask_strength * overuse


# -----------------------
# Training utilities
# -----------------------


def flatten_action(gate: int, difficulty: int, lo: int, modality: int) -> int:
    # gate=0 -> question (uses difficulty), gate=1 -> content (uses modality)
    if gate == 0:
        return lo * 3 + difficulty
    return 90 + lo * 6 + modality


def collect_episode(env: AdaptiveLearningEnv, policy, dataset: List[Tuple[np.ndarray, int, float, np.ndarray]], random_policy: bool = False) -> Tuple[float, float, Dict]:
    start = time.time()
    obs = env.reset()
    done = False
    episode_return = 0.0
    while not done:
        action = random.randint(0, 269) if random_policy else policy(obs)
        next_obs, reward, done, _ = env.step(action)
        dataset.append((obs, action, reward, next_obs))
        obs = next_obs
        episode_return += reward
    duration = time.time() - start
    metrics = env.get_episode_metrics()
    metrics["duration_s"] = duration
    return episode_return, duration, metrics


def train_ensemble(ensemble: List[EnsembleMember], dataset: List[Tuple[np.ndarray, int, float, np.ndarray]], model_cfg: ModelConfig, device: torch.device) -> None:
    if not dataset:
        return
    states = torch.tensor(np.stack([d[0] for d in dataset]), device=device, dtype=torch.float32)
    actions = torch.tensor([d[1] for d in dataset], device=device, dtype=torch.long)
    rewards = torch.tensor([d[2] for d in dataset], device=device, dtype=torch.float32)
    next_states = torch.tensor(np.stack([d[3] for d in dataset]), device=device, dtype=torch.float32)

    optimizers = [torch.optim.Adam(m.parameters(), lr=model_cfg.learning_rate, weight_decay=model_cfg.weight_decay) for m in ensemble]

    for epoch in range(model_cfg.train_epochs):
        perm = torch.randperm(len(dataset), device=device)
        for start in range(0, len(dataset), model_cfg.batch_size):
            idx = perm[start:start + model_cfg.batch_size]
            s_b = states[idx]
            a_b = actions[idx]
            r_b = rewards[idx]
            ns_b = next_states[idx]
            for model, opt in zip(ensemble, optimizers):
                opt.zero_grad()
                pred_mean, pred_logvar, pred_reward = model(s_b, a_b)
                pred_logvar = pred_logvar.clamp(*model_cfg.logvar_clamp)
                inv_var = torch.exp(-pred_logvar)
                state_loss = 0.5 * (pred_logvar + (ns_b - pred_mean) ** 2 * inv_var).mean()
                reward_loss = F.mse_loss(pred_reward, r_b)
                loss = state_loss + reward_loss
                loss.backward()
                opt.step()


def generate_dataset(num_learners: int = 200, episodes_per_learner: int = 1, seed_offset: int = 0) -> List[Dict]:
    env = AdaptiveLearningEnv(ENV_CONFIG)
    data: List[Dict] = []
    for learner_id in range(num_learners):
        for ep in range(episodes_per_learner):
            seed = seed_offset + learner_id * episodes_per_learner + ep
            obs = env.reset(seed=seed)
            done = False
            while not done:
                action = random.randint(0, 269)
                next_obs, reward, done, info = env.step(action)
                data.append({
                    "learner_id": learner_id,
                    "episode": ep,
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": done,
                    "info": info,
                })
                obs = next_obs
    return data


def validate_simulator() -> None:
    env = AdaptiveLearningEnv(ENV_CONFIG)
    obs = env.reset()
    assert obs.shape == (32,), "State shape mismatch"
    assert np.all(obs >= 0) and np.all(obs <= 1), "State out of bounds"

    # Action validity
    for action in range(270):
        _, reward, done, _ = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        if done:
            env.reset()

    # Mastery progression sanity: force correct question
    env.reset()
    before = env.learner_state["mastery"][0]
    env._execute_question({"lo": 0, "difficulty": "Easy", "difficulty_idx": 0})
    after = env.learner_state["mastery"][0]
    assert after >= before, "Mastery did not increase on correct answer path"

    print("✓ Simulator validation checks passed")


# -----------------------
# Additional Metrics for Fair Comparison
# -----------------------


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


# -----------------------
# Main training loop
# -----------------------


def export_results_for_paper(
    mean_curve, std_curve, all_episode_metrics, all_modality_gains,
    all_calibration_predicted, all_calibration_actual,
    all_seed_returns, all_seed_mastery_steps,
    mastery_mean, mastery_std, reward_mean, reward_std, boot_ci,
    time_mean, time_std,
    all_seed_auc: List[float] = None, all_seed_checkpoints: List[Dict] = None  # NEW
) -> None:
    """Export comprehensive results matching paper requirements"""
    import json
    import csv
    import os
    
    output_dir = "results/pets"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Learning curve data (for Figure: learning_curve.png)
    learning_curve_data = {
        "episodes": list(range(len(mean_curve))),
        "mean_reward": mean_curve.tolist(),
        "std_reward": std_curve.tolist(),
    }
    with open(os.path.join(output_dir, "learning_curve_data.json"), "w") as f:
        json.dump(learning_curve_data, f, indent=2)
    print("✓ Exported learning_curve_data.json")
    
    # 2. Performance summary table (Table: tab:perf_summary)
    # Compute aggregate metrics from all_episode_metrics
    final_episode_metrics = [m for m in all_episode_metrics if m.get("episode", 0) >= TRAIN_CONFIG.total_episodes - 5]
    
    # NEW: Aggregate AUC@10k and checkpoints across seeds to match DQN/MBPO/PPO standard format
    auc_10k_mean = float(np.mean(all_seed_auc)) if all_seed_auc else 0.0
    auc_10k_std = float(np.std(all_seed_auc)) if all_seed_auc else 0.0
    
    # Aggregate checkpoints: average each checkpoint's metrics across seeds
    aggregated_checkpoints = {}
    if all_seed_checkpoints:
        all_checkpoint_keys = set()
        for seed_ckpt in all_seed_checkpoints:
            all_checkpoint_keys.update(seed_ckpt.keys())
        for ckpt in sorted(all_checkpoint_keys):
            values = [seed_ckpt.get(ckpt, {}).get("cumulative_reward", 0.0) for seed_ckpt in all_seed_checkpoints if ckpt in seed_ckpt]
            if values:
                aggregated_checkpoints[ckpt] = float(np.mean(values))
    
    # Compute std for metrics across final episodes
    qa_values = [m.get("question_accuracy", 0) for m in final_episode_metrics] if final_episode_metrics else []
    ba_values = [m.get("blueprint_adherence", 0) for m in final_episode_metrics] if final_episode_metrics else []
    fr_values = [m.get("mean_frustration", 0) for m in final_episode_metrics] if final_episode_metrics else []
    fm_values = [m.get("final_mastery", 0) for m in final_episode_metrics] if final_episode_metrics else []
    pcg_values = [m.get("post_content_gain", 0) for m in final_episode_metrics] if final_episode_metrics else []
    
    perf_summary = {
        "auc_10k": {
            "mean": auc_10k_mean,
            "std": auc_10k_std,
        },
        "wall_clock_time_minutes": {
            "mean": time_mean / 60.0,
            "std": time_std / 60.0,
        },
        "checkpoints": aggregated_checkpoints,
        "time_to_mastery": {
            "mean": mastery_mean,
            "std": mastery_std,
        },
        "cumulative_reward": {
            "mean": reward_mean,
            "std": reward_std,
            "ci_95": [boot_ci[0], boot_ci[1]],
        },
        "question_accuracy": {
            "mean": float(np.mean(qa_values)) if qa_values else 0.0,
            "std": float(np.std(qa_values)) if qa_values else 0.0,
        },
        "blueprint_adherence": {
            "mean": float(np.mean(ba_values)) if ba_values else 0.0,
            "std": float(np.std(ba_values)) if ba_values else 0.0,
        },
        "post_content_gain": {
            "mean": float(np.mean(pcg_values)) if pcg_values else 0.0,
            "std": float(np.std(pcg_values)) if pcg_values else 0.0,
        },
        "mean_frustration": {
            "mean": float(np.mean(fr_values)) if fr_values else 0.0,
            "std": float(np.std(fr_values)) if fr_values else 0.0,
        },
        "final_mastery": {
            "mean": float(np.mean(fm_values)) if fm_values else 0.0,
            "std": float(np.std(fm_values)) if fm_values else 0.0,
        },
        "num_seeds": len(TRAIN_CONFIG.seeds),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(perf_summary, f, indent=2)
    print("✓ Exported summary.json")
    
    # 3. Modality gains (for Figure: modality_gains.png and Table: tab:modality_gain)
    modality_stats = {}
    for mod, gains in all_modality_gains.items():
        if gains:
            modality_stats[mod] = {
                "mean": float(np.mean(gains)),
                "std": float(np.std(gains)),
                "count": len(gains)
            }
        else:
            modality_stats[mod] = {"mean": 0.0, "std": 0.0, "count": 0}
    
    with open(os.path.join(output_dir, "modality_gains.json"), "w") as f:
        json.dump(modality_stats, f, indent=2)
    print("✓ Exported modality_gains.json")
    
    # 4. Calibration data (for Figure: calibration.png)
    if all_calibration_predicted and all_calibration_actual:
        calibration_data = {
            "predicted_mastery": [float(x) for x in all_calibration_predicted],
            "empirical_correct": [float(x) for x in all_calibration_actual]
        }
        with open(os.path.join(output_dir, "calibration_data.json"), "w") as f:
            json.dump(calibration_data, f, indent=2)
        print("✓ Exported calibration_data.json")
    
    # 5. Variance data (for Figure: variance_bands_all.png)
    variance_data = {
        "seed_returns": [r.tolist() if hasattr(r, 'tolist') else list(r) for r in all_seed_returns],
        "episodes": list(range(len(mean_curve)))
    }
    with open(os.path.join(output_dir, "variance_data.json"), "w") as f:
        json.dump(variance_data, f, indent=2)
    print("✓ Exported variance_data.json")
    
    # 6. LaTeX table fragments for direct inclusion
    generate_latex_tables(perf_summary, modality_stats, output_dir)
    
    print("\\n" + "="*60)
    print("RESULTS EXPORT COMPLETE")
    print("="*60)
    print("Files saved to results/ directory:")
    print("  - learning_curve_data.json")
    print("  - performance_summary.json")
    print("  - modality_gains.json")
    print("  - calibration_data.json")
    print("  - variance_data.json")
    print("  - table_ppo_perf.tex (LaTeX table)")
    print("  - table_modality.tex (LaTeX table)")
    print("\\nUse these files to generate figures and tables for the paper.")

    # 7. Episodes CSV (align columns with DQN/PPO/MBPO)
    # Build canonical header with modality breakdown right after post_content_gain
    episodes_csv_path = os.path.join("results", "pets", "episodes.csv")
    os.makedirs(os.path.dirname(episodes_csv_path), exist_ok=True)
    with open(episodes_csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "seed", "episode", "return", "cumulative_reward", "ttm",
            "total_steps", "question_accuracy", "content_rate", "blueprint_adherence",
            "post_content_gain",
            "post_content_gain_video", "post_content_gain_PPT", "post_content_gain_text",
            "post_content_gain_blog", "post_content_gain_article", "post_content_gain_handout",
            "final_mastery", "mean_frustration"
        ])

        # Iterate through all_episode_metrics (passed in) and write rows
        for em in all_episode_metrics:
            seed = em.get("seed")
            modality_stats = em.get("modality_gains", {})
            # Compute overall post_content_gain as count-weighted mean across modalities
            total_count = 0
            weighted_sum = 0.0
            modal_means = {
                "video": 0.0, "PPT": 0.0, "text": 0.0,
                "blog": 0.0, "article": 0.0, "handout": 0.0
            }
            for mod in modal_means.keys():
                stats = modality_stats.get(mod, {})
                m = float(stats.get("mean", 0.0))
                c = int(stats.get("count", 0))
                modal_means[mod] = m
                weighted_sum += m * c
                total_count += c
            overall_pcg = (weighted_sum / total_count) if total_count > 0 else 0.0

            # Compute content_rate from counts if not present
            content_rate = em.get("content_rate")
            if content_rate is None:
                steps = int(em.get("total_steps", 0))
                ccount = int(em.get("content_count", 0))
                content_rate = (ccount / steps) if steps > 0 else 0.0

            writer.writerow([
                seed,
                em.get("episode", 0),
                em.get("return", 0.0),
                em.get("cumulative_reward", 0.0),
                em.get("time_to_mastery", 0),  # mapped to CSV column 'ttm'
                em.get("total_steps", 0),
                em.get("question_accuracy", 0.0),
                content_rate,
                em.get("blueprint_adherence", 0.0),
                overall_pcg,
                modal_means["video"], modal_means["PPT"], modal_means["text"],
                modal_means["blog"], modal_means["article"], modal_means["handout"],
                em.get("final_mastery", 0.0),
                em.get("mean_frustration", 0.0),
            ])
    print("✓ Exported episodes.csv")


def generate_latex_tables(perf_summary, modality_stats, output_dir):
    """Generate LaTeX table fragments for direct inclusion in paper"""
    
    # Performance summary table
    latex_perf = f"""% Auto-generated PETS performance table
\\\\begin{{tabular}}{{lc}}
\\\\toprule
\\\\textbf{{Metric}} & \\\\textbf{{PETS}} \\\\\\\\
\\\\midrule
Time-to-Mastery (steps) & {perf_summary['time_to_mastery_mean']:.1f} $\\\\pm$ {perf_summary['time_to_mastery_std']:.1f} \\\\\\\\
Cumulative Reward & {perf_summary['cumulative_reward_mean']:.2f} $\\\\pm$ {perf_summary['cumulative_reward_std']:.2f} \\\\\\\\
Question Accuracy (\\\\%) & {perf_summary['question_accuracy_mean']*100:.1f} \\\\\\\\
Blueprint Adherence (\\\\%) & {perf_summary['blueprint_adherence_mean']*100:.1f} \\\\\\\\
Mean Frustration & {perf_summary['mean_frustration']:.3f} \\\\\\\\
Final Mastery & {perf_summary['final_mastery_mean']:.3f} \\\\\\\\
Wall-Clock Time (s) & {perf_summary['wall_clock_mean_s']:.1f} $\\\\pm$ {perf_summary['wall_clock_std_s']:.1f} \\\\\\\\
\\\\bottomrule
\\\\end{{tabular}}
"""
    with open(os.path.join(output_dir, "table_ppo_perf.tex"), "w") as f:
        f.write(latex_perf)
    
    # Modality gains table
    latex_mod = """% Auto-generated modality gains table
\\\\begin{tabular}{lcc}
\\\\toprule
\\\\textbf{Modality} & \\\\textbf{Mean Gain} & \\\\textbf{Std Dev} \\\\\\\\
\\\\midrule
"""
    for mod in ["video", "PPT", "text", "blog", "article", "handout"]:
        stats = modality_stats.get(mod, {"mean": 0, "std": 0})
        latex_mod += f"{mod} & {stats['mean']:.3f} & {stats['std']:.3f} \\\\\\\\\n"
    latex_mod += """\\\\bottomrule
\\\\end{tabular}
"""
    with open(os.path.join(output_dir, "table_modality.tex"), "w") as f:
        f.write(latex_mod)
    
    print("✓ Generated LaTeX tables: table_ppo_perf.tex, table_modality.tex")


def main() -> None:
    print(f"\n{'='*70}")
    print(f"🚀 PETS TRAINING STARTED")
    print(f"{'='*70}")
    print(f"Seeds: {TRAIN_CONFIG.seeds}")
    print(f"Total episodes: {TRAIN_CONFIG.total_episodes}")
    print(f"Device: {TRAIN_CONFIG.device}")
    print(f"{'='*70}\n")
    
    device = torch.device(TRAIN_CONFIG.device)
    all_seed_returns: List[List[float]] = []
    all_seed_mastery_steps: List[float] = []
    all_seed_episode_steps: List[List[int]] = []  # NEW: Track steps per episode for AUC/checkpoints
    all_seed_auc: List[float] = []  # NEW: Store AUC@10k per seed
    all_seed_checkpoints: List[Dict] = []  # NEW: Store checkpoints per seed
    seed_durations: List[float] = []
    episode_durations: List[float] = []
    cem_elite_logs: List[float] = []
    cem_disagreement_logs: List[float] = []
    
    # NEW: Comprehensive cross-seed metrics for paper
    all_seed_metrics: List[Dict] = []
    all_episode_metrics: List[Dict] = []
    all_calibration_predicted: List[float] = []
    all_calibration_actual: List[float] = []
    all_modality_gains: Dict[str, List[float]] = {mod: [] for mod in ["video", "PPT", "text", "blog", "article", "handout"]}

    for seed in TRAIN_CONFIG.seeds:
        print(f"\n{'='*70}")
        print(f"📍 SEED {seed} STARTING")
        print(f"{'='*70}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"✅ Random seeds set")
        print(f"🏗️  Creating environment...")
        env = AdaptiveLearningEnv(ENV_CONFIG)
        print(f"✅ Environment created")
        ensemble = [EnsembleMember(hidden=MODEL_CONFIG.hidden_dim).to(device) for _ in range(MODEL_CONFIG.ensemble_size)]
        planner = FactorizedCategoricalCEM(MPC_CONFIG, ensemble, device, get_diff_proportions=env.get_difficulty_proportions)

        dataset: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
        returns: List[float] = []
        mastery_steps: List[float] = []
        episode_steps: List[int] = []  # NEW: Track steps per episode

        seed_start = time.time()
        print(f"🏁 Initial exploration: {TRAIN_CONFIG.initial_exploration} episodes")

        for exp_ep in range(TRAIN_CONFIG.initial_exploration):
            print(f"   [EXPLORE] Episode {exp_ep+1}/{TRAIN_CONFIG.initial_exploration}", end="", flush=True)
            ret, dur, metrics = collect_episode(env, lambda _: 0, dataset, random_policy=True)
            returns.append(ret)
            episode_durations.append(dur)
            episode_steps.append(metrics.get("total_steps", 0))  # NEW
            if metrics.get("time_to_mastery") is not None:
                mastery_steps.append(metrics["time_to_mastery"])
            print(f" ✅ Ret: {ret:.2f}, TTM: {metrics.get('time_to_mastery', 'N/A')}")

        print(f"🎓 Training: {TRAIN_CONFIG.total_episodes - TRAIN_CONFIG.initial_exploration} episodes")
        for ep in range(TRAIN_CONFIG.initial_exploration, TRAIN_CONFIG.total_episodes):
            print(f"   [TRAIN] Episode {ep+1}/{TRAIN_CONFIG.total_episodes}", end="", flush=True)
            try:
                train_ensemble(ensemble, dataset, MODEL_CONFIG, device)
                ret, dur, metrics = collect_episode(env, lambda obs: planner.plan(obs), dataset, random_policy=False)
                returns.append(ret)
                episode_durations.append(dur)
                episode_steps.append(metrics.get("total_steps", 0))  # NEW
                print(f" ✅ Ret: {ret:.2f}, TTM: {metrics.get('time_to_mastery', 'N/A')}, Steps: {metrics.get('total_steps', 0)}")
            except Exception as e:
                print(f" ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # NEW: Collect comprehensive metrics (include per-episode return for CSV export)
            all_episode_metrics.append({**metrics, "seed": seed, "episode": ep, "return": ret})
            
            # Collect modality gains
            if "modality_gains" in metrics:
                for mod, stats in metrics["modality_gains"].items():
                    if stats["count"] > 0:
                        all_modality_gains[mod].append(stats["mean"])
            
            # Collect calibration data
            if "calibration_data" in metrics:
                all_calibration_predicted.extend(metrics["calibration_data"]["predicted_mastery"])
                all_calibration_actual.extend(metrics["calibration_data"]["empirical_correct"])
            
            if metrics.get("time_to_mastery") is not None:
                mastery_steps.append(metrics["time_to_mastery"])
            diag = getattr(planner, "last_diag", {})
            if diag.get("elite_mean_per_iter"):
                cem_elite_logs.append(float(np.mean(diag["elite_mean_per_iter"])))
            if diag.get("disagreement_mean") is not None:
                cem_disagreement_logs.append(diag["disagreement_mean"])
            if (ep + 1) % 5 == 0:
                mean_last = np.mean(returns[-5:])
                print(f"Seed {seed} | Episode {ep+1}/{TRAIN_CONFIG.total_episodes} | recent avg return: {mean_last:.2f} | dataset size: {len(dataset)}")

        # NEW: Compute per-seed AUC and checkpoints
        auc_10k = compute_auc_at_10k(returns, episode_steps)
        checkpoints = compute_checkpoint_metrics(returns, all_episode_metrics, episode_steps)
        
        all_seed_returns.append(returns)
        all_seed_episode_steps.append(episode_steps)  # NEW
        all_seed_auc.append(auc_10k)  # NEW: Store per-seed AUC
        all_seed_checkpoints.append(checkpoints)  # NEW: Store per-seed checkpoints
        if mastery_steps:
            all_seed_mastery_steps.append(float(np.mean(mastery_steps)))
        seed_durations.append(time.time() - seed_start)
        print(f"Seed {seed} complete. Final recent-5 avg return: {np.mean(returns[-5:]):.2f} | AUC@10k: {auc_10k:.1f}")

    # Aggregate across seeds
    max_len = max(len(r) for r in all_seed_returns)
    padded = [np.pad(r, (0, max_len - len(r)), constant_values=np.nan) for r in all_seed_returns]
    arr = np.vstack(padded)
    mean_curve = np.nanmean(arr, axis=0)
    std_curve = np.nanstd(arr, axis=0)
    last5_mean = np.nanmean(mean_curve[-5:])
    last5_std = np.nanmean(std_curve[-5:])
    n = len(TRAIN_CONFIG.seeds)
    ci95 = 1.96 * last5_std / math.sqrt(n) if n > 0 else float("nan")
    # Bootstrap CI on last-5 mean across seeds
    boot_means = []
    boot_iters = 1000
    if n > 0:
        seed_means = []
        for r in all_seed_returns:
            m = np.nanmean(r[-5:]) if len(r) else np.nan
            seed_means.append(m)
        seed_means = np.array(seed_means)
        for _ in range(boot_iters):
            sample = np.random.choice(seed_means, size=n, replace=True)
            boot_means.append(np.nanmean(sample))
    boot_ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))) if boot_means else (float("nan"), float("nan"))
    mastery_mean = float(np.mean(all_seed_mastery_steps)) if all_seed_mastery_steps else float("nan")
    mastery_std = float(np.std(all_seed_mastery_steps)) if all_seed_mastery_steps else float("nan")
    time_mean = float(np.mean(seed_durations)) if seed_durations else float("nan")
    time_std = float(np.std(seed_durations)) if seed_durations else float("nan")
    ep_time_mean = float(np.mean(episode_durations)) if episode_durations else float("nan")
    ep_time_std = float(np.std(episode_durations)) if episode_durations else float("nan")
    cem_elite_mean = float(np.mean(cem_elite_logs)) if cem_elite_logs else float("nan")
    cem_elite_std = float(np.std(cem_elite_logs)) if cem_elite_logs else float("nan")
    cem_disagree_mean = float(np.mean(cem_disagreement_logs)) if cem_disagreement_logs else float("nan")
    cem_disagree_std = float(np.std(cem_disagreement_logs)) if cem_disagreement_logs else float("nan")

    print("Across seeds: mean±std of last 5 episodes:", f"{last5_mean:.2f} ± {last5_std:.2f}")
    print("95% CI (normal approx) on last-5 mean:", f"{last5_mean:.2f} ± {ci95:.2f}")
    print("Bootstrap 95% CI on last-5 mean:", f"[{boot_ci[0]:.2f}, {boot_ci[1]:.2f}]")
    print("Time-to-mastery (steps) mean±std (per seed avg):", f"{mastery_mean:.2f} ± {mastery_std:.2f}")
    print("Wall-clock per seed (s) mean±std:", f"{time_mean:.2f} ± {time_std:.2f}")
    print("Wall-clock per episode (s) mean±std:", f"{ep_time_mean:.2f} ± {ep_time_std:.2f}")
    print("CEM elite return (per-iter mean) mean±std:", f"{cem_elite_mean:.2f} ± {cem_elite_std:.2f}")
    print("Ensemble disagreement mean±std:", f"{cem_disagree_mean:.4f} ± {cem_disagree_std:.4f}")
    
    # NEW: Export comprehensive results for paper
    export_results_for_paper(
        mean_curve, std_curve, all_episode_metrics, all_modality_gains,
        all_calibration_predicted, all_calibration_actual, 
        all_seed_returns, all_seed_mastery_steps,
        mastery_mean, mastery_std, last5_mean, last5_std, boot_ci,
        time_mean, time_std,
        all_seed_auc=all_seed_auc, all_seed_checkpoints=all_seed_checkpoints  # NEW
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PETS training runner")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run (overrides TRAIN_CONFIG.seeds)")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes (overrides TRAIN_CONFIG.total_episodes)")
    parser.add_argument("--steps", type=int, default=None, help="Deprecated: use --episodes instead")
    cli_args = parser.parse_args()

    if cli_args.seed is not None:
        TRAIN_CONFIG.seeds = (cli_args.seed,)
    
    # Handle both --episodes and --steps arguments
    episodes_val = cli_args.episodes if cli_args.episodes is not None else cli_args.steps
    if episodes_val is not None:
        TRAIN_CONFIG.total_episodes = episodes_val

    main()
