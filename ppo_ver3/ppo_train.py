"""
PPO training script for adaptive mock-interview simulator.
Aligned with DQN/PETS/MBPO for 1:1 replication via shared_config.py
"""
import argparse
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
from torch.distributions import Categorical

# Import unified configuration for 1:1 replication
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared_config import UNIFIED_SEEDS, UNIFIED_EPISODES, UNIFIED_MAX_STEPS_PER_EPISODE

# Configuration pulled from specs
CONFIG = {
    "state_dim": 32,
    "action_dim": 270,
    "hidden_layers": [256, 256],
    "learning_rate": 3e-4,
    "discount_gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_loss_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "rollout_steps": 2048,
    "num_envs": 1,
    "epochs_per_update": 10,
    "minibatch_size": 64,
    "max_episodes": UNIFIED_EPISODES,  # 295 episodes (aligned with other algorithms)
    "max_steps_per_episode": UNIFIED_MAX_STEPS_PER_EPISODE,  # 140 steps max
    "device": "cpu",
    "use_value_clipping": True,
    "value_clip_epsilon": 0.2,
    "target_kl": 0.01,  # set to None to disable early stop
}

# -----------------------------
# Simulator
# -----------------------------


def _sample_range(low: float, high: float) -> float:
    return np.random.uniform(low, high)


class AdaptiveLearningEnv:
    def __init__(self):
        self.num_los = 30
        self.num_questions = 600  # 20 per LO across difficulties
        self.num_contents = 180   # 6 modalities × 30 LOs
        self.max_steps = CONFIG["max_steps_per_episode"]
        self.observation_space = (CONFIG["state_dim"],)
        self.action_space_n = CONFIG["action_dim"]
        self.reward_weights = {
            "correctness": 1.0,
            "mastery_gain": 0.5,
            "frustration_penalty": 0.3,
            "post_content_gain": 2.0,
            "engagement_bonus": 0.5,
        }
        self.blueprint_target = {0: 0.2, 1: 0.6, 2: 0.2}  # difficulty: easy/med/hard
        self.questions = self._load_questions()
        self.contents = self._load_contents()
        self.reset()

    def _load_questions(self):
        questions = []
        per_lo = 20
        per_diff = {0: int(0.2 * per_lo), 1: int(0.6 * per_lo), 2: per_lo - int(0.2 * per_lo) - int(0.6 * per_lo)}
        for lo in range(self.num_los):
            for difficulty, b_range, a_range in [
                (0, (-2.0, -0.5), (0.5, 1.0)),
                (1, (-0.5, 0.5), (1.0, 1.5)),
                (2, (0.5, 2.0), (1.5, 2.0)),
            ]:
                for _ in range(per_diff[difficulty]):
                    questions.append({
                        "lo": lo,
                        "difficulty": difficulty,
                        "irt_a": _sample_range(*a_range),
                        "irt_b": _sample_range(*b_range),
                        "irt_c": _sample_range(0.1, 0.25),
                        "response_time_mean": 30.0,
                        "response_time_std": 10.0,
                    })
        return questions

    def _load_contents(self):
        modalities = {
            0: (0.10, 0.15, -0.08),  # video
            1: (0.08, 0.12, -0.05),  # PPT
            2: (0.05, 0.08, 0.02),   # text
            3: (0.07, 0.10, -0.03),  # blog
            4: (0.06, 0.09, 0.0),    # article
            5: (0.05, 0.08, 0.05),   # handout
        }
        contents = []
        for lo in range(self.num_los):
            for modality, (low_gain, high_gain, frustr_impact) in modalities.items():
                contents.append({
                    "lo": lo,
                    "modality": modality,
                    "effectiveness": _sample_range(low_gain, high_gain),
                    "engagement_impact": frustr_impact,
                })
        return contents

    def reset(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        self.learner_state = {
            "mastery": np.random.beta(a=2, b=5, size=self.num_los),
            "ability": np.random.normal(0, 1),
            "frustration": 0.0,
            "response_time": 0.0,
            "fail_streak": 0,
            "engagement": 1.0,
        }
        self.step_count = 0
        self.episode_log: List[Dict] = []
        self.difficulty_counts = {0: 0, 1: 0, 2: 0}
        return self._get_observation()

    def _normalize_rt(self, rt: float) -> float:
        return min(max(rt / 120.0, 0.0), 1.0)

    def _get_observation(self):
        return np.concatenate([
            self.learner_state["mastery"],
            [self.learner_state["frustration"]],
            [self._normalize_rt(self.learner_state.get("response_time", 0.0))],
        ]).astype(np.float32)

    def _decode_action(self, action_id: int) -> Dict:
        if action_id < 90:
            lo = action_id // 3
            difficulty = action_id % 3
            return {"type": "question", "lo": lo, "difficulty": difficulty}
        content_id = action_id - 90
        lo = content_id // 6
        modality = content_id % 6
        return {"type": "content", "lo": lo, "modality": modality}

    def _get_question(self, lo: int, difficulty: int):
        candidates = [q for q in self.questions if q["lo"] == lo and q["difficulty"] == difficulty]
        return random.choice(candidates)

    def _get_content(self, lo: int, modality: int):
        candidates = [c for c in self.contents if c["lo"] == lo and c["modality"] == modality]
        return random.choice(candidates)

    def _sample_correctness(self, question: Dict) -> bool:
        theta = self.learner_state["ability"]
        a, b, c = question["irt_a"], question["irt_b"], question["irt_c"]
        prob_correct = c + (1 - c) / (1 + math.exp(-a * (theta - b)))
        return np.random.rand() < prob_correct

    def _sample_response_time(self, question: Dict) -> float:
        mean, std = question["response_time_mean"], question["response_time_std"]
        rt = np.random.normal(mean, std)
        return max(5.0, rt)

    def _update_frustration(self, correct: bool, difficulty: int, pre_mastery: float):
        if correct:
            self.learner_state["frustration"] = max(0.0, self.learner_state["frustration"] - 0.05)
        else:
            delta = 0.10
            if difficulty == 2 and pre_mastery < 0.5:
                delta += 0.05
            self.learner_state["frustration"] = min(1.0, self.learner_state["frustration"] + delta)

    def _compute_content_gain(self, content: Dict, pre_mastery: float) -> float:
        base_gain = content["effectiveness"]
        effective_gain = base_gain * (1 - pre_mastery)
        frustration_penalty = self.learner_state["frustration"] * 0.5
        effective_gain *= (1 - frustration_penalty)
        noise = np.random.normal(0, 0.02)
        return max(0.0, effective_gain + noise)

    def _compute_reward(self, result: Dict) -> float:
        rw = self.reward_weights
        reward = 0.0
        if result["type"] == "question":
            if result["correct"]:
                reward += rw["correctness"]
            reward += rw["mastery_gain"] * result.get("mastery_gain", 0.0)
            reward -= rw["frustration_penalty"] * self.learner_state["frustration"]
        else:
            reward += rw["post_content_gain"] * result.get("mastery_gain", 0.0)
            reward += rw["engagement_bonus"] * (-result.get("frustration_delta", 0.0))
        return float(reward)

    def _compute_blueprint_adherence(self) -> float:
        total = sum(self.difficulty_counts.values())
        if total == 0:
            return 100.0
        actual = {
            0: self.difficulty_counts[0] / total,
            1: self.difficulty_counts[1] / total,
            2: self.difficulty_counts[2] / total,
        }
        deviation = sum(abs(actual[d] - self.blueprint_target[d]) for d in self.blueprint_target) / len(self.blueprint_target)
        return (1.0 - deviation) * 100.0

    def _enforce_blueprint(self, lo: int, difficulty: int) -> Tuple[int, bool]:
        """Ensure 20/60/20 difficulty mix by redirecting to an under-target bucket if needed."""
        total_q = sum(self.difficulty_counts.values())
        if difficulty not in (0, 1, 2):
            return difficulty, False
        allowed = math.ceil(self.blueprint_target[difficulty] * max(1, total_q + 1))
        if self.difficulty_counts[difficulty] < allowed:
            return difficulty, False
        # find alternative difficulty under quota
        candidates = []
        for d in (0, 1, 2):
            quota = math.ceil(self.blueprint_target[d] * max(1, total_q + 1))
            if self.difficulty_counts[d] < quota:
                candidates.append(d)
        if not candidates:
            return difficulty, False
        new_diff = random.choice(candidates)
        return new_diff, True

    def _compute_accuracy(self) -> float:
        question_events = [e for e in self.episode_log if e["action_type"] == "question"]
        if not question_events:
            return 0.0
        correct = sum(1 for e in question_events if e.get("correct"))
        return correct / len(question_events)

    def _compute_content_rate(self) -> float:
        if not self.episode_log:
            return 0.0
        content_count = sum(1 for e in self.episode_log if e["action_type"] == "content")
        return content_count / len(self.episode_log)

    def _time_to_mastery(self, threshold: float = 0.8):
        for idx, step in enumerate(self.episode_log):
            mean_mastery = float(np.mean(step["mastery_vector"]))
            if mean_mastery >= threshold:
                return idx + 1
        return None

    def get_episode_metrics(self) -> Dict:
        if not self.episode_log:
            return {}
        question_events = [e for e in self.episode_log if e["action_type"] == "question"]
        content_events = [e for e in self.episode_log if e["action_type"] == "content"]

        post_content_gain = 0.0
        if content_events:
            post_content_gain = float(np.mean([e.get("mastery_gain", 0.0) for e in content_events]))

        mean_frustration = float(np.mean([e.get("frustration", 0.0) for e in self.episode_log]))
        return {
            "total_steps": self.step_count,
            "final_mastery": float(np.mean(self.learner_state["mastery"])),
            "cumulative_reward": float(sum(e["reward"] for e in self.episode_log)),
            "question_accuracy": self._compute_accuracy(),
            "content_rate": self._compute_content_rate(),
            "blueprint_adherence": self._compute_blueprint_adherence(),
            "post_content_gain": post_content_gain,
            "time_to_mastery": self._time_to_mastery(),
            "mean_frustration": mean_frustration,
        }

    def _is_terminal(self) -> Tuple[bool, str]:
        mean_mastery = float(np.mean(self.learner_state["mastery"]))
        if mean_mastery >= 0.8:
            return True, "mastery_achieved"
        if self.step_count >= self.max_steps:
            return True, "step_limit"
        if self.learner_state["frustration"] >= 0.95:
            return True, "critical_frustration"
        return False, ""

    def step(self, action_id: int):
        forced_content = False
        decoded = self._decode_action(action_id)
        if decoded["type"] == "question" and self.learner_state["fail_streak"] >= 3:
            # Remediation gate: convert to content (video modality) for same LO
            forced_content = True
            decoded = {"type": "content", "lo": decoded["lo"], "modality": 0}

        if decoded["type"] == "question":
            enforced_difficulty, bp_override = self._enforce_blueprint(decoded["lo"], decoded["difficulty"])
            question = self._get_question(decoded["lo"], enforced_difficulty)
            correct = self._sample_correctness(question)
            pre_mastery = self.learner_state["mastery"][decoded["lo"]]
            gain = 0.05 * (1 - pre_mastery) if correct else 0.0
            if correct:
                self.learner_state["mastery"][decoded["lo"]] += gain
                self.learner_state["fail_streak"] = 0
                self.learner_state["ability"] += 0.02
            else:
                self.learner_state["fail_streak"] += 1
            self._update_frustration(correct, enforced_difficulty, pre_mastery)
            response_time = self._sample_response_time(question)
            self.learner_state["response_time"] = response_time
            self.difficulty_counts[enforced_difficulty] += 1
            result = {
                "type": "question",
                "lo": decoded["lo"],
                "difficulty": enforced_difficulty,
                "correct": correct,
                "mastery_gain": gain,
                "frustration": self.learner_state["frustration"],
                "response_time": response_time,
                "forced_content": forced_content,
                "blueprint_override": bp_override,
            }
        else:
            content = self._get_content(decoded["lo"], decoded["modality"])
            pre_mastery = self.learner_state["mastery"][decoded["lo"]]
            gain = self._compute_content_gain(content, pre_mastery)
            self.learner_state["mastery"][decoded["lo"]] = min(1.0, pre_mastery + gain)
            frustration_delta = content["engagement_impact"]
            self.learner_state["frustration"] = np.clip(
                self.learner_state["frustration"] + frustration_delta, 0, 1
            )
            self.learner_state["fail_streak"] = 0
            result = {
                "type": "content",
                "lo": decoded["lo"],
                "modality": decoded["modality"],
                "mastery_gain": gain,
                "frustration_delta": frustration_delta,
                "frustration": self.learner_state["frustration"],
                "forced_content": forced_content,
            }

        reward = self._compute_reward(result)
        self.step_count += 1
        done, reason = self._is_terminal()
        obs = self._get_observation()
        info = {
            "termination_reason": reason,
            "mean_mastery": float(np.mean(self.learner_state["mastery"])),
            "blueprint_adherence": self._compute_blueprint_adherence(),
            # Pass through result fields for training loop tracking
            "type": result["type"],
            "mastery_gain": result.get("mastery_gain", 0.0),
            "correct": result.get("correct", False),
            "modality": result.get("modality"),
            "difficulty": result.get("difficulty"),
        }
        self.episode_log.append({
            "state": obs,
            "action": action_id,
            "reward": reward,
            "done": done,
            "info": info,
            "result": result,
            "mastery_vector": self.learner_state["mastery"].copy(),
            "action_type": result["type"],
            "difficulty": result.get("difficulty"),
            "correct": result.get("correct"),
            "mastery_gain": result.get("mastery_gain", 0.0),
            "frustration": result.get("frustration", 0.0),
        })
        return obs, reward, done, info


# -----------------------------
# PPO Agent
# -----------------------------


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        last = state_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, action_dim)
        self.value_head = nn.Linear(last, 1)

    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None):
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, log_prob, entropy, value


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float):
    advantages = torch.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def value_clip_loss(value: torch.Tensor, value_old: torch.Tensor, returns: torch.Tensor, epsilon: float) -> torch.Tensor:
    value_clipped = value_old + torch.clamp(value - value_old, -epsilon, epsilon)
    loss_unclipped = (value - returns) ** 2
    loss_clipped = (value_clipped - returns) ** 2
    return torch.max(loss_unclipped, loss_clipped).mean()


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


class PPOAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.model = ActorCritic(config["state_dim"], config["action_dim"], config["hidden_layers"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])

    def select_action(self, state: np.ndarray):
        state_t = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.model.get_action_and_value(state_t)
        return action.item(), log_prob.item(), value.item()

    def update(self, batch: RolloutBatch):
        cfg = self.config
        advantages, returns = compute_gae(
            rewards=batch.rewards,
            values=batch.values,
            dones=batch.dones,
            gamma=cfg["discount_gamma"],
            lam=cfg["gae_lambda"],
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        old_logprobs = batch.logprobs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        batch_size = states.size(0)
        for epoch in range(cfg["epochs_per_update"]):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, cfg["minibatch_size"]):
                end = start + cfg["minibatch_size"]
                mb_idx = indices[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]

                _, new_logprobs, entropy, values = self.model.get_action_and_value(mb_states, mb_actions)
                ratio = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg["clip_epsilon"], 1 + cfg["clip_epsilon"]) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                if cfg.get("use_value_clipping", False):
                    value_loss = value_clip_loss(values, mb_returns.detach(), mb_returns, cfg.get("value_clip_epsilon", 0.2))
                else:
                    value_loss = (values - mb_returns).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg["value_loss_coef"] * value_loss + cfg["entropy_coef"] * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

            # Early stop on high KL divergence
            with torch.no_grad():
                _, new_logprobs, _, _ = self.model.get_action_and_value(states, actions)
                approx_kl = (old_logprobs - new_logprobs).mean().abs().item()
            if cfg.get("target_kl") is not None and approx_kl > cfg["target_kl"]:
                break


# -----------------------------
# Training Loop
# -----------------------------


def collect_rollout(env: AdaptiveLearningEnv, agent: PPOAgent, steps: int) -> RolloutBatch:
    states: List[np.ndarray] = []
    actions: List[int] = []
    logprobs: List[float] = []
    rewards: List[float] = []
    dones: List[float] = []
    values: List[float] = []

    state = env.reset()
    for _ in range(steps):
        action, logprob, value = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        logprobs.append(logprob)
        rewards.append(reward)
        dones.append(float(done))
        values.append(value)
        state = next_state if not done else env.reset()
    return RolloutBatch(
        states=torch.tensor(np.array(states), dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.int64),
        logprobs=torch.tensor(logprobs, dtype=torch.float32),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        dones=torch.tensor(dones, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
    )


def train_single_seed(seed: int, config: Dict, num_episodes: int = None) -> Dict:
    """
    Train PPO for a single seed with comprehensive metrics tracking.
    Returns: Dict with episode_metrics, returns, time_to_mastery, duration
    """
    print(f"\n{'='*70}")
    print(f"PPO Training: Seed {seed}")
    print(f"{'='*70}")
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = AdaptiveLearningEnv()
    agent = PPOAgent(config)
    
    num_episodes = num_episodes or config["max_episodes"]
    max_steps_per_ep = config["max_steps_per_episode"]
    
    episode_rewards = []
    episode_ttm = []
    episode_metrics = []
    episode_steps: List[int] = []  # NEW: Track steps per episode for AUC/checkpoints
    
    episodes_completed = 0
    start_time = time.time()
    
    while episodes_completed < num_episodes:
        # Run episode
        ep_states, ep_actions, ep_logprobs, ep_rewards, ep_values = [], [], [], [], []
        ep_info_log = []
        
        state = env.reset(seed=seed + episodes_completed)
        ep_return = 0.0
        ep_mastery_reached_step = None
        ep_question_total = 0
        ep_question_correct = 0
        ep_content_count = 0
        ep_question_diffs = []
        ep_content_gains = []
        ep_modality_gains = {"video": [], "PPT": [], "text": [], "blog": [], "article": [], "handout": []}
        ep_frustration_history = []
        modality_names = ["video", "PPT", "text", "blog", "article", "handout"]
        
        for step in range(max_steps_per_ep):
            action, logprob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            ep_states.append(state)
            ep_actions.append(action)
            ep_logprobs.append(logprob)
            ep_rewards.append(reward)
            ep_values.append(value)
            ep_return += reward
            
            # Track metrics from info
            if info:
                ep_info_log.append(info)
                
                # Mastery tracking
                mean_mastery = np.mean(env.learner_state["mastery"])
                if ep_mastery_reached_step is None and mean_mastery >= 0.8:
                    ep_mastery_reached_step = step + 1
                
                # Frustration
                ep_frustration_history.append(env.learner_state["frustration"])
                
                # Action-specific tracking (use info dict, not re-decoding)
                action_type = info.get("type")
                if action_type == "content":
                    ep_content_count += 1
                    mastery_gain = info.get("mastery_gain", 0.0)
                    ep_content_gains.append(mastery_gain)
                    modality = info.get("modality")
                    if modality is not None and 0 <= modality < len(modality_names):
                        ep_modality_gains[modality_names[modality]].append(mastery_gain)
                elif action_type == "question":
                    ep_question_total += 1
                    if info.get("correct", False):
                        ep_question_correct += 1
                    ep_question_diffs.append(info.get("difficulty", 0))
            
            state = next_state
            if done:
                break
        
        # Update PPO with collected episode data
        batch = RolloutBatch(
            states=torch.tensor(np.array(ep_states), dtype=torch.float32),
            actions=torch.tensor(ep_actions, dtype=torch.long),
            logprobs=torch.tensor(ep_logprobs, dtype=torch.float32),
            rewards=torch.tensor(ep_rewards, dtype=torch.float32),
            dones=torch.zeros(len(ep_rewards), dtype=torch.float32),  # All within episode
            values=torch.tensor(ep_values, dtype=torch.float32),
        )
        agent.update(batch)
        
        # Compute episode metrics
        ep_metric = {
            "episode": episodes_completed + 1,
            "return": ep_return,
            "cumulative_reward": ep_return,
            "time_to_mastery": ep_mastery_reached_step if ep_mastery_reached_step else len(ep_rewards),
            "total_steps": len(ep_rewards),
            "question_accuracy": ep_question_correct / ep_question_total if ep_question_total > 0 else 0.0,
            "question_total": ep_question_total,
            "question_correct": ep_question_correct,
            "content_count": ep_content_count,
            "content_rate": ep_content_count / len(ep_rewards) if len(ep_rewards) > 0 else 0.0,
            "blueprint_adherence": _compute_blueprint_adherence(ep_question_diffs),
            "post_content_gain": float(np.mean(ep_content_gains)) if ep_content_gains else 0.0,
            "final_mastery": float(np.mean(next_state[:30])) if len(next_state) >= 30 else 0.0,
            "mean_frustration": float(np.mean(ep_frustration_history)) if ep_frustration_history else 0.0,
            "modality_gains": {
                mod: float(np.mean(gains)) if gains else 0.0
                for mod, gains in ep_modality_gains.items()
            },
        }
        
        episode_metrics.append(ep_metric)
        episode_rewards.append(ep_return)
        episode_steps.append(len(ep_rewards))  # NEW: Record episode length
        if ep_mastery_reached_step:
            episode_ttm.append(ep_mastery_reached_step)
        
        episodes_completed += 1
        
        if (episodes_completed) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episodes_completed}/{num_episodes} | Avg reward (last 10): {avg_reward:.2f}")
    
    elapsed = time.time() - start_time
    print(f"Seed {seed} completed in {elapsed:.1f}s")
    
    wall_clock_time_seconds = time.time() - start_time
    wall_clock_time_minutes = wall_clock_time_seconds / 60.0
    
    # Compute additional metrics for comparison
    auc_10k = compute_auc_at_10k(episode_rewards, episode_steps)
    checkpoints = compute_checkpoint_metrics(episode_rewards, episode_metrics, episode_steps)
    
    return {
        "seed": seed,
        "returns": episode_rewards,
        "time_to_mastery": episode_ttm,
        "episode_metrics": episode_metrics,
        "duration_s": elapsed,
        "wall_clock_time_minutes": wall_clock_time_minutes,
        "auc_10k": auc_10k,
        "checkpoints": checkpoints,
        "total_steps_per_episode": episode_steps,
    }


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


def _compute_blueprint_adherence(question_diffs: List[int]) -> float:
    """Compute blueprint adherence score."""
    if not question_diffs:
        return 100.0
    counts = np.bincount(np.array(question_diffs), minlength=3)[:3].astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 100.0
    actual = counts / total
    target = np.array([0.20, 0.60, 0.20], dtype=np.float64)
    deviation = np.abs(actual - target).mean()
    return (1.0 - deviation) * 100.0


# -----------------------------
# Multi-seed training and export
# -----------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    # Handle both file paths and directory paths
    if path.endswith(('.json', '.csv', '.txt')):
        d = os.path.dirname(path)
    else:
        d = path
    if d:
        os.makedirs(d, exist_ok=True)


def summarize_across_seeds(results: List[Dict]) -> Dict:
    """Compute mean±SD and 95% CI across seeds."""
    if not results:
        return {}
    
    # Aggregate metrics
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
    
    # Bootstrap CI
    def bootstrap_ci(values, n_boot=1000, ci=0.95):
        if len(values) < 2:
            return (float(values[0]), float(values[0])) if values else (0.0, 0.0)
        boot_means = [float(np.mean(np.random.choice(values, len(values), replace=True))) for _ in range(n_boot)]
        lower = np.percentile(boot_means, (1-ci)/2 * 100)
        upper = np.percentile(boot_means, (1+ci)/2 * 100)
        return (float(lower), float(upper))
    
    cum_rewards = [float(np.sum(r.get("returns", []))) for r in results]
    ci_cumulative = bootstrap_ci(cum_rewards)
    
    ttm_values = [float(np.mean(r.get("time_to_mastery", []))) if r.get("time_to_mastery") else 0.0 for r in results]
    ci_ttm = bootstrap_ci(ttm_values)
    
    return {
        "cumulative_reward": {"mean": mean_cumulative, "std": std_cumulative, "ci_95": ci_cumulative},
        "time_to_mastery": {"mean": mean_ttm, "std": std_ttm, "ci_95": ci_ttm},
        "question_accuracy": {"mean": mean_accuracy, "std": std_accuracy},
        "blueprint_adherence": {"mean": mean_blueprint, "std": std_blueprint},
        "post_content_gain": {"mean": mean_post_content, "std": std_post_content},
        "mean_frustration": {"mean": mean_frustration, "std": std_frustration},
        "final_mastery": {"mean": mean_final_mastery, "std": std_final_mastery},
        "num_seeds": len(results),
    }


def train_multi_seed(seeds: List[int], config: Dict, num_episodes: int, output_dir: str = "results/ppo") -> Dict:
    """Train PPO across multiple seeds and export results."""
    ensure_dir(output_dir)
    
    results = []
    for seed in seeds:
        result = train_single_seed(seed, config, num_episodes)
        results.append(result)
    
    # Aggregate statistics
    summary = summarize_across_seeds(results)
    
    # Export JSON
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
    
    # Export CSV
    csv_path = os.path.join(output_dir, "episodes.csv")
    export_episodes_csv(results, csv_path)
    
    # Export figures
    figures_dir = os.path.join(output_dir, "figures")
    export_figures(results, figures_dir)
    
    return {"summary": summary, "results": results}


def export_episodes_csv(results: List[Dict], path: str) -> None:
    """Export per-episode metrics to CSV."""
    import csv
    ensure_dir(path)
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
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
                    seed, em.get("episode", 0), em.get("return", 0.0),
                    em.get("cumulative_reward", 0.0), em.get("time_to_mastery", 0),
                    em.get("total_steps", 0), em.get("question_accuracy", 0.0),
                    em.get("content_rate", 0.0), em.get("blueprint_adherence", 0.0),
                    em.get("post_content_gain", 0.0),
                    modality_gains.get("video", 0.0), modality_gains.get("PPT", 0.0),
                    modality_gains.get("text", 0.0), modality_gains.get("blog", 0.0),
                    modality_gains.get("article", 0.0), modality_gains.get("handout", 0.0),
                    em.get("final_mastery", 0.0), em.get("mean_frustration", 0.0),
                ])
    
    print(f"✓ Episodes CSV exported to {path}")


def export_figures(results: List[Dict], figures_dir: str) -> None:
    """Export learning curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping figures")
        return
    
    ensure_dir(os.path.join(figures_dir, "placeholder"))
    
    plt.figure(figsize=(8, 5))
    for result in results:
        returns = result.get("returns", [])
        if returns:
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
    plt.title("PPO Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "learning_curve.png"), dpi=200)
    plt.close()
    
    print(f"✓ Figures exported to {figures_dir}/")


# -----------------------------
# Legacy evaluation metrics (kept for compatibility)
# -----------------------------


def train():
    """Legacy train function for backward compatibility."""
    env = AdaptiveLearningEnv()
    agent = PPOAgent(CONFIG)
    total_iterations = math.ceil(CONFIG["max_episodes"] * CONFIG["max_steps_per_episode"] / CONFIG["rollout_steps"])

    for it in range(total_iterations):
        batch = collect_rollout(env, agent, CONFIG["rollout_steps"])
        agent.update(batch)
        mean_reward = batch.rewards.mean().item()
        print(f"Iter {it+1:04d} | batch_reward={mean_reward:.3f} | steps={len(batch.rewards)}")

    torch.save(agent.model.state_dict(), "ppo_agent.pt")
    print("Training complete. Model saved to ppo_agent.pt")


# -----------------------------
# Evaluation Metrics (spec-aligned)
# -----------------------------


def compute_time_to_mastery(episode_log: List[Dict], threshold: float = 0.8):
    for idx, step in enumerate(episode_log):
        mean_mastery = float(np.mean(step["mastery_vector"]))
        if mean_mastery >= threshold:
            return idx + 1
    return None


def compute_cumulative_reward(episode_log: List[Dict]) -> float:
    return float(sum(transition["reward"] for transition in episode_log))


def compute_post_content_gain(episode_log: List[Dict]) -> float:
    gains = [t.get("mastery_gain", 0.0) for t in episode_log if t.get("action_type") == "content"]
    if not gains:
        return 0.0
    return float(np.mean(gains))


def compute_blueprint_adherence(episode_log: List[Dict]) -> float:
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    for t in episode_log:
        if t.get("action_type") == "question":
            diff = t.get("difficulty")
            label = {0: "easy", 1: "medium", 2: "hard"}.get(diff)
            if label is not None:
                difficulty_counts[label] += 1
    total = sum(difficulty_counts.values())
    if total == 0:
        return 100.0
    actual = {k: v / total for k, v in difficulty_counts.items()}
    target = {"easy": 0.20, "medium": 0.60, "hard": 0.20}
    deviation = sum(abs(actual[d] - target[d]) for d in target) / len(target)
    return (1.0 - deviation) * 100.0


def compute_question_accuracy(episode_log: List[Dict]) -> float:
    correct = sum(1 for t in episode_log if t.get("action_type") == "question" and t.get("correct"))
    total = sum(1 for t in episode_log if t.get("action_type") == "question")
    return correct / total if total > 0 else 0.0


def compute_content_rate(episode_log: List[Dict]) -> float:
    if not episode_log:
        return 0.0
    content_count = sum(1 for t in episode_log if t.get("action_type") == "content")
    return content_count / len(episode_log)


def compute_final_mastery(episode_log: List[Dict]) -> float:
    if not episode_log:
        return 0.0
    return float(np.mean(episode_log[-1]["mastery_vector"]))


def compute_mean_frustration(episode_log: List[Dict]) -> float:
    if not episode_log:
        return 0.0
    return float(np.mean([t.get("frustration", 0.0) for t in episode_log]))


def compute_policy_stability(results_across_seeds: List[Dict]) -> float:
    cumulative_rewards = [r["cumulative_reward"] for r in results_across_seeds]
    return float(np.std(cumulative_rewards)) if cumulative_rewards else 0.0


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🚀 PPO TRAINING STARTED")
    print(f"{'='*70}\n")
    
    parser = argparse.ArgumentParser(description="Train PPO for adaptive mock interviews (aligned with DQN/PETS/MBPO)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (overrides multi-seed)")
    parser.add_argument("--seeds", type=int, nargs="*", default=UNIFIED_SEEDS,
                        help=f"Multiple seeds for statistical validation (default: {UNIFIED_SEEDS})")
    parser.add_argument("--episodes", type=int, default=UNIFIED_EPISODES,
                        help=f"Number of episodes (default: {UNIFIED_EPISODES} for ~30k steps)")
    parser.add_argument("--output", type=str, default="results/ppo", help="Output directory")
    parser.add_argument("--legacy", action="store_true", help="Use legacy train() function")
    args = parser.parse_args()
    
    print(f"📋 Configuration:")
    if args.seed:
        print(f"   Seed: {args.seed}")
    else:
        print(f"   Seeds: {args.seeds}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Output: {args.output}")
    print(f"   Device: {CONFIG['device']}")
    print(f"{'='*70}\n")
    
    if args.legacy:
        # Legacy mode
        print(f"🎓 LEGACY TRAINING MODE")
        try:
            train()
            print(f"✅ Training complete!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        # Modern multi-seed mode
        if args.seed is not None:
            seeds = [args.seed]
        else:
            seeds = args.seeds
        
        if len(seeds) == 1:
            # Single seed
            print(f"📍 Single seed training (seed={seeds[0]})")
            try:
                result = train_single_seed(seeds[0], CONFIG, args.episodes)
                print(f"✅ Training complete!")
                print(f"\n{json.dumps(result['episode_metrics'][-5:], indent=2)}")
                
                # NEW: Export JSON for single seed (for smoke tests and production consistency)
                ensure_dir(args.output)
                summary_path = os.path.join(args.output, "summary.json")
                with open(summary_path, "w") as f:
                    json.dump({
                        "auc_10k": float(np.sum(result.get("returns", [])[:100])),  # Approximate AUC@10k
                        "checkpoints": {},  # Not computed for single seed smoke test
                        "wall_clock_time_minutes": result.get("duration_s", 0) / 60.0,
                        "seed": result["seed"],
                        "episodes_completed": len(result.get("returns", [])),
                        "total_return": float(np.sum(result.get("returns", []))),
                        "mean_return": float(np.mean(result.get("returns", []))) if result.get("returns") else 0.0,
                    }, f, indent=2)
                print(f"✅ Exported: {summary_path}")
            except Exception as e:
                print(f"❌ Training failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            # Multi-seed
            print(f"📍 Multi-seed training ({len(seeds)} seeds)")
            try:
                output = train_multi_seed(seeds, CONFIG, args.episodes, args.output)
                print(f"\n{'='*70}")
                print(f"✅ Multi-seed training complete!")
                print(f"Summary: {json.dumps(output['summary'], indent=2)}")
                print(f"{'='*70}")
            except Exception as e:
                print(f"❌ Multi-seed training failed: {e}")
                import traceback
                traceback.print_exc()
                raise

