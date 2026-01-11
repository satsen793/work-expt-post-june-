"""
DQN (with prioritized replay) training script for the adaptive mock-interview simulator.
Implements the environment described in the spec_* files and a baseline DQN agent.
"""
from __future__ import annotations

import math
import random
import sys
import os
import time  # NEW: For wall-clock timing
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import unified configuration for 1:1 replication with PETS_VER3
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared_config import UNIFIED_SEEDS, UNIFIED_EPISODES, UNIFIED_MAX_STEPS_PER_EPISODE, DEFAULT_WARMUP_STEPS

# Spec-aligned simulator configuration
SIMULATOR_CONFIG = {
    "num_los": 30,
    "num_questions_per_lo": 20,
    "num_contents_per_lo": 6,
    "max_episode_steps": 140,
    "min_episode_steps": 80,
    "irt": {
        "difficulty_ranges": {
            "easy": (-2.0, -0.5),
            "medium": (-0.5, 0.5),
            "hard": (0.5, 2.0),
        },
        "discrimination_ranges": {
            "easy": (0.5, 1.0),
            "medium": (1.0, 1.5),
            "hard": (1.5, 2.0),
        },
        "guessing_range": (0.1, 0.25),
    },
    "content": {
        "effectiveness_by_modality": {
            "video": (0.10, 0.15),
            "PPT": (0.08, 0.12),
            "text": (0.05, 0.08),
            "blog": (0.07, 0.10),
            "article": (0.06, 0.09),
            "handout": (0.05, 0.08),
        }
    },
    "reward_weights": {
        "correctness": 1.0,
        "mastery_gain": 0.5,
        "frustration_penalty": 0.3,
        "post_content_gain": 2.0,
        "engagement_bonus": 0.5,
    },
    "termination": {
        "mastery_threshold": 0.8,
        "max_frustration": 0.95,
    },
}

# --------- Environment: Adaptive Learning Simulator ---------


@dataclass
class Question:
    lo: int
    difficulty: int  # 0=Easy, 1=Medium, 2=Hard
    a: float  # discrimination
    b: float  # difficulty
    c: float  # guessing
    rt_mean: float
    rt_std: float


@dataclass
class Content:
    lo: int
    modality: int  # 0=video, 1=PPT, 2=text, 3=blog, 4=article, 5=handout
    effectiveness: float
    engagement_impact: float
    duration_min: float
    duration_max: float
    reading_complexity: float


class DiscreteActionSpace:
    """Simple wrapper for action_space.sample() compatibility with gym-based algorithms."""
    def __init__(self, n: int):
        self.n = n
    
    def sample(self):
        """Sample random action."""
        return np.random.randint(0, self.n)


class AdaptiveLearningEnv:
    def __init__(self, seed: int = 0, max_steps: int | None = None, config: Dict | None = None):
        self.cfg = config or SIMULATOR_CONFIG
        self.num_los = self.cfg["num_los"]
        self.num_questions = self.cfg["num_los"] * self.cfg["num_questions_per_lo"]
        self.num_contents = self.cfg["num_los"] * self.cfg["num_contents_per_lo"]
        self.max_steps_cap = max_steps or self.cfg["max_episode_steps"]  # upper bound; per-episode sampled 80-140
        self.min_steps_cap = self.cfg.get("min_episode_steps", 80)
        self.action_space_n = 270
        self.action_space = DiscreteActionSpace(270)  # NEW: For gym-compatible algorithms (MBPO)
        self.rng = np.random.default_rng(seed)

        # Blueprint difficulty targets: Easy 20%, Medium 60%, Hard 20%
        self.blueprint_target = np.array([0.20, 0.60, 0.20], dtype=np.float32)
        self.blueprint_penalty_weight = 0.0  # strict masking replaces penalty

        self.questions = self._build_questions()
        self.contents = self._build_contents()

        self.state = None
        self.step_count = 0
        self.question_counts = np.zeros(3, dtype=np.int32)
        self.episode_log: List[Dict] = []  # step-level transitions (answered/content_completed)
        self.event_log: List[Dict] = []    # all events including shown/completed
        self.max_steps_current = max_steps

    # --- Initialization helpers ---
    def _build_questions(self) -> Dict[Tuple[int, int], List[Question]]:
        questions: Dict[Tuple[int, int], List[Question]] = {}
        # 20 questions per LO → 600 total, apportioned 20/60/20 by difficulty
        per_lo_counts = {0: 4, 1: 12, 2: 4}
        ranges_b = self.cfg["irt"]["difficulty_ranges"]
        ranges_a = self.cfg["irt"]["discrimination_ranges"]
        g_low, g_high = self.cfg["irt"]["guessing_range"]
        for lo in range(self.num_los):
            for diff in range(3):
                bucket = []
                for _ in range(per_lo_counts[diff]):
                    if diff == 0:  # Easy
                        b = self.rng.uniform(*ranges_b["easy"])
                        a = self.rng.uniform(*ranges_a["easy"])
                    elif diff == 1:  # Medium
                        b = self.rng.uniform(*ranges_b["medium"])
                        a = self.rng.uniform(*ranges_a["medium"])
                    else:  # Hard
                        b = self.rng.uniform(*ranges_b["hard"])
                        a = self.rng.uniform(*ranges_a["hard"])
                    c = self.rng.uniform(g_low, g_high)
                    bucket.append(
                        Question(
                            lo=lo,
                            difficulty=diff,
                            a=a,
                            b=b,
                            c=c,
                            rt_mean=30.0,
                            rt_std=10.0,
                        )
                    )
                questions[(lo, diff)] = bucket
        return questions

    def _build_contents(self) -> Dict[Tuple[int, int], Content]:
        contents: Dict[Tuple[int, int], Content] = {}
        # Honor the provided config rather than the global default
        m = self.cfg["content"]["effectiveness_by_modality"]
        # duration ranges are modality-specific per spec_simulator.md
        modality_specs = {
            0: (m["video"][0], m["video"][1], -0.08, 15.0, 25.0),
            1: (m["PPT"][0], m["PPT"][1], -0.05, 10.0, 20.0),
            2: (m["text"][0], m["text"][1], 0.02, 5.0, 10.0),
            3: (m["blog"][0], m["blog"][1], -0.03, 8.0, 15.0),
            4: (m["article"][0], m["article"][1], 0.00, 10.0, 18.0),
            5: (m["handout"][0], m["handout"][1], 0.05, 5.0, 12.0),
        }
        for lo in range(self.num_los):
            for modality, (eff_min, eff_max, engage, dur_min, dur_max) in modality_specs.items():
                contents[(lo, modality)] = Content(
                    lo=lo,
                    modality=modality,
                    effectiveness=self.rng.uniform(eff_min, eff_max),
                    engagement_impact=engage,
                    duration_min=dur_min,
                    duration_max=dur_max,
                    reading_complexity=float(self.rng.uniform(0.2, 0.8)),
                )
        return contents

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.question_counts = np.zeros(3, dtype=np.int32)
        # Sample per-episode budget in [80, max_steps_cap] to mirror 80-140 spec
        self.max_steps_current = int(self.rng.integers(self.min_steps_cap, self.max_steps_cap + 1))
        mastery = self.rng.beta(a=2.0, b=5.0, size=self.num_los)
        self.state = {
            "mastery": mastery,
            "ability": self.rng.normal(0.0, 1.0),
            "frustration": 0.0,
            "response_time": 0.0,
            "fail_streak": 0,
            "engagement": 1.0,
        }
        self.episode_log = []
        self.event_log = []
        return self._get_obs()

    # --- Encoding helpers ---
    def _get_obs(self) -> np.ndarray:
        mastery = self.state["mastery"]
        frustration = np.array([np.clip(self.state["frustration"], 0.0, 1.0)])
        rt = np.array([np.clip(self.state["response_time"], 0.0, 1.0)])
        return np.concatenate([mastery, frustration, rt]).astype(np.float32)

    def _decode_action(self, action_id: int) -> Dict:
        if action_id < 90:
            lo = action_id // 3
            diff = action_id % 3
            return {"type": "question", "lo": lo, "difficulty": diff}
        else:
            cid = action_id - 90
            lo = cid // 6
            modality = cid % 6
            return {"type": "content", "lo": lo, "modality": modality}

    # --- Transition dynamics ---
    def step(self, action_id: int):
        action = self._decode_action(action_id)

        # Fail-streak gate: redirect to content-video if streak >= 3 and a question was chosen.
        if action["type"] == "question" and self.state["fail_streak"] >= 3:
            action = {"type": "content", "lo": action["lo"], "modality": 0}

        if action["type"] == "question":
            result = self._execute_question(action)
        else:
            result = self._execute_content(action)

        reward = self._compute_reward(result)
        self.step_count += 1
        done, reason = self._is_terminal()
        obs = self._get_obs()
        info = {
            "result": result,
            "termination_reason": reason,
            "step": self.step_count,
            "mean_mastery": float(np.mean(self.state["mastery"])),
        }

        # Event log: shown
        self.event_log.append(
            {
                "event_type": "question_shown" if result["type"] == "question" else "content_shown",
                "step": self.step_count,
                "action": action_id,
                "info": info,
            }
        )

        # Step-level transition (answered/content_completed) with reward
        step_entry = {
            "obs": obs,
            "action": action_id,
            "reward": reward,
            "done": done,
            "info": info,
            "action_type": result["type"],
            "event_type": "answered" if result["type"] == "question" else "content_completed",
            "difficulty": result.get("difficulty"),
            "modality": result.get("modality"),
            "correct": result.get("correct"),
            "mastery_gain": result.get("mastery_gain", 0.0),
            "frustration": result.get("frustration", 0.0),
            "response_time": result.get("response_time", 0.0),
            "mastery_vector": self.state["mastery"].copy(),
            "post_content_gain": result.get("mastery_gain", 0.0) if result["type"] == "content" else 0.0,
        }
        self.episode_log.append(step_entry)
        self.event_log.append(
            {
                "event_type": step_entry["event_type"],
                "step": self.step_count,
                "action": action_id,
                "info": info,
            }
        )
        return obs, reward, done, info

    def sample_action(self) -> int:
        """Blueprint-aware action sampler used for warmup/random data collection."""
        question_prob = 90 / 270.0
        if self.rng.random() < question_prob:
            # Strict blueprint: prefer difficulties with remaining quota vs target
            mask = self._difficulty_mask()
            deficits = self._difficulty_deficit()
            if mask.sum() == 0:
                # If all over target, pick smallest surplus
                surplus = self._difficulty_surplus()
                diff = int(self.rng.choice(np.flatnonzero(surplus == surplus.min())))
            else:
                weights = deficits * mask
                if weights.sum() == 0:
                    weights = mask / mask.sum()
                else:
                    weights = weights / weights.sum()
                diff = int(self.rng.choice([0, 1, 2], p=weights))
            lo = int(self.rng.integers(self.num_los))
            return lo * 3 + diff
        # Content action
        content_id = int(self.rng.integers(180))
        return 90 + content_id

    def _execute_question(self, action: Dict) -> Dict:
        lo = action["lo"]
        diff = action["difficulty"]
        question = self.rng.choice(self.questions[(lo, diff)])

        theta = self.state["ability"]
        prob_correct = question.c + (1 - question.c) / (1 + math.exp(-question.a * (theta - question.b)))
        correct = self.rng.random() < prob_correct

        current_mastery = self.state["mastery"][lo]
        mastery_gain = 0.0
        if correct:
            mastery_gain = 0.05 * (1 - current_mastery)
            self.state["mastery"][lo] = np.clip(current_mastery + mastery_gain, 0.0, 1.0)
            self.state["fail_streak"] = 0
            self.state["ability"] += 0.02
        else:
            self.state["fail_streak"] += 1

        if correct:
            self.state["frustration"] = max(0.0, self.state["frustration"] - 0.05)
        else:
            delta = 0.10
            if diff == 2 and current_mastery < 0.5:
                delta += 0.05
            self.state["frustration"] = min(1.0, self.state["frustration"] + delta)

        # Simple engagement update tied to frustration (placeholder per spec engagement tracking)
        self.state["engagement"] = max(0.0, 1.0 - self.state["frustration"])

        # Track blueprint adherence
        self.question_counts[diff] += 1

        base_time = question.rt_mean
        rt = max(5.0, self.rng.normal(base_time, question.rt_std))
        self.state["response_time"] = min(rt / 120.0, 1.0)

        return {
            "type": "question",
            "lo": lo,
            "difficulty": diff,
            "correct": correct,
            "mastery_gain": mastery_gain,
            "frustration": self.state["frustration"],
            "response_time": rt,
        }

    def _execute_content(self, action: Dict) -> Dict:
        lo = action["lo"]
        modality = action["modality"]
        content = self.contents[(lo, modality)]

        pre_mastery = self.state["mastery"][lo]
        effective_gain = content.effectiveness * (1 - pre_mastery)
        frustration_penalty = self.state["frustration"] * 0.5
        effective_gain *= max(0.0, 1 - frustration_penalty)
        noise = self.rng.normal(0.0, 0.02)
        final_gain = max(0.0, effective_gain + noise)

        self.state["mastery"][lo] = np.clip(pre_mastery + final_gain, 0.0, 1.0)
        self.state["frustration"] = np.clip(self.state["frustration"] + content.engagement_impact, 0.0, 1.0)
        self.state["fail_streak"] = 0
        self.state["response_time"] = 0.0
        self.state["engagement"] = max(0.0, 1.0 - self.state["frustration"])

        return {
            "type": "content",
            "lo": lo,
            "modality": modality,
            "mastery_gain": final_gain,
            "frustration_delta": content.engagement_impact,
            "frustration": self.state["frustration"],
        }

    # --- Reward and termination ---
    def _compute_reward(self, result: Dict) -> float:
        rw = self.cfg["reward_weights"]
        reward = 0.0
        if result["type"] == "question":
            if result["correct"]:
                reward += rw["correctness"]
            reward += rw["mastery_gain"] * result["mastery_gain"]
            reward -= rw["frustration_penalty"] * result["frustration"]
        else:
            reward += rw["post_content_gain"] * result["mastery_gain"]
            reward += rw["engagement_bonus"] * (-result["frustration_delta"])
        return reward

    # --- Episode metrics (spec_evaluation) ---
    def get_episode_metrics(self) -> Dict[str, float]:
        if not self.episode_log:
            return {}
        step_events = [t for t in self.episode_log if t.get("event_type") in {"answered", "content_completed"}]
        cumulative_reward = sum(t["reward"] for t in step_events)
        mean_frustration = float(np.mean([t["frustration"] for t in step_events])) if step_events else 0.0
        question_accuracy = self._compute_question_accuracy()
        blueprint = self._compute_blueprint_adherence()
        post_content_gain = self._compute_post_content_gain()
        post_content_gain_by_modality = compute_post_content_gain_by_modality(self.episode_log)

        return {
            "total_steps": self.step_count,
            "final_mastery": float(np.mean(self.state["mastery"])),
            "cumulative_reward": cumulative_reward,
            "question_accuracy": question_accuracy,
            "content_rate": self._compute_content_rate(),
            "blueprint_adherence": blueprint,
            "post_content_gain": post_content_gain,
            "post_content_gain_by_modality": post_content_gain_by_modality,
            "mean_frustration": mean_frustration,
        }

    def _compute_question_accuracy(self) -> float:
        question_transitions = [t for t in self.episode_log if t.get("event_type") == "answered"]
        if not question_transitions:
            return 0.0
        correct = sum(1 for t in question_transitions if t.get("correct"))
        return correct / len(question_transitions)

    def _compute_content_rate(self) -> float:
        content_events = [t for t in self.episode_log if t.get("event_type") == "content_completed"]
        step_events = [t for t in self.episode_log if t.get("event_type") in {"answered", "content_completed"}]
        if not step_events:
            return 0.0
        return float(len(content_events) / len(step_events))

    def _compute_post_content_gain(self) -> float:
        gains = [t.get("mastery_gain", 0.0) for t in self.episode_log if t.get("event_type") == "content_completed"]
        if not gains:
            return 0.0
        return float(np.mean(gains))

    def _compute_blueprint_adherence(self) -> float:
        question_transitions = [t for t in self.episode_log if t.get("event_type") == "answered"]
        if not question_transitions:
            return 100.0
        counts = {0: 0, 1: 0, 2: 0}
        for t in question_transitions:
            diff = t.get("difficulty")
            if diff is not None:
                counts[diff] += 1
        total = sum(counts.values())
        actual = np.array([counts[0], counts[1], counts[2]], dtype=np.float32) / total
        deviation = np.abs(actual - self.blueprint_target).mean()
        adherence = 1.0 - deviation
        return float(adherence * 100.0)

    def _difficulty_mask(self) -> np.ndarray:
        total_q = max(1, int(self.question_counts.sum()))
        actual = self.question_counts / total_q
        mask = (actual < self.blueprint_target).astype(np.float32)
        # If all exceed targets, allow the minimum surplus difficulty
        if mask.sum() == 0:
            surplus = self._difficulty_surplus()
            mask = (surplus == surplus.min()).astype(np.float32)
        return mask

    def _difficulty_deficit(self) -> np.ndarray:
        total_q = max(1, int(self.question_counts.sum()))
        actual = self.question_counts / total_q
        return np.clip(self.blueprint_target - actual, 0.0, None)

    def _difficulty_surplus(self) -> np.ndarray:
        total_q = max(1, int(self.question_counts.sum()))
        actual = self.question_counts / total_q
        return np.clip(actual - self.blueprint_target, 0.0, None)

    def _is_terminal(self) -> Tuple[bool, str | None]:
        mean_mastery = float(np.mean(self.state["mastery"]))
        if mean_mastery >= self.cfg["termination"]["mastery_threshold"]:
            return True, "mastery_achieved"
        if self.step_count >= self.max_steps_current:
            return True, "step_limit"
        if self.state["frustration"] >= self.cfg["termination"]["max_frustration"]:
            return True, "critical_frustration"
        return False, None


# --------- DQN with Prioritized Experience Replay ---------


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 200_000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        batch = list(zip(*samples))
        states = torch.tensor(np.stack(batch[0]), dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(np.stack(batch[3]), dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 200_000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.total_steps = 0
        self.action_dim = action_dim

    def select_action(
        self,
        state: np.ndarray,
        blueprint_counts: np.ndarray | None = None,
        blueprint_target: np.ndarray | None = None,
    ) -> int | None:
        self.total_steps += 1
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.total_steps / self.eps_decay)
        if random.random() < epsilon:
            return None  # Caller can supply env.sample_action to enforce blueprint during exploration

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q(state_t).squeeze(0)

            if blueprint_counts is not None and blueprint_target is not None:
                total_q = max(1, int(blueprint_counts.sum()))
                actual = blueprint_counts / total_q
                mask = (actual < blueprint_target)
                valid_actions = torch.ones_like(q_vals, dtype=torch.bool)
                if mask.any():
                    for action_id in range(90):
                        diff = action_id % 3
                        if not mask[diff]:
                            valid_actions[action_id] = False
                else:
                    surplus = np.clip(actual - blueprint_target, 0.0, None)
                    min_surplus = surplus.min()
                    valid_diffs = np.flatnonzero(surplus == min_surplus)
                    valid_actions = torch.zeros_like(q_vals, dtype=torch.bool)
                    for action_id in range(90):
                        if action_id % 3 in valid_diffs:
                            valid_actions[action_id] = True

                q_vals = q_vals.masked_fill(~valid_actions, -1e9)

            return int(torch.argmax(q_vals).item())

    def train_step(self, replay: PrioritizedReplay, batch_size: int):
        if len(replay) < batch_size:
            return None
        states, actions, rewards, next_states, dones, weights, indices = replay.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        with torch.no_grad():
            next_q = self.q(next_states)
            next_actions = torch.argmax(next_q, dim=1)
            next_q_target = self.q_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        current_q = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_error = target_q - current_q
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.optimizer.step()

        new_prios = td_error.detach().abs().cpu().numpy() + 1e-5
        replay.update_priorities(indices, new_prios)

        # Soft update
        with torch.no_grad():
            for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

        return float(loss.item()), float(td_error.abs().mean().item())


# --------- Training Loop ---------


def run_training(
    num_episodes: int = UNIFIED_EPISODES,  # 295 episodes for ~30k total steps
    buffer_size: int = 200_000,
    batch_size: int = 128,
    start_steps: int = DEFAULT_WARMUP_STEPS,  # 5000 steps of random exploration
    max_steps_per_episode: int = UNIFIED_MAX_STEPS_PER_EPISODE,  # 140 steps max (variable [80,140] per episode)
    seed: int = 42,
    total_steps_budget: int | None = None,
):
    start_time = time.time()  # NEW: Track wall-clock time
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = AdaptiveLearningEnv(seed=seed, max_steps=max_steps_per_episode)
    state_dim = 32
    action_dim = env.action_space_n

    agent = DQNAgent(state_dim, action_dim)
    replay = PrioritizedReplay(capacity=buffer_size)

    total_env_steps = 0
    episode_returns: List[float] = []
    episode_ttm: List[int] = []
    episode_metrics: List[Dict] = []
    episode_logs: List[List[Dict]] = []
    total_steps_per_episode: List[int] = []  # NEW: Track steps per episode for AUC/checkpoints

    for ep in range(1, num_episodes + 1):
        state = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ttm = None
        ep_steps = 0  # NEW: Count steps in current episode

        while not done:
            if total_steps_budget is not None and total_env_steps >= total_steps_budget:
                # Force stop current episode due to global step budget
                break
            if total_env_steps < start_steps:
                action = env.sample_action()
            else:
                act = agent.select_action(state, env.question_counts, env.blueprint_target)
                action = env.sample_action() if act is None else act

            next_state, reward, done, info = env.step(action)
            ep_return += reward
            if ttm is None and info["mean_mastery"] >= 0.8:
                ttm = info["step"]

            replay.push((state, action, reward, next_state, float(done)))
            state = next_state
            total_env_steps += 1
            ep_steps += 1  # NEW: Track episode steps

            if total_env_steps >= start_steps:
                agent.train_step(replay, batch_size)

        episode_returns.append(ep_return)
        episode_ttm.append(ttm if ttm is not None else max_steps_per_episode)
        episode_metrics.append(env.get_episode_metrics())
        # Keep a copy of the per-step episode log for downstream analysis/plots
        episode_logs.append(env.episode_log.copy())
        total_steps_per_episode.append(ep_steps)  # NEW: Record episode length

        if ep % 10 == 0:
            mean_return = np.mean(episode_returns[-10:])
            mean_ttm = np.mean(episode_ttm[-10:])
            print(f"Episode {ep:04d} | return={mean_return:.2f} | ttm={mean_ttm:.1f} | steps={total_env_steps}")

        # Stop training entirely if we've reached the total step budget
        if total_steps_budget is not None and total_env_steps >= total_steps_budget:
            break

    # NEW: Compute AUC@10k and checkpoint metrics
    auc_10k = compute_auc_at_10k(episode_returns, total_steps_per_episode)
    checkpoints = compute_checkpoint_metrics(episode_returns, episode_ttm, episode_metrics, total_steps_per_episode)
    
    wall_clock_time_seconds = time.time() - start_time  # NEW: Total training time
    wall_clock_time_minutes = wall_clock_time_seconds / 60.0

    print("Training finished.")
    return {
        "returns": episode_returns,
        "time_to_mastery": episode_ttm,
        "episode_metrics": episode_metrics,
        "episode_logs": episode_logs,
        "auc_10k": auc_10k,  # NEW: For Table 4
        "checkpoints": checkpoints,  # NEW: For Table 5
        "total_steps_per_episode": total_steps_per_episode,  # NEW: For downstream analysis
        "wall_clock_time_minutes": wall_clock_time_minutes,  # NEW: For Table 4 compute cost
    }


def run_multi_seed(
    seeds: List[int],
    num_episodes: int = 200,
    buffer_size: int = 200_000,
    batch_size: int = 128,
    start_steps: int = 5_000,
    max_steps_per_episode: int = 140,
):
    results = []
    for seed in seeds:
        out = run_training(
            num_episodes=num_episodes,
            buffer_size=buffer_size,
            batch_size=batch_size,
            start_steps=start_steps,
            max_steps_per_episode=max_steps_per_episode,
            seed=seed,
        )
        results.append(out)
    return results


def run_multi_seed_with_summary(
    seeds: List[int],
    num_episodes: int = 200,
    buffer_size: int = 200_000,
    batch_size: int = 128,
    start_steps: int = 5_000,
    max_steps_per_episode: int = 140,
):
    """Runs multi-seed DQN and returns both raw results and spec-aligned summary (mean/SD/CI, median/IQR for TTM)."""
    results = run_multi_seed(
        seeds=seeds,
        num_episodes=num_episodes,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_steps=start_steps,
        max_steps_per_episode=max_steps_per_episode,
    )
    summary = summarize_across_seeds(results)
    return {"results": results, "summary": summary}


def evaluate_policy(env: AdaptiveLearningEnv, agent: DQNAgent, episodes: int = 10, seed: int = 0):
    """Greedy evaluation with strict blueprint masking; returns per-episode logs and metrics."""
    eval_logs = []
    metrics = []
    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        done = False
        episode_log: List[Dict] = []
        while not done:
            action = agent.select_action(state, env.question_counts, env.blueprint_target)
            if action is None:
                action = env.sample_action()
            next_state, reward, done, info = env.step(action)
            # Use the step-level entry appended in env.episode_log (last element)
            if env.episode_log:
                episode_log.append(env.episode_log[-1])
            state = next_state
        eval_logs.append(episode_log)
        metrics.append(aggregate_episode_metrics(episode_log))
    return eval_logs, metrics


# --------- Dataset generation and validation (spec-aligned) ---------


def generate_dataset(num_learners: int = 200, episodes_per_learner: int = 3, seed: int = 0):
    env = AdaptiveLearningEnv(seed=seed)
    dataset = []

    for learner_id in range(num_learners):
        for episode in range(episodes_per_learner):
            obs = env.reset(seed=learner_id * episodes_per_learner + episode)
            done = False
            while not done:
                action = env.sample_action()
                next_obs, reward, done, info = env.step(int(action))
                dataset.append(
                    {
                        "learner_id": learner_id,
                        "episode": episode,
                        "obs": obs,
                        "action": int(action),
                        "reward": reward,
                        "next_obs": next_obs,
                        "done": done,
                        "info": info,
                        "event_log": env.event_log.copy(),
                    }
                )
                obs = next_obs
    return dataset


def validate_simulator():
    env = AdaptiveLearningEnv()

    obs = env.reset()
    assert obs.shape == (32,), "State shape mismatch"
    assert np.all(obs >= 0) and np.all(obs <= 1), "State out of bounds"

    for action in range(env.action_space_n):
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float), "Reward not float"
        assert isinstance(done, bool), "Done not bool"
        if done:
            env.reset()

    print("✓ Simulator validation checks passed")


# --------- Metric helpers (spec_evaluation) ---------


def compute_time_to_mastery(episode_log: List[Dict], threshold: float = 0.8) -> int | None:
    filtered = [e for e in episode_log if e.get("event_type") in {"answered", "content_completed"}]
    for idx, entry in enumerate(filtered):
        mastery_vec = entry.get("mastery_vector")
        if mastery_vec is None:
            mastery_vec = entry["obs"][:30]
        mean_mastery = float(np.mean(mastery_vec))
        if mean_mastery >= threshold:
            return idx + 1
    return None


def compute_cumulative_reward(episode_log: List[Dict]) -> float:
    step_events = [t for t in episode_log if t.get("event_type") in {"answered", "content_completed"}]
    return float(sum(t["reward"] for t in step_events))


def compute_post_content_gain(episode_log: List[Dict]) -> float:
    gains = [t.get("mastery_gain", 0.0) for t in episode_log if t.get("event_type") == "content_completed"]
    if not gains:
        return 0.0
    return float(np.mean(gains))


def compute_post_content_gain_by_modality(episode_log: List[Dict]) -> Dict[str, float]:
    modality_map = {0: "video", 1: "PPT", 2: "text", 3: "blog", 4: "article", 5: "handout"}
    buckets: Dict[str, List[float]] = {name: [] for name in modality_map.values()}
    for t in episode_log:
        if t.get("event_type") == "content_completed":
            modality = t.get("modality")
            name = modality_map.get(modality)
            if name is not None:
                buckets[name].append(t.get("mastery_gain", 0.0))
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in buckets.items()}


def compute_final_mastery(episode_log: List[Dict]) -> float:
    step_events = [t for t in episode_log if t.get("event_type") in {"answered", "content_completed"}]
    if not step_events:
        return 0.0
    return float(np.mean(step_events[-1]["mastery_vector"]))


def compute_mean_frustration(episode_log: List[Dict]) -> float:
    step_events = [t for t in episode_log if t.get("event_type") in {"answered", "content_completed"}]
    if not step_events:
        return 0.0
    return float(np.mean([t.get("frustration", 0.0) for t in step_events]))


def compute_blueprint_adherence(episode_log: List[Dict]) -> float:
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    for t in episode_log:
        if t.get("event_type") == "answered":
            diff = t.get("difficulty")
            if diff == 0:
                difficulty_counts["easy"] += 1
            elif diff == 1:
                difficulty_counts["medium"] += 1
            elif diff == 2:
                difficulty_counts["hard"] += 1
    total = sum(difficulty_counts.values())
    if total == 0:
        return 100.0
    actual = {
        "easy": difficulty_counts["easy"] / total,
        "medium": difficulty_counts["medium"] / total,
        "hard": difficulty_counts["hard"] / total,
    }
    target = {"easy": 0.20, "medium": 0.60, "hard": 0.20}
    deviation = sum(abs(actual[d] - target[d]) for d in target) / len(target)
    adherence = 1.0 - deviation
    return adherence * 100.0


def compute_policy_stability(results_across_seeds: List[Dict]) -> float:
    # Accept either precomputed cumulative_reward per seed or sum of returns
    cumulative_rewards = []
    for r in results_across_seeds:
        if "cumulative_reward" in r:
            cumulative_rewards.append(r["cumulative_reward"])
        elif "returns" in r:
            cumulative_rewards.append(float(np.sum(r["returns"])))
    return float(np.std(cumulative_rewards)) if cumulative_rewards else 0.0


def policy_stability_summary(results_across_seeds: List[Dict]) -> Dict[str, float]:
    # Returns SD, CV, and 95% bootstrap CI for cumulative reward across seeds (spec_evaluation)
    cumulative_rewards = []
    for r in results_across_seeds:
        if "cumulative_reward" in r:
            cumulative_rewards.append(r["cumulative_reward"])
        elif "returns" in r:
            cumulative_rewards.append(float(np.sum(r["returns"])))
    if not cumulative_rewards:
        return {"sd": 0.0, "cv": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    sd = float(np.std(cumulative_rewards))
    mean = float(np.mean(cumulative_rewards))
    cv = float(sd / mean) if mean != 0 else 0.0
    ci_lower, ci_upper = bootstrap_ci(cumulative_rewards, np.mean)
    return {"sd": sd, "cv": cv, "ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}


def median_iqr(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"median": 0.0, "p25": 0.0, "p75": 0.0}
    p25, median, p75 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "p25": float(p25), "p75": float(p75)}


def compute_question_accuracy_for_log(episode_log: List[Dict]) -> float:
    answered = [t for t in episode_log if t.get("event_type") == "answered"]
    if not answered:
        return 0.0
    correct = sum(1 for t in answered if t.get("correct"))
    return correct / len(answered)


def compute_content_rate_for_log(episode_log: List[Dict]) -> float:
    step_events = [t for t in episode_log if t.get("event_type") in {"answered", "content_completed"}]
    if not step_events:
        return 0.0
    content_events = [t for t in step_events if t.get("event_type") == "content_completed"]
    return len(content_events) / len(step_events)


def compute_auc_at_10k(episode_returns: List[float], total_steps_per_episode: List[int]) -> float:
    """
    Compute area under the cumulative reward curve up to the first 10,000 environment steps.
    Required by Table 4 for sample-efficiency comparison across algorithms.
    """
    cumulative_steps = 0
    cumulative_reward = 0.0
    for ret, steps in zip(episode_returns, total_steps_per_episode):
        if cumulative_steps >= 10_000:
            break
        cumulative_steps += steps
        cumulative_reward += ret
    return cumulative_reward


def compute_checkpoint_metrics(
    episode_returns: List[float],
    episode_ttm: List[int],
    episode_metrics: List[Dict],
    total_steps_per_episode: List[int],
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
    
    for ret, ttm, em, steps in zip(episode_returns, episode_ttm, episode_metrics, total_steps_per_episode):
        cumulative_steps += steps
        cumulative_reward += ret
        ttm_buffer.append(ttm)
        blueprint_buffer.append(em.get("blueprint_adherence", 0.0))
        
        # Check if we've crossed any checkpoints
        for checkpoint in checkpoints:
            if checkpoint not in results and cumulative_steps >= checkpoint:
                results[checkpoint] = {
                    "cumulative_reward": cumulative_reward,
                    "mean_ttm": float(np.mean(ttm_buffer)),
                    "blueprint_adherence": float(np.mean(blueprint_buffer))
                }
    
    return results


def bootstrap_ci(data: List[float], statistic_fn, confidence: float = 0.95, n_bootstrap: int = 1000):
    stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stats.append(statistic_fn(sample))
    lower = np.percentile(stats, (1 - confidence) / 2 * 100)
    upper = np.percentile(stats, (1 + confidence) / 2 * 100)
    return lower, upper


def compare_algorithms(values1: List[float], values2: List[float]):
    try:
        from scipy.stats import ttest_rel, shapiro, wilcoxon
    except ImportError:
        return {"error": "scipy not installed"}

    _, p1 = shapiro(values1)
    _, p2 = shapiro(values2)
    if p1 > 0.05 and p2 > 0.05:
        t_stat, p_value = ttest_rel(values1, values2)
        test_used = "paired_t_test"
    else:
        t_stat, p_value = wilcoxon(values1, values2)
        test_used = "wilcoxon"

    differences = np.array(values1) - np.array(values2)
    d = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0.0

    return {
        "test": test_used,
        "statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(d),
        "significant": bool(p_value < 0.05),
    }


def aggregate_episode_metrics(episode_log: List[Dict]) -> Dict[str, float]:
    modality_gains = compute_post_content_gain_by_modality(episode_log)
    return {
        "time_to_mastery": compute_time_to_mastery(episode_log) or 0,
        "cumulative_reward": compute_cumulative_reward(episode_log),
        "post_content_gain": compute_post_content_gain(episode_log),
        "post_content_gain_by_modality": modality_gains,
        "modality_gains": modality_gains,  # NEW: Alias for compatibility with template
        "blueprint_adherence": compute_blueprint_adherence(episode_log),
        "question_accuracy": compute_question_accuracy_for_log(episode_log),
        "content_rate": compute_content_rate_for_log(episode_log),
        "final_mastery": compute_final_mastery(episode_log),
        "mean_frustration": compute_mean_frustration(episode_log),
    }


def summarize_across_seeds(results_across_seeds: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Spec-aligned summary: mean, SD, and 95% bootstrap CI for primary metrics across seeds."""
    def summarize(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "sd": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        mean = float(np.mean(values))
        sd = float(np.std(values))
        ci_lower, ci_upper = bootstrap_ci(values, np.mean)
        return {"mean": mean, "sd": sd, "ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}

    cumulative_rewards = []
    ttms = []
    blueprint = []
    post_content = []
    for r in results_across_seeds:
        cumulative_rewards.append(float(np.sum(r.get("returns", []))))
        ttm_list = r.get("time_to_mastery", [])
        ttms.append(float(np.mean(ttm_list)) if ttm_list else 0.0)
        em = r.get("episode_metrics", [])
        if em:
            blueprint.append(float(np.mean([e.get("blueprint_adherence", 0.0) for e in em])))
            post_content.append(float(np.mean([e.get("post_content_gain", 0.0) for e in em])))

    stability = policy_stability_summary(results_across_seeds)
    return {
        "cumulative_reward": summarize(cumulative_rewards),
        "time_to_mastery": {**summarize(ttms), **median_iqr(ttms)},
        "blueprint_adherence": summarize(blueprint),
        "post_content_gain": summarize(post_content),
        "policy_stability": stability,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN for the adaptive learning simulator (aligned with PETS_VER3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"Max steps per episode (default {UNIFIED_MAX_STEPS_PER_EPISODE} from shared config)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=f"Number of training episodes (default {UNIFIED_EPISODES} from shared config for ~30k total steps)",
    )
    parser.add_argument(
        "--start-steps",
        type=int,
        default=None,
        help=f"Warmup steps before learning starts (default {DEFAULT_WARMUP_STEPS} from shared config)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable warmup phase (set start_steps to 0) for comparison with PETS",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to write per-episode metrics CSV",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to write results JSON",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Optional global environment step budget (stops training once reached)",
    )
    parser.add_argument(
        "--out-steps-csv",
        type=str,
        default=None,
        help="Optional path to write per-step logs across episodes (for figures/calibration)",
    )
    args = parser.parse_args()

    # Determine warmup steps
    if args.no_warmup:
        warmup_steps = 0
    elif args.start_steps is not None:
        warmup_steps = args.start_steps
    else:
        warmup_steps = DEFAULT_WARMUP_STEPS

    results = run_training(
        num_episodes=args.episodes if args.episodes is not None else UNIFIED_EPISODES,
        max_steps_per_episode=args.steps if args.steps is not None else UNIFIED_MAX_STEPS_PER_EPISODE,
        start_steps=warmup_steps,
        seed=args.seed,
        total_steps_budget=args.total_steps,
    )

    # Optional dumps
    def _ensure_dir(path: str):
        import os
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    if args.out_csv:
        import csv
        _ensure_dir(args.out_csv)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            header = [
                "episode",
                "return",
                "ttm",
                "total_steps",
                "final_mastery",
                "cumulative_reward",
                "question_accuracy",
                "content_rate",
                "blueprint_adherence",
                "post_content_gain",
                "post_content_gain_video",
                "post_content_gain_PPT",
                "post_content_gain_text",
                "post_content_gain_blog",
                "post_content_gain_article",
                "post_content_gain_handout",
                "mean_frustration",
            ]
            writer.writerow(header)
            for i, (ret, ttm, em) in enumerate(
                zip(
                    results.get("returns", []),
                    results.get("time_to_mastery", []),
                    results.get("episode_metrics", []),
                ),
                start=1,
            ):
                row = [
                    i,
                    float(ret),
                    int(ttm) if ttm is not None else 0,
                    int(em.get("total_steps", 0)),
                    float(em.get("final_mastery", 0.0)),
                    float(em.get("cumulative_reward", 0.0)),
                    float(em.get("question_accuracy", 0.0)),
                    float(em.get("content_rate", 0.0)),
                    float(em.get("blueprint_adherence", 0.0)),
                    float(em.get("post_content_gain", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("video", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("PPT", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("text", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("blog", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("article", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("handout", 0.0)),
                    float(em.get("mean_frustration", 0.0)),
                ]
                writer.writerow(row)

    # Optional per-step dump for figures/calibration
    if args.out_steps_csv:
        import csv
        _ensure_dir(args.out_steps_csv)
        with open(args.out_steps_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "episode",
                "step",
                "event_type",
                "action_type",
                "difficulty",
                "modality",
                "reward",
                "correct",
                "mean_mastery",
                "frustration",
                "response_time",
            ]
            writer.writerow(header)
            logs = results.get("episode_logs", [])
            metrics = results.get("episode_metrics", [])
            for ep_idx, ep_log in enumerate(logs, start=1):
                mean_mastery_final = float(metrics[ep_idx - 1].get("final_mastery", 0.0)) if metrics else 0.0
                for step_idx, entry in enumerate(ep_log, start=1):
                    # Prefer per-step mastery if available, else use final episode mastery
                    mastery_vec = entry.get("mastery_vector")
                    mean_mastery = float(np.mean(mastery_vec)) if mastery_vec is not None else mean_mastery_final
                    row = [
                        ep_idx,
                        step_idx,
                        entry.get("event_type"),
                        entry.get("action_type"),
                        entry.get("difficulty"),
                        entry.get("modality"),
                        float(entry.get("reward", 0.0)),
                        bool(entry.get("correct")) if entry.get("action_type") == "question" else '',
                        mean_mastery,
                        float(entry.get("frustration", 0.0)),
                        float(entry.get("response_time", 0.0)),
                    ]
                    writer.writerow(row)

    if args.out_json:
        import json
        _ensure_dir(args.out_json)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj)
            return obj
        
        serializable_results = convert_to_serializable(results)
        with open(args.out_json, "w") as f:
            json.dump(serializable_results, f, indent=2)
