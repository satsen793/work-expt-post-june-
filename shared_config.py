"""
Unified configuration for dqn_ver3 and pets_ver3 - ensures 1:1 replication.
Both projects must use these values for consistency and fair comparison.
"""

# Unified random seeds for reproducible multi-seed experiments
UNIFIED_SEEDS = [0, 1, 2, 3, 4]

# Unified episode budget: 295 episodes × 5 seeds ≈ 30,500 total environment steps
# Assuming ~100-140 steps per episode, this gives ~30k total steps
UNIFIED_EPISODES = 295

# Fixed steps per episode (max for DQN; fixed for PETS)
UNIFIED_MAX_STEPS_PER_EPISODE = 140

# Default warmup steps (for DQN; ignored by PETS)
DEFAULT_WARMUP_STEPS = 5000

# Configuration metadata
CONFIG_VERSION = "1.0"
CONFIG_DATE = "2026-01-11"
CONFIG_NOTES = """
Purpose: Ensure 1:1 replication between DQN_VER3 and PETS_VER3
- Both algorithms train on identical MDP (spec_simulator.md)
- Both use identical seeds for fair algorithmic comparison
- Both use identical episode budget (~30k steps total)
- Different episode length strategies documented in training scripts:
  * DQN: Variable [80, 140] steps per episode (random per episode)
  * PETS: Fixed 140 steps per episode
- Output formats remain separate (DQN: CSV/JSON; PETS: LaTeX/PNG) by design
"""
