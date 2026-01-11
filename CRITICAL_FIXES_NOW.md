# 🔴 CRITICAL FIXES REQUIRED BEFORE TRAINING

**Time Required:** 15 minutes  
**Priority:** MUST FIX NOW

---

## Fix 1: DQN - Add wall_clock_time_minutes (5 min)

**File:** `dqn_ver3/train_dqn.py`

**Problem:** Table 1 in template requires wall-clock time, but DQN doesn't export it.

**Solution:** Add timing to multi-seed summary function around line 1124:

```python
# At start of multi_seed_summary or equivalent:
import time
start_time = time.time()

# ... run training ...

# In return dict:
return {
    "cumulative_reward": summarize(cumulative_rewards),
    "time_to_mastery": {**summarize(ttms), **median_iqr(ttms)},
    "blueprint_adherence": summarize(blueprint),
    "post_content_gain": summarize(post_content),
    "policy_stability": stability,
    "wall_clock_time_minutes": {
        "mean": (time.time() - start_time) / 60.0,
        "std": 0.0  # Or compute across seeds if multi-seed runner
    },
    "total_steps_per_episode": total_steps_list  # Also add this
}
```

---

## Fix 2: MBPO - Fix calibration_mae aggregation (5 min)

**File:** `mbpo_ver3/train_mbpo.py`

**Problem:** Single-seed returns `calibration_data.mae` as float, but Table 1 needs `calibration_mae.mean ± std` across seeds.

**Solution:** In multi-seed runner, collect MAE per seed:

```python
# After running all seeds:
all_mae_values = [result["calibration_data"]["mae"] for result in results_per_seed]

# In final summary:
return {
    # ... other metrics ...
    "calibration_mae": {
        "mean": float(np.mean(all_mae_values)),
        "std": float(np.std(all_mae_values))
    }
}
```

---

## Fix 3: DQN - Add total_steps_per_episode (5 min)

**File:** `dqn_ver3/train_dqn.py`

**Problem:** AUC@10k and checkpoints need episode step counts.

**Solution:** Track and export:

```python
# In run_training():
total_steps_list = []

for ep in episodes:
    # ... episode logic ...
    total_steps_list.append(episode_total_steps)

# In return:
return {
    # ... existing fields ...
    "total_steps_per_episode": total_steps_list
}
```

---

## Verification Steps

After fixes:

1. **Run smoke test:**
   ```bash
   python dqn_ver3/train_dqn.py --episodes 10 --seed 0
   python mbpo_ver3/train_mbpo.py --seed 0 # (check output)
   ```

2. **Verify JSON outputs contain:**
   - DQN: `wall_clock_time_minutes`, `total_steps_per_episode`
   - MBPO: `calibration_mae` with `mean` and `std` keys

3. **Test Table 1 generation:**
   ```bash
   python compare_all_4.py --dqn results/dqn/summary.json \
                           --pets results/pets/summary.json \
                           --mbpo results/mbpo/summary.json \
                           --ppo results/ppo/summary.json \
                           --output comparison/
   ```

4. **Check for errors in table_performance_comparison.tex**

---

## After These Fixes: GO FOR TRAINING 🚀

All critical blockers will be resolved. Launch 30+ hours of Lightning AI training with confidence.
