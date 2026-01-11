#!/usr/bin/env python3
"""
Quick smoke test to verify all 4 algorithms run without crashes after Option B fixes.
Tests minimal configuration (5 episodes, seed 0) to verify:
1. Imports work
2. New metrics compute correctly
3. JSON exports contain required fields

Run this BEFORE full experiments on Lightning AI!
"""
import subprocess
import sys
import json
import os
from pathlib import Path

def run_command(cmd: list, description: str) -> bool:
    """Run command and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"STDERR: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT (>5 min)")
        return False
    except Exception as e:
        print(f"💥 {description} - EXCEPTION: {e}")
        return False


def verify_json_output(json_path: str, algo_name: str) -> bool:
    """Verify JSON output contains required fields."""
    print(f"\n🔍 Verifying {algo_name} JSON output...")
    
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return False
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check for required fields
        required = ["auc_10k", "checkpoints", "wall_clock_time_minutes"]
        missing = []
        
        for field in required:
            if field not in data:
                missing.append(field)
        
        if missing:
            print(f"❌ Missing fields: {', '.join(missing)}")
            return False
        
        # Verify checkpoints structure
        checkpoints = data.get("checkpoints", {})
        if not isinstance(checkpoints, dict):
            print(f"❌ 'checkpoints' should be dict, got {type(checkpoints)}")
            return False
        
        print(f"✅ {algo_name} JSON verified:")
        print(f"   - AUC@10k: {data['auc_10k']:.2f}")
        print(f"   - Checkpoints: {list(checkpoints.keys())}")
        print(f"   - Wall-clock time: {data['wall_clock_time_minutes']:.2f} min")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return False


def main():
    """Run smoke tests for all 4 algorithms."""
    print("="*60)
    print("🚀 OPTION B SMOKE TEST")
    print("="*60)
    print("Testing: DQN, PETS, MBPO, PPO")
    print("Config: 5 episodes, seed 0, minimal runtime")
    print("Goal: Verify no crashes + new metrics export correctly")
    print("="*60)
    
    # Create temp output directory
    os.makedirs("smoketest_outputs", exist_ok=True)
    
    tests = [
        {
            "name": "DQN",
            "cmd": [
                sys.executable,
                "dqn_ver3/train_dqn.py",
                "--seed", "0",
                "--episodes", "5",
                "--no-warmup",
                "--out-json", "smoketest_outputs/dqn_smoke.json"
            ],
            "json_path": "smoketest_outputs/dqn_smoke.json"
        },
        {
            "name": "PETS",
            "cmd": [
                sys.executable,
                "pets_ver3/pets_train.py",
                # PETS uses TrainConfig internally, no CLI args needed for smoke test
            ],
            "json_path": "pets_ver3/results/summary.json"  # PETS default output location
        },
        {
            "name": "MBPO",
            "cmd": [
                sys.executable,
                "mbpo_ver3/train_mbpo.py",
                "--seed", "0",
                "--episodes", "5",
                "--output", "smoketest_outputs/mbpo"
            ],
            "json_path": "smoketest_outputs/mbpo/summary.json"
        },
        {
            "name": "PPO",
            "cmd": [
                sys.executable,
                "ppo_ver3/ppo_train.py",
                "--seed", "0",
                "--episodes", "5",
                "--output", "smoketest_outputs/ppo"
            ],
            "json_path": "smoketest_outputs/ppo/summary.json"
        }
    ]
    
    results = {}
    
    for test in tests:
        success = run_command(test["cmd"], f"{test['name']} Training")
        results[test["name"]] = {"training": success}
        
        if success and test.get("json_path"):
            json_valid = verify_json_output(test["json_path"], test["name"])
            results[test["name"]]["json"] = json_valid
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SMOKE TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for algo, status in results.items():
        training = "✅" if status.get("training") else "❌"
        json_status = status.get("json")
        json_str = "✅" if json_status else ("❌" if json_status is False else "⏭️")
        
        print(f"{algo:6s}: Training {training}  JSON {json_str}")
        
        if not status.get("training") or (json_status is not None and not json_status):
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Ready to run full experiments on Lightning AI")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❌ Fix errors before running full experiments")
        return 1


if __name__ == "__main__":
    sys.exit(main())
