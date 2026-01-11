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
    print(f"⏱️  Starting at {os.popen('date').read().strip()}")
    
    try:
        print(f"📝 Executing... (timeout: 5 min)")
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show live output instead of capturing
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} - PASSED")
            print(f"⏱️  Completed successfully\n")
            return True
        else:
            print(f"\n❌ {description} - FAILED")
            print(f"Return code: {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {description} - TIMEOUT (>5 min)")
        print(f"Process took too long to complete")
        return False
    except Exception as e:
        print(f"\n💥 {description} - EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_json_output(json_path: str, algo_name: str) -> bool:
    """Verify JSON output contains required fields."""
    print(f"\n🔍 Verifying {algo_name} JSON output...")
    print(f"   Looking for: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Contents of output dir: {os.listdir(os.path.dirname(json_path)) if os.path.exists(os.path.dirname(json_path)) else 'DIR NOT FOUND'}")
        return False
    
    print(f"✅ JSON file found, parsing...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"   Total keys in JSON: {len(data)}")
        print(f"   Keys: {list(data.keys())}")
        
        # Check for required fields
        required = ["auc_10k", "checkpoints", "wall_clock_time_minutes"]
        missing = []
        
        for field in required:
            if field not in data:
                missing.append(field)
                print(f"   ❌ Missing: {field}")
            else:
                print(f"   ✅ Found: {field}")
        
        if missing:
            print(f"❌ Missing required fields: {', '.join(missing)}")
            return False
        
        # Verify checkpoints structure
        checkpoints = data.get("checkpoints", {})
        if not isinstance(checkpoints, dict):
            print(f"❌ 'checkpoints' should be dict, got {type(checkpoints)}")
            return False
        
        print(f"✅ {algo_name} JSON verified:")
        print(f"   📊 AUC@10k: {data['auc_10k']:.2f}")
        print(f"   🏁 Checkpoints: {list(checkpoints.keys())}")
        print(f"   ⏱️  Wall-clock time: {data['wall_clock_time_minutes']:.2f} min")
        
        # Show additional metrics if available
        if "calibration_data" in data:
            print(f"   📈 Calibration data present")
        if "modality_gains" in data:
            print(f"   🎯 Modality gains present")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        print(f"   File path: {json_path}")
        return False
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        import traceback
        traceback.print_exc()
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
    print(f"🖥️  Python: {sys.version.split()[0]}")
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"⏰ Start time: {os.popen('date').read().strip()}")
    print("="*60)
    
    # Create temp output directory
    os.makedirs("smoketest_outputs", exist_ok=True)
    print(f"\n📁 Created output directory: smoketest_outputs/")
    print(f"   Contents before tests: {os.listdir('smoketest_outputs')}\n")
    
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
                "--seed", "0",
                "--episodes", "5"
            ],
            "json_path": "pets_ver3/results/performance_summary.json"  # Relative to smoke test working dir
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
    test_number = 1
    
    for test in tests:
        print(f"\n{'='*60}")
        print(f"🔄 Test {test_number}/{len(tests)}: {test['name']}")
        print(f"{'='*60}")
        
        success = run_command(test["cmd"], f"{test['name']} Training")
        results[test["name"]] = {"training": success}
        
        if success and test.get("json_path"):
            print(f"\n📋 Verifying {test['name']} outputs...")
            json_valid = verify_json_output(test["json_path"], test["name"])
            results[test["name"]]["json"] = json_valid
        else:
            print(f"⏭️  Skipping JSON verification (training failed)")
            results[test["name"]]["json"] = None
        
        test_number += 1
    
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
    print(f"⏰ End time: {os.popen('date').read().strip()}")
    print("="*60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Ready to run full experiments on Lightning AI")
        print("="*60)
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❌ Fix errors before running full experiments")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
