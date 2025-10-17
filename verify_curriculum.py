#!/usr/bin/env python3
"""
Verify curriculum learning is working by checking training logs or wandb data.

This script helps validate that harder samples are being introduced progressively.
"""

import json
import sys

def load_difficulty_summary(path='mmmg_difficulty_summary.json'):
    """Load the difficulty summary"""
    with open(path, 'r') as f:
        return json.load(f)

def get_level_for_tuple(level, discipline, summary):
    """Find which difficulty level a (level, discipline) tuple belongs to"""
    tuples = summary['tuples']
    num_levels = 5  # Match training config
    
    # Find tuple index
    tuple_idx = None
    for i, t in enumerate(tuples):
        if t['level'] == level and t['discipline'] == discipline:
            tuple_idx = i
            break
    
    if tuple_idx is None:
        return None
    
    # Calculate which level this tuple belongs to
    total = len(tuples)
    group_size = total // num_levels
    remainder = total % num_levels
    
    cumulative = 0
    for i in range(num_levels):
        current_group_size = group_size + (1 if i < remainder else 0)
        cumulative += current_group_size
        if tuple_idx < cumulative:
            return i + 1
    
    return num_levels

def analyze_curriculum_from_logs(log_file=None):
    """
    Analyze curriculum progression from logs.
    
    If log_file is provided, parse it. Otherwise, provide instructions.
    """
    summary = load_difficulty_summary()
    
    if log_file:
        print("Analyzing curriculum from log file...")
        print("(Implementation: parse your training logs here)")
    else:
        print("=" * 80)
        print("CURRICULUM LEARNING VERIFICATION GUIDE")
        print("=" * 80)
        
        print("\n📋 Expected Progression (with num_diff_lvls=5, max_steps=500):")
        print("-" * 80)
        
        steps = [0, 100, 200, 300, 400, 500]
        max_steps = 500
        num_levels = 5
        
        for step in steps:
            progress = step / max_steps
            level = min(int(progress * num_levels), num_levels - 1) + 1
            
            # Count cumulative samples
            tuples = summary['tuples']
            total = len(tuples)
            group_size = total // num_levels
            remainder = total % num_levels
            
            cumulative = 0
            for i in range(level):
                current_group_size = group_size + (1 if i < remainder else 0)
                cumulative += current_group_size
            
            print(f"Step {step:3}: Level {level}/5 → {cumulative:2} tuples available")
        
        print("\n" + "=" * 80)
        print("📊 HOW TO VERIFY IN YOUR LOGS:")
        print("=" * 80)
        
        print("\n1. Check wandb logs for (level, discipline) pairs at different steps:")
        print("   - Before step 100: Only Level 1 tuples (12 total)")
        print("   - After step 100:  Level 1 + 2 tuples (24 total)")
        print("   - After step 200:  Level 1 + 2 + 3 tuples (36 total)")
        print("   - etc.")
        
        print("\n2. Look for these specific indicators:")
        
        print("\n   ✅ At step 100, you should START seeing:")
        level2_tuples = summary['tuples'][12:24]
        for i, t in enumerate(level2_tuples[:5]):  # Show first 5
            print(f"      • {t['level']:18} {t['discipline']:15}")
        print("      ... (and 7 more Level 2 tuples)")
        
        print("\n   ✅ At step 200, you should START seeing:")
        level3_tuples = summary['tuples'][24:36]
        for i, t in enumerate(level3_tuples[:5]):  # Show first 5
            print(f"      • {t['level']:18} {t['discipline']:15}")
        print("      ... (and 7 more Level 3 tuples)")
        
        print("\n3. Key Point: OLD samples are still valid!")
        print("   - Seeing 'preschool Biology' at step 100 is CORRECT")
        print("   - It's in Level 1, which remains available at all steps")
        print("   - The key is that NEW harder samples are also appearing")
        
        print("\n" + "=" * 80)
        print("🔍 QUICK CHECK:")
        print("=" * 80)
        print("\nGrep your training logs for level/discipline mentions:")
        print("  grep -E '(Level [0-9]|preschool|primaryschool|PhD)' training.log | grep -E 'step.*[0-9]+'")
        
        print("\nOr check wandb:")
        print("  1. Go to your wandb run")
        print("  2. Check 'completions_with_images' or 'generated_images' table")
        print("  3. Filter by step > 100")
        print("  4. Look at the 'prompt' or 'key' fields")
        print("  5. If you see ANY primaryschool/highschool/PhD samples, curriculum is working!")

def check_specific_sample(level, discipline):
    """Check which difficulty level a specific sample belongs to"""
    summary = load_difficulty_summary()
    curriculum_level = get_level_for_tuple(level, discipline, summary)
    
    if curriculum_level is None:
        print(f"Sample '{level} {discipline}' not found in difficulty summary")
        return
    
    print(f"\n✓ Sample: {level} {discipline}")
    print(f"  Curriculum Level: {curriculum_level}/5")
    
    # Show at which step this should become available
    max_steps = 500
    num_levels = 5
    step_per_level = max_steps / num_levels
    first_step = int((curriculum_level - 1) * step_per_level)
    
    print(f"  Available from: Step {first_step}")
    print(f"  (Becomes available at Level {curriculum_level})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            if len(sys.argv) >= 4:
                level = sys.argv[2]
                discipline = sys.argv[3]
                check_specific_sample(level, discipline)
            else:
                print("Usage: python verify_curriculum.py check <level> <discipline>")
                print("Example: python verify_curriculum.py check preschool Biology")
        else:
            analyze_curriculum_from_logs(sys.argv[1])
    else:
        analyze_curriculum_from_logs()

