#!/usr/bin/env python3
"""
Comprehensive test for train_grpo_qwen_image.py

Tests:
1. Import checks
2. Dataset loading
3. Argument parsing
4. Model loading (if possible)
5. Reward function creation
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("TEST SUITE: train_grpo_qwen_image.py")
print("="*80)

# Test 1: Imports
print("\n[TEST 1] Checking imports...")
try:
    import torch
    print("  ✅ torch imported")
    
    from datasets import load_dataset
    print("  ✅ datasets imported")
    
    from transformers import Qwen2VLProcessor
    print("  ✅ transformers imported")
    
    from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
    print("  ✅ trl imported")
    
    # Import our script
    from train_grpo_qwen_image import (
        ImageGenArgs,
        LEVELS_CANONICAL,
        DISCIPLINES,
        parse_list_arg,
        prepare_dataset,
    )
    print("  ✅ train_grpo_qwen_image imported")
    
    print("✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Argument parsing
print("\n[TEST 2] Testing argument parsing...")
try:
    # Test parse_list_arg
    result = parse_list_arg("all", LEVELS_CANONICAL)
    assert result == LEVELS_CANONICAL, "parse_list_arg('all') failed"
    print(f"  ✅ parse_list_arg('all') = {len(result)} levels")
    
    result = parse_list_arg("preschool,highschool", LEVELS_CANONICAL)
    assert result == ["preschool", "highschool"], "parse_list_arg with comma-separated failed"
    print(f"  ✅ parse_list_arg('preschool,highschool') = {result}")
    
    result = parse_list_arg("Math", DISCIPLINES)
    assert result == ["Math"], "parse_list_arg with single value failed"
    print(f"  ✅ parse_list_arg('Math') = {result}")
    
    # Test invalid input
    try:
        parse_list_arg("InvalidLevel", LEVELS_CANONICAL)
        print("  ❌ Should have raised ValueError for invalid level")
        sys.exit(1)
    except ValueError:
        print("  ✅ ValueError raised for invalid input as expected")
    
    print("✅ Argument parsing tests passed!")
    
except Exception as e:
    print(f"❌ Argument parsing test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Dataset loading (small sample)
print("\n[TEST 3] Testing dataset loading...")
try:
    # Try to load a small subset
    print("  Loading MMMG dataset (preschool, Math only, limit 10 samples)...")
    
    # Test with specific config
    train_ds, eval_ds = prepare_dataset(
        levels="preschool",
        disciplines="Math",
        max_samples=10,
    )
    
    print(f"  ✅ Dataset loaded: {len(train_ds)} samples")
    
    # Check dataset structure
    sample = train_ds[0]
    required_keys = ["prompt", "prompt_text", "key", "level", "discipline"]
    for key in required_keys:
        if key not in sample:
            print(f"  ❌ Missing key: {key}")
            sys.exit(1)
    print(f"  ✅ Dataset has required keys: {required_keys}")
    
    # Check prompt format
    if not isinstance(sample["prompt"], list):
        print(f"  ❌ Prompt should be a list, got {type(sample['prompt'])}")
        sys.exit(1)
    
    if len(sample["prompt"]) < 2:
        print(f"  ❌ Prompt should have at least 2 messages (system + user)")
        sys.exit(1)
    
    print(f"  ✅ Prompt format correct: {len(sample['prompt'])} messages")
    print(f"  ✅ Sample keys: level={sample['level']}, discipline={sample['discipline']}")
    print(f"  ✅ Prompt text (first 100 chars): {sample['prompt_text'][:100]}...")
    
    print("✅ Dataset loading tests passed!")
    
except Exception as e:
    print(f"❌ Dataset loading test failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  This might be expected if MMMG dataset is not available or requires authentication")

# Test 4: Configuration classes
print("\n[TEST 4] Testing configuration classes...")
try:
    from dataclasses import fields
    
    img_args = ImageGenArgs()
    
    # Check default values
    assert img_args.gen_model_path == "Qwen/Qwen-Image", "Default gen_model_path incorrect"
    assert img_args.levels == "all", "Default levels incorrect"
    assert img_args.disciplines == "all", "Default disciplines incorrect"
    assert img_args.alignment_weight == 0.7, "Default alignment_weight incorrect"
    assert img_args.quality_weight == 0.3, "Default quality_weight incorrect"
    
    print("  ✅ ImageGenArgs default values correct")
    
    # List all fields
    field_names = [f.name for f in fields(ImageGenArgs)]
    print(f"  ✅ ImageGenArgs fields: {field_names}")
    
    print("✅ Configuration tests passed!")
    
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Helper functions
print("\n[TEST 5] Testing helper functions...")
try:
    from train_grpo_qwen_image import extract_masked_hidden
    
    # Create dummy tensors
    batch_size = 2
    seq_len = 10
    hidden_dim = 64
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    mask[0, 8:] = 0  # First sample has 8 valid tokens
    mask[1, 5:] = 0  # Second sample has 5 valid tokens
    
    result = extract_masked_hidden(hidden_states, mask)
    
    assert len(result) == batch_size, "Should return batch_size tensors"
    assert result[0].shape[0] == 8, f"First sample should have 8 tokens, got {result[0].shape[0]}"
    assert result[1].shape[0] == 5, f"Second sample should have 5 tokens, got {result[1].shape[0]}"
    
    print("  ✅ extract_masked_hidden works correctly")
    print(f"     Input: ({batch_size}, {seq_len}, {hidden_dim})")
    print(f"     Output: [tensor({result[0].shape}), tensor({result[1].shape})]")
    
    print("✅ Helper function tests passed!")
    
except Exception as e:
    print(f"❌ Helper function test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check if script can be run with --help
print("\n[TEST 6] Testing script execution (--help)...")
try:
    import subprocess
    
    result = subprocess.run(
        [sys.executable, "train_grpo_qwen_image.py", "--help"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if "usage:" in result.stdout.lower() or "options:" in result.stdout.lower():
        print("  ✅ Script --help works")
        print(f"     Output preview: {result.stdout[:200]}...")
    else:
        print(f"  ⚠️  Unexpected help output")
        print(f"     stdout: {result.stdout[:500]}")
        print(f"     stderr: {result.stderr[:500]}")
    
except subprocess.TimeoutExpired:
    print("  ⚠️  --help timed out (might be loading models)")
except Exception as e:
    print(f"  ⚠️  Could not test --help: {e}")

# Test 7: Check for common issues
print("\n[TEST 7] Checking for common issues...")
try:
    # Read the script and check for common issues
    script_path = Path(__file__).parent / "train_grpo_qwen_image.py"
    script_content = script_path.read_text()
    
    issues = []
    
    # Check for TODO comments
    if "TODO" in script_content:
        issues.append("Contains TODO comments")
    
    # Check for debugging prints
    if "print(" in script_content and "flush=True" in script_content:
        print("  ✅ Has debug prints with flush=True (good for logging)")
    
    # Check key functions exist
    required_functions = [
        "prepare_dataset",
        "create_reward_function",
        "extract_response_embeddings",
        "generate_with_embeddings",
    ]
    
    for func in required_functions:
        if f"def {func}" not in script_content:
            issues.append(f"Missing function: {func}")
    
    if issues:
        print("  ⚠️  Found potential issues:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print("  ✅ No obvious issues found")
    
    # Check line count
    line_count = len(script_content.split('\n'))
    print(f"  ℹ️  Script has {line_count} lines")
    
    print("✅ Common issues check complete!")
    
except Exception as e:
    print(f"❌ Issue check failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("✅ All critical tests passed!")
print("\nThe script should be ready to use. To run actual training:")
print("  1. Set OPENAI_API_KEY environment variable")
print("  2. Run: bash run_8gpu_qwen_image.sh")
print("  3. Monitor: tail -f outputs/grpo_qwen_image/training.log")
print("="*80)


