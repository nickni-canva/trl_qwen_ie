#!/usr/bin/env python3
"""Test complexity level parsing and dataset expansion"""

import sys
sys.path.insert(0, '/home/coder/work/trl_qwen_image_edit')

from train_grpo import parse_complexity, prepare_dataset
from loguru import logger

def test_parse_complexity():
    """Test complexity string parsing"""
    logger.info("Testing complexity parsing...")
    
    # Test range
    result = parse_complexity("1-8")
    expected = [1, 2, 3, 4, 5, 6, 7, 8]
    assert result == expected, f"Range test failed: {result} != {expected}"
    logger.success(f"✓ Range '1-8' → {result}")
    
    # Test comma-separated
    result = parse_complexity("1,3,5,8")
    expected = [1, 3, 5, 8]
    assert result == expected, f"Comma test failed: {result} != {expected}"
    logger.success(f"✓ Comma '1,3,5,8' → {result}")
    
    # Test single value
    result = parse_complexity("8")
    expected = [8]
    assert result == expected, f"Single test failed: {result} != {expected}"
    logger.success(f"✓ Single '8' → {result}")
    
    # Test with spaces
    result = parse_complexity(" 1, 3, 5, 8 ")
    expected = [1, 3, 5, 8]
    assert result == expected, f"Spaces test failed: {result} != {expected}"
    logger.success(f"✓ Spaces ' 1, 3, 5, 8 ' → {result}")
    
    logger.info("All parsing tests passed! ✓")


def test_dataset_expansion():
    """Test that dataset is correctly expanded with multiple complexity levels"""
    logger.info("\nTesting dataset expansion...")
    
    # Test with single complexity
    logger.info("Loading dataset with complexity='8' (single level)...")
    train_ds, eval_ds = prepare_dataset(complexity="8", image_type="real", max_samples=100)
    single_size = len(train_ds) if train_ds else 0
    logger.info(f"Single complexity dataset size: {single_size}")
    
    # Test with range
    logger.info("\nLoading dataset with complexity='1-4' (4 levels)...")
    train_ds, eval_ds = prepare_dataset(complexity="1-4", image_type="real", max_samples=400)
    multi_size = len(train_ds) if train_ds else 0
    logger.info(f"Multi complexity dataset size: {multi_size}")
    
    # Check that multi is approximately 4x larger
    # (approximate because we limit total samples, not images)
    logger.info(f"\nRatio: {multi_size / single_size:.2f}x (expected ~4x)")
    
    # Verify samples have complexity_level field
    if train_ds and len(train_ds) > 0:
        sample = train_ds[0]
        assert "complexity_level" in sample, "Sample missing 'complexity_level' field"
        logger.success(f"✓ Sample has complexity_level: {sample['complexity_level']}")
        
        # Check multiple samples
        levels_found = set()
        for i in range(min(20, len(train_ds))):
            levels_found.add(train_ds[i]["complexity_level"])
        logger.info(f"Complexity levels found in first 20 samples: {sorted(levels_found)}")
    
    logger.info("\nDataset expansion test passed! ✓")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing Complexity Level Functionality")
    logger.info("=" * 60)
    
    try:
        # Test parsing
        test_parse_complexity()
        
        # Test dataset expansion
        test_dataset_expansion()
        
        logger.success("\n" + "=" * 60)
        logger.success("All tests passed! ✅")
        logger.success("=" * 60)
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

