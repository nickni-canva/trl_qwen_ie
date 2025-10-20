# generate_with_embeddings Token Length Test Suite

## Overview

This test suite thoroughly tests the `generate_with_embeddings` function from `train_grpo_qwen_image.py` to ensure it handles various token lengths correctly without causing size mismatch errors in rotary embeddings.

## Problem Being Tested

Previously, the function would sometimes encounter errors like:
```
Size mismatch in rotary embeddings (512 vs 489)
```

This test suite verifies that the function works correctly for:
- Short sequences (< 512 tokens)
- Medium sequences (~512 tokens)
- Long sequences (> 512 tokens)
- Mixed length batches
- Edge cases (very short and very long sequences)
- Various batch sizes

## Files

- **`test_generate_embeddings_lengths.py`** - Main test file with comprehensive test cases
- **`run_embedding_length_tests.sh`** - Shell script to run tests easily
- **`TEST_GENERATE_EMBEDDINGS_README.md`** - This documentation file

## Test Cases

### Test 1: All Short Sequences (< 512 tokens)
- 16 samples with completion texts ~50-100 words
- Expected token length: ~200-400 after system prompt
- Verifies: No crashes, 16 images returned

### Test 2: All Medium Sequences (~512 tokens)
- 16 samples with completion texts ~150-200 words
- Expected token length: ~512 tokens
- Verifies: No crashes at the critical 512 token boundary

### Test 3: All Long Sequences (> 512 tokens)
- 16 samples with completion texts ~300+ words
- Expected token length: ~600-800 tokens
- Verifies: Handles padding correctly for longer sequences

### Test 4: Mixed Lengths in Same Batch
- 4 short (~200 tokens)
- 4 medium (~512 tokens)
- 4 long (~700 tokens)
- 4 very long (~1000 tokens)
- Verifies: Padding/masking works correctly for heterogeneous batches

### Test 5: Exact 489 Tokens (⚠️ CRITICAL - Original Error Trigger)
- 16 samples with completions targeting exactly ~489 tokens
- **This is the exact problematic length from the original error: "Size mismatch (512 vs 489)"**
- Verifies: The specific token count that caused the original bug is now handled correctly
- This test is the most important for regression prevention

### Test 6: Edge Cases
- 1 very short (minimal text, ~50 tokens)
- 1 very long (maximum text, ~1500 tokens)
- Verifies: Extreme cases are handled gracefully

### Test 7: Different Batch Sizes
- Test 7a: Batch size = 1
- Test 7b: Batch size = 8
- Test 7c: Batch size = 32
- Verifies: Sub-batching logic works for all sizes, dynamic batching adapts correctly

## Usage

### Method 1: Using the shell script (recommended)

```bash
cd /home/coder/work/trl_qwen_image_edit
./run_embedding_length_tests.sh
```

This will:
- Run all test cases
- Save output to `test_generate_embeddings_lengths.log`
- Print a summary at the end

### Method 2: Direct Python execution

```bash
cd /home/coder/work/trl_qwen_image_edit
export PYTHONPATH=/home/coder/work/trl_qwen_image_edit:/home/coder/work/customized_diffusers/src:$PYTHONPATH
python test_generate_embeddings_lengths.py
```

## Expected Output

Each test will show:
1. Token counts for all samples
2. Embedding shapes and mask dimensions
3. Number of images generated
4. Image validation results
5. Pass/fail status

Final summary shows:
```
FINAL TEST SUMMARY
==================
Total Tests: 10
Passed: 10
Failed: 0

✅ ALL TESTS PASSED!
   generate_with_embeddings handles all token lengths correctly:
   - Short sequences (< 512 tokens)
   - Medium sequences (~512 tokens)
   - Long sequences (> 512 tokens)
   - Mixed length batches
   - ⚠️  EXACT 489 TOKENS (the original error trigger)
   - Edge cases (very short and very long)
   - Various batch sizes (1, 8, 16, 32)
   - No size mismatch errors
   - No OOM errors
```

## Requirements

- PyTorch
- Transformers (with Qwen2VL support)
- Diffusers (custom version with QwenImageResponsePipeline)
- CUDA-capable GPU (recommended, but can run on CPU)
- Sufficient VRAM/RAM for model loading

## Implementation Details

The test suite:
1. Loads the actual VLM model (Qwen2.5-VL-7B-Instruct)
2. Loads the actual diffusion pipeline (Qwen-Image)
3. Applies the same optimizations as the training code
4. Tests the real `extract_response_embeddings` and `generate_with_embeddings` functions
5. Verifies that all generated images are valid PIL images with correct dimensions

## Success Criteria

All tests pass without:
- Size mismatch errors in rotary embeddings
- OOM errors
- Crashes or exceptions
- Each test generates the correct number of valid images

## Troubleshooting

### OOM Errors
If you encounter OOM errors, the test will still complete and report which tests failed. Consider:
- Running tests individually
- Reducing batch sizes in test cases
- Using a GPU with more VRAM

### Missing Dependencies
Ensure you have the custom diffusers package installed:
```bash
pip install -e /home/coder/work/customized_diffusers
```

### Path Issues
Make sure PYTHONPATH includes both:
- `/home/coder/work/trl_qwen_image_edit` (for train_grpo_qwen_image.py)
- `/home/coder/work/customized_diffusers/src` (for QwenImageResponsePipeline)

## References

- Training script: `train_grpo_qwen_image.py`
- Function under test: `generate_with_embeddings` (lines 2143-2229)
- Related function: `extract_response_embeddings` (lines 2034-2132)
- Pipeline initialization: lines 665-713

