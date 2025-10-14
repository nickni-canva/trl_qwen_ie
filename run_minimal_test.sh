#!/bin/bash

# Minimal test run for train_grpo_qwen_image.py
# This runs a very short training (2 steps) to verify the script works

echo "=================================="
echo "Minimal Training Test"
echo "=================================="

# Set API key from the main script (for GPT-4V evaluation)
export OPENAI_API_KEY=

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
    echo "✅ GPU available"
else
    echo "⚠️  No GPU detected - training will be very slow"
fi

echo ""
echo "Configuration:"
echo "  - Max steps: 2 (minimal)"
echo "  - Batch size: 1"
echo "  - Dataset: preschool Math only"
echo "  - Generations: 2 per prompt"
echo "  - LoRA rank: 8 (minimal)"
echo ""
echo "Starting training..."
echo "=================================="
echo ""

# Run with single GPU (or CPU if no GPU)
python train_grpo_qwen_image.py \
    --config test_config_minimal.yaml \
    2>&1 | tee /tmp/test_grpo_qwen_image.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully!"
    echo ""
    echo "Check the log for details:"
    echo "  tail -100 /tmp/test_grpo_qwen_image.log"
else
    echo "❌ Test failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check the log for errors:"
    echo "  tail -100 /tmp/test_grpo_qwen_image.log"
fi
echo "=================================="

exit $EXIT_CODE


