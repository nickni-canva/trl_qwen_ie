#!/bin/bash
#
# Run comprehensive tests for generate_with_embeddings function
# Tests various token lengths to ensure no size mismatch errors
#

echo "======================================================================"
echo "Running generate_with_embeddings Token Length Tests"
echo "======================================================================"
echo ""
echo "This will test the following scenarios:"
echo "  1. All Short Sequences (< 512 tokens)"
echo "  2. All Medium Sequences (~512 tokens)"
echo "  3. All Long Sequences (> 512 tokens)"
echo "  4. Mixed Lengths in Same Batch"
echo "  5. Edge Cases (very short and very long)"
echo "  6. Different Batch Sizes (1, 8, 16, 32)"
echo ""
echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
sleep 3

# Set Python path
export PYTHONPATH=/home/coder/work/trl_qwen_image_edit:/home/coder/work/customized_diffusers/src:$PYTHONPATH

# Run the test
python test_generate_embeddings_lengths.py 2>&1 | tee test_generate_embeddings_lengths.log

echo ""
echo "======================================================================"
echo "Test complete! Output saved to: test_generate_embeddings_lengths.log"
echo "======================================================================"

