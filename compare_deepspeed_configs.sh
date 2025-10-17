#!/bin/bash

# Compare DeepSpeed ZeRO Stages
# Runs tests with ZeRO-2, ZeRO-3, and ZeRO-3+Offload to find the best configuration

export WANDB_MODE=disabled

echo "================================================================"
echo "DeepSpeed ZeRO Configuration Comparison"
echo "================================================================"
echo ""
echo "This script will test 3 configurations:"
echo "  1. ZeRO-2: Baseline (optimizer + gradient partitioning)"
echo "  2. ZeRO-3: + Parameter partitioning"
echo "  3. ZeRO-3 + Offload: + CPU offloading (maximum memory savings)"
echo ""
echo "Each test runs 5 steps to measure:"
echo "  - GPU memory usage"
echo "  - Training speed (time per step)"
echo "  - Success/failure"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Create output directory for results
mkdir -p outputs/deepspeed_comparison
RESULTS_FILE="outputs/deepspeed_comparison/results.txt"
echo "DeepSpeed ZeRO Comparison Results" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Test 1: ZeRO-2
echo ""
echo "================================================================"
echo "Test 1/3: ZeRO-2 (Baseline)"
echo "================================================================"
echo ""

START_TIME=$(date +%s)

accelerate launch \
    --config_file deepspeed_zero2.yaml \
    test_grpo_config.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir outputs/deepspeed_comparison/zero2 \
    --max_steps 5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_generations 16 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --logging_steps 1 \
    --save_strategy no \
    --dtype bfloat16 \
    --report_to none \
    --loss_type "dapo" \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_samples 50 \
    --verbose False \
    2>&1 | tee outputs/deepspeed_comparison/zero2.log

ZERO2_EXIT=$?
END_TIME=$(date +%s)
ZERO2_TIME=$((END_TIME - START_TIME))

echo "" >> $RESULTS_FILE
echo "ZeRO-2 Results:" >> $RESULTS_FILE
if [ $ZERO2_EXIT -eq 0 ]; then
    echo "  Status: SUCCESS ✅" >> $RESULTS_FILE
    echo "  Time: ${ZERO2_TIME}s" >> $RESULTS_FILE
    # Extract max memory from log
    MAX_MEM=$(grep "Max allocated" outputs/deepspeed_comparison/zero2.log | tail -1 | grep -oP '\d+\.\d+' | head -1)
    if [ ! -z "$MAX_MEM" ]; then
        echo "  Max GPU Memory: ${MAX_MEM} GB" >> $RESULTS_FILE
    fi
else
    echo "  Status: FAILED ❌ (likely OOM)" >> $RESULTS_FILE
fi

# Test 2: ZeRO-3
echo ""
echo "================================================================"
echo "Test 2/3: ZeRO-3 (Parameter Partitioning)"
echo "================================================================"
echo ""

START_TIME=$(date +%s)

accelerate launch \
    --config_file deepspeed_zero3.yaml \
    test_grpo_config.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir outputs/deepspeed_comparison/zero3 \
    --max_steps 5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_generations 16 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --logging_steps 1 \
    --save_strategy no \
    --dtype bfloat16 \
    --report_to none \
    --loss_type "dapo" \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_samples 50 \
    --verbose False \
    2>&1 | tee outputs/deepspeed_comparison/zero3.log

ZERO3_EXIT=$?
END_TIME=$(date +%s)
ZERO3_TIME=$((END_TIME - START_TIME))

echo "" >> $RESULTS_FILE
echo "ZeRO-3 Results:" >> $RESULTS_FILE
if [ $ZERO3_EXIT -eq 0 ]; then
    echo "  Status: SUCCESS ✅" >> $RESULTS_FILE
    echo "  Time: ${ZERO3_TIME}s" >> $RESULTS_FILE
    MAX_MEM=$(grep "Max allocated" outputs/deepspeed_comparison/zero3.log | tail -1 | grep -oP '\d+\.\d+' | head -1)
    if [ ! -z "$MAX_MEM" ]; then
        echo "  Max GPU Memory: ${MAX_MEM} GB" >> $RESULTS_FILE
    fi
    # Calculate slowdown
    if [ $ZERO2_EXIT -eq 0 ]; then
        SLOWDOWN=$(echo "scale=1; ($ZERO3_TIME - $ZERO2_TIME) * 100 / $ZERO2_TIME" | bc)
        echo "  Slowdown vs ZeRO-2: ${SLOWDOWN}%" >> $RESULTS_FILE
    fi
else
    echo "  Status: FAILED ❌ (likely OOM)" >> $RESULTS_FILE
fi

# Test 3: ZeRO-3 + Offload (only if ZeRO-3 failed or we want maximum memory test)
echo ""
echo "================================================================"
echo "Test 3/3: ZeRO-3 + CPU Offload (Maximum Memory Savings)"
echo "================================================================"
echo ""
echo "Note: This is much slower but uses minimal GPU memory"
echo ""

START_TIME=$(date +%s)

accelerate launch \
    --config_file deepspeed_zero3_offload.yaml \
    test_grpo_config.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir outputs/deepspeed_comparison/zero3_offload \
    --max_steps 5 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --num_generations 24 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --logging_steps 1 \
    --save_strategy no \
    --dtype bfloat16 \
    --report_to none \
    --loss_type "dapo" \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_samples 50 \
    --verbose False \
    2>&1 | tee outputs/deepspeed_comparison/zero3_offload.log

OFFLOAD_EXIT=$?
END_TIME=$(date +%s)
OFFLOAD_TIME=$((END_TIME - START_TIME))

echo "" >> $RESULTS_FILE
echo "ZeRO-3 + Offload Results:" >> $RESULTS_FILE
if [ $OFFLOAD_EXIT -eq 0 ]; then
    echo "  Status: SUCCESS ✅" >> $RESULTS_FILE
    echo "  Time: ${OFFLOAD_TIME}s" >> $RESULTS_FILE
    MAX_MEM=$(grep "Max allocated" outputs/deepspeed_comparison/zero3_offload.log | tail -1 | grep -oP '\d+\.\d+' | head -1)
    if [ ! -z "$MAX_MEM" ]; then
        echo "  Max GPU Memory: ${MAX_MEM} GB" >> $RESULTS_FILE
    fi
    # Calculate slowdown
    if [ $ZERO2_EXIT -eq 0 ]; then
        SLOWDOWN=$(echo "scale=1; ($OFFLOAD_TIME - $ZERO2_TIME) * 100 / $ZERO2_TIME" | bc)
        echo "  Slowdown vs ZeRO-2: ${SLOWDOWN}%" >> $RESULTS_FILE
    fi
else
    echo "  Status: FAILED ❌" >> $RESULTS_FILE
fi

# Print summary
echo ""
echo "================================================================"
echo "COMPARISON SUMMARY"
echo "================================================================"
echo ""
cat $RESULTS_FILE
echo ""
echo "================================================================"
echo "RECOMMENDATIONS"
echo "================================================================"
echo ""

if [ $ZERO2_EXIT -eq 0 ]; then
    echo "✅ ZeRO-2 works: USE THIS (fastest option)"
    echo "   Config: deepspeed_zero2.yaml"
elif [ $ZERO3_EXIT -eq 0 ]; then
    echo "✅ ZeRO-3 works: USE THIS (good balance)"
    echo "   Config: deepspeed_zero3.yaml"
    echo "   Trade-off: ~10-20% slower but uses less memory"
elif [ $OFFLOAD_EXIT -eq 0 ]; then
    echo "✅ ZeRO-3 + Offload works: USE THIS (last resort)"
    echo "   Config: deepspeed_zero3_offload.yaml"
    echo "   Trade-off: ~30-50% slower but minimal GPU memory"
else
    echo "❌ All configs failed!"
    echo ""
    echo "Further optimization needed:"
    echo "  - Reduce per_device_train_batch_size"
    echo "  - Enable gradient_checkpointing"
    echo "  - Reduce lora_r"
    echo "  - Reduce num_generations"
fi

echo ""
echo "Detailed logs saved to: outputs/deepspeed_comparison/"
echo "Results summary: $RESULTS_FILE"
echo ""

