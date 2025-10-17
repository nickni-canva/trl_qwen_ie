#!/bin/bash

# Example script for running GRPO training with curriculum learning
# This demonstrates how to use the --num_diff_lvls argument

export OPENAI_API_KEY="your-api-key-here"

# Configuration
NUM_GPUS=8
NUM_DIFF_LVLS=3  # Use 3 difficulty levels (easy -> medium -> hard)
MAX_STEPS=300    # Curriculum transitions at steps 100 and 200

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --config_file accelerate_config.yaml \
    train_grpo_qwen_image.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --gen_model_path Qwen/Qwen-Image \
    --torch_dtype bfloat16 \
    --output_dir ./output_curriculum_3lvl \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --max_steps ${MAX_STEPS} \
    --learning_rate 1e-5 \
    --warmup_steps 10 \
    --logging_steps 10 \
    --save_steps 50 \
    --eval_steps 50 \
    --report_to wandb \
    --run_name "qwen-image-grpo-curriculum-${NUM_DIFF_LVLS}lvl" \
    --bf16 \
    --gradient_checkpointing \
    --num_sample_generations 2 \
    --levels all \
    --disciplines all \
    --local_data_path /mnt/ephemeral/MMMG_train/train.json \
    --alignment_weight 0.7 \
    --quality_weight 0.3 \
    --num_diff_lvls ${NUM_DIFF_LVLS} \
    --difficulty_summary_path mmmg_difficulty_summary.json

# Curriculum Schedule for this configuration:
# - Steps 0-99:    Train on easiest 20 tuples only (~200 samples)
# - Steps 100-199: Expand to include 40 easiest tuples (~400 samples)  
# - Steps 200-300: Use all 59 tuples (~600 samples)

