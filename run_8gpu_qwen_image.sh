#!/bin/bash

# GRPO Training on 8 H200 GPUs - LoRA Fine-Tuning for Qwen-Image-Response
# Text-to-Image Generation using MMMG dataset

export OPENAI_API_KEY=
export WANDB_PROJECT=grpo-qwen-image
export WANDB_NAME=qwen-image-modified-template-dapo-lora128qkvo

accelerate launch \
    --config_file deepspeed_zero2.yaml \
    train_grpo_qwen_image.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --gen_model_path Qwen/Qwen-Image \
    --output_dir outputs/grpo_qwen_image \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --max_steps 1000 \
    --per_device_train_batch_size 2 \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --levels "all" \
    --disciplines "all" \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --alignment_weight 0.7 \
    --quality_weight 0.3 \
    --n_evals 3 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 10 \
    --save_only_model True \
    --dtype bfloat16 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --report_to wandb \
    --log_completions \
    --loss_type "dapo" \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
    --lora_r 128 \
    --lora_alpha 256 \
    --gradient_checkpointing False

# Effective batch size: 1 × 8 × 2 = 16
# Dataset: MMMG (all levels and disciplines)
# Task: Text-to-Image Generation


