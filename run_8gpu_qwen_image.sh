#!/bin/bash

# GRPO Training on 8 H200 GPUs - LoRA Fine-Tuning for Qwen-Image-Response
# Text-to-Image Generation using MMMG dataset

export OPENAI_API_KEY=
export WANDB_PROJECT=grpo-qwen-image
export WANDB_NAME=qwen-image-mmmg-readability-curriculum-8steps-completion-reward

accelerate launch \
    --config_file deepspeed_zero3.yaml \
    train_grpo_qwen_image.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --gen_model_path Qwen/Qwen-Image \
    --output_dir outputs/dapo_qwen_image_reason_cl_mmmgscore_8steps_completion_reward \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --max_steps 500 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --levels "all" \
    --disciplines "all" \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --alignment_weight 0.5 \
    --quality_weight 0.5 \
    --n_evals 3 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 10 \
    --save_only_model True \
    --dtype bfloat16 \
    --num_generations 16 \
    --report_to wandb \
    --loss_type "dapo" \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
    --lora_r 128 \
    --lora_alpha 256 \
    --gradient_checkpointing True \
    --epsilon_high 0.28 \
    --lr_scheduler_type "constant_with_warmup" \
    --num_diff_lvls 5 \
    --difficulty_summary_path mmmg_difficulty_summary.json \
    --sam2_checkpoint /home/coder/work/MMMG/mmmg_eval/sam2/checkpoints/sam2.1_hiera_large.pt \
    --verbose True \
    --wandb_log_images 8 \
    --wandb_image_size 512 \
    --use_completion_quality_reward \
    --completion_reward_weight 0.5 \
    --max_completion_eval_timeout 120

# Effective batch size: 1 × 8 × 2 = 16
# Dataset: MMMG (all levels and disciplines)
# Task: Text-to-Image Generation

# Wandb Optimization Settings:
# - verbose True: Enable detailed logging with rank information
# - wandb_log_images 4: Log only top 4 + bottom 4 images per rank (8 total per rank, 64 total across 8 GPUs)
#   This reduces upload volume by ~87.5% (from 128 to 64 images per step)
# - wandb_image_size 512: Resize images to max 512px dimension before upload
#   This reduces image file size significantly (original images are much larger)
# Benefits:
#   - Async logging: Non-rank-0 GPUs continue training while rank 0 uploads
#   - JPEG compression: 85% quality, optimized, much faster than PNG
#   - Sampled logging: Only best/worst examples logged, reduces bandwidth
#   - Single table: Unified view across all ranks, easier to compare
# To disable wandb image logging entirely, set --wandb_log_images 0


