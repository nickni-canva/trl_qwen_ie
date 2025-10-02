#!/bin/bash

# GRPO Training on 8 H200 GPUs - LoRA Fine-Tuning
# Clean implementation following grpo_vlm.py

export OPENAI_API_KEY="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"

accelerate launch \
    --config_file deepspeed_zero2.yaml \
    train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --edit_model_path Qwen/Qwen-Image-Edit-2509 \
    --output_dir outputs/drgrpo_qwenie \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --max_steps 1000 \
    --per_device_train_batch_size 2 \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --complexity 8 \
    --image_type real \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --alignment_weight 0.6 \
    --quality_weight 0.4 \
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
    --loss_type "dr_grpo" \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
    --lora_r 128 \
    --lora_alpha 256 \
    --gradient_checkpointing False

# Effective batch size: 1 × 8 × 2 = 16

