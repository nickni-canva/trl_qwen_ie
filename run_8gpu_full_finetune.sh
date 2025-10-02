#!/bin/bash

# GRPO Training on 8 H200 GPUs - FULL FINE-TUNING (No LoRA)
# Clean implementation following grpo_vlm.py

export OPENAI_API_KEY="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"

accelerate launch \
    --config_file deepspeed_zero2.yaml \
    train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --edit_model_path Qwen/Qwen-Image-Edit-2509 \
    --output_dir outputs/grpo_full_finetune \
    --learning_rate 5e-6 \
    --num_train_epochs 3 \
    --max_steps 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --complexity 8 \
    --image_type real \
    --alignment_weight 0.6 \
    --quality_weight 0.4 \
    --n_evals 3 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 20 \
    --save_only_model True \
    --bf16 True \
    --num_generations 8 \
    --generation_batch_size 32 \
    --warmup_steps 100 \
    --report_to wandb \
    --run_name grpo_qwen_full_finetune \
    --log_completions \
    --mask_truncated_completions \
    --gradient_checkpointing True

# Note: Full fine-tuning requires more memory!
# - Reduced learning rate: 5e-6 (vs 1e-5 for LoRA)
# - Increased gradient accumulation: 4 (vs 2)
# - Enabled gradient checkpointing for memory efficiency
# - NO --use_peft flag = full fine-tuning
# - Effective batch size: 1 × 8 × 4 = 32


