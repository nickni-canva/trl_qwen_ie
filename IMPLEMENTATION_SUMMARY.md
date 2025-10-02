# Clean Implementation Summary

## Overview

This is a **clean, simple implementation** following the `grpo_vlm.py` pattern for training Qwen2.5-VL with GRPO for image editing.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. GRPO Training Loop                                       │
│    - Generates VLM completions (TEXT only, no embeddings)  │
│    - Completions = descriptions of how to edit the image   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Reward Function (our code)                              │
│    - Receives: completion text, input image, instruction   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Extract Embeddings (Option C)                           │
│    - Build full prompt with completion:                     │
│      [system + user + instruction + assistant + completion]│
│    - Forward pass through VLM → get hidden states          │
│    - Drop system prompt tokens                              │
│    - Result: embeddings for [user + instruction +          │
│               assistant + completion]                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Generate Edited Image                                    │
│    - Pass embeddings to QwenImageEditResponsePipeline      │
│    - Pipeline uses embeddings to condition DiT             │
│    - DiT generates edited image                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Evaluate with GPT-4V                                     │
│    - Alignment: does edit follow instruction?               │
│    - Quality: is edited image high quality?                 │
│    - Combined reward = 0.6*alignment + 0.4*quality         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. GRPO Updates VLM                                         │
│    - Uses reward to compute policy gradient                 │
│    - Updates VLM parameters via LoRA                        │
│    - DiT stays frozen                                       │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Choices

### 1. Embedding Extraction (Option C)

**User's Question**: How to get embeddings from GRPO's completions?

**Answer**: Do a forward pass on the completion text.

**Why**: 
- GRPO generates text but doesn't return hidden states
- Regenerating would give different results (mismatch issue)
- Forward pass on exact completion text gives correct embeddings

**Implementation**:
```python
# GRPO gives us: completion_text = "Key features: ..."
# We do:
full_prompt = f"[system]...[user]...[assistant]{completion_text}"
outputs = vlm_model(full_prompt, ...)
embeddings = outputs.hidden_states[-1]
```

### 2. Embedding Concatenation

**User's Question**: What embeddings should we use?

**Answer**: 
- **Without response**: `[system + user + instruction]` → drop system → `[user + instruction]`
- **With response**: `[system + user + instruction + assistant + completion]` → drop system → `[user + instruction + assistant + completion]`

**Why**:
- System prompt is generic, not informative for editing
- User instruction tells what to change
- Assistant completion describes how to change it
- Together they guide the diffusion model

**Implementation**:
```python
# Drop first 64 tokens (system prompt) after accounting for padding
drop_idx = num_padding_tokens + 64
embeddings = hidden_states[drop_idx:]  # Keep user + instruction + assistant + response
```

### 3. Text First, Embeddings Second

**User's Question**: Should we modify the pipeline?

**Answer**: No need. Just generate text first (GRPO does this), then compute embeddings (we do this).

**Why**:
- Simpler than modifying pipeline internals
- GRPO already handles text generation
- We just extract embeddings in reward function

### 4. Only VLM is Trained

**Critical**: 
- VLM (text_encoder) is trained with LoRA via GRPO
- DiT (transformer) stays **completely frozen**
- Pipeline's text_encoder is replaced with trained VLM

**Implementation**:
```python
# After GRPO trainer initializes:
pipe.text_encoder = trainer.model  # Use GRPO-trained VLM

# When training:
# - GRPO updates trainer.model (VLM with LoRA)
# - Pipeline uses this updated VLM
# - DiT stays frozen (no gradients)
```

## Comparison with grpo_vlm.py

### Similarities
- Same TRL setup (GRPOConfig, GRPOTrainer)
- Same argument parsing (TrlParser)
- Same LoRA config (via get_peft_config)
- Same training loop (trainer.train())
- Reward function signature: `reward_fn(completions, **kwargs) -> List[float]`

### Differences
- **Input**: VLM gets `[image + text instruction]` instead of math problem
- **Reward**: GPT-4V evaluates images instead of math verification
- **Embeddings**: We extract and pass to diffusion model
- **Output**: Edited image instead of text answer

## Files

```
trl_qwen_image_edit/
├── train_grpo.py              # Main script (327 lines, clean!)
├── deepspeed_zero2.yaml       # DeepSpeed config
├── run_8gpu.sh                # Training launcher
├── README.md                  # User-facing guide
└── IMPLEMENTATION_SUMMARY.md  # This file (technical details)
```

## Memory Budget

**Per GPU** (H200 with 141 GB):
- VLM (7B with LoRA): ~25 GB
- DiT (frozen): ~15 GB
- Diffusion pipeline (VAE, etc.): ~20 GB
- Activations & gradients: ~15 GB
- **Total**: ~75 GB
- **Headroom**: 66 GB ✅

## Verification

**Algorithmic correctness**:
- ✅ VLM generates completion text
- ✅ Forward pass extracts embeddings from that text
- ✅ System prompt dropped, response included
- ✅ Embeddings guide diffusion
- ✅ Only VLM updated, DiT frozen

**No mismatch issues**:
- ✅ We use GRPO's exact completion text
- ✅ No regeneration (was causing mismatches)
- ✅ Forward pass is deterministic

**Clean code**:
- ✅ Follows grpo_vlm.py pattern
- ✅ No complex verification logic
- ✅ Easy to understand and debug
- ✅ Self-contained (one file)

## Next Steps

1. **Set API key**: `export OPENAI_API_KEY="..."`
2. **Activate environment**: `conda activate diffusers`
3. **Run training**: `bash run_8gpu.sh`
4. **Monitor**: `tail -f outputs/grpo_clean/training.log`

The implementation is **ready to run** on 8 H200 GPUs!

