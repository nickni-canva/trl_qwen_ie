# GRPO Training for Qwen-Image-Edit-Response

Clean, standalone implementation for training Qwen2.5-VL with GRPO for image editing.

## Pipeline

1. **GRPO generates VLM completions** (text descriptions)
2. **Extract embeddings** from completion text via forward pass
3. **Generate edited images** using embeddings in diffusion model
4. **Evaluate** with GPT-4V (alignment + quality)
5. **Update VLM** via GRPO (DiT stays frozen)

## Key Design Decisions

### Embedding Extraction (Option C)
- GRPO generates completion text
- We do a **forward pass** on the completion text to get embeddings
- No regeneration needed - uses exact GRPO completions

### Embedding Concatenation
- Original: `[system + user + instruction]` → drop system → `[user + instruction]`
- With response: `[system + user + instruction + assistant + completion]` → drop system → `[user + instruction + assistant + completion]`
- These concatenated embeddings guide the diffusion model

### Model Updates
- **Only VLM** (text_encoder) is trained via GRPO + LoRA
- **DiT** (transformer) stays frozen
- Pipeline's `text_encoder` is replaced with the GRPO-trained VLM

## Requirements

```bash
conda activate diffusers  # Your environment with:
# - trl (from /home/coder/work/trl)
# - diffusers (from /home/coder/work/customized_diffusers)
# - transformers, torch, etc.
```

## Usage

### 1. Set API Key

```bash
export OPENAI_API_KEY="your-openai-key"
```

### 2. Run Training

```bash
bash run_8gpu.sh
```

### 3. Monitor

```bash
tail -f outputs/grpo_clean/training.log
```

## Configuration

**Hardware**: 8× H200 GPUs (141 GB each)

**Memory per GPU**:
- VLM (7B with LoRA): ~25 GB
- Diffusion pipeline: ~35 GB
- Total: ~60 GB (plenty of headroom)

**Training**:
- Batch size: 1 per GPU
- Gradient accumulation: 2
- Effective batch: 16
- Generations per prompt: 8
- Learning rate: 1e-5
- Precision: bfloat16

## Files

```
trl_qwen_image_edit/
├── train_grpo.py          # Main training script
├── deepspeed_zero2.yaml   # DeepSpeed config
├── run_8gpu.sh            # Training launcher
└── README.md              # This file
```

## Algorithm Correctness

✅ **Verified**:
- VLM completions are used (no mismatch)
- Embeddings extracted from completion text (Option C)
- System prompt dropped, response embeddings concatenated
- Only VLM updated, DiT frozen
- Follows grpo_vlm.py pattern

## Differences from Previous Implementation

**Old** (qwen-image-edit-response/):
- Complex regeneration logic
- Verification code to check mismatches
- Multiple attempts to fix padding issues
- Messy with many band-aid fixes

**New** (trl_qwen_image_edit/):
- Simple forward pass on completion text
- No regeneration needed
- Clean, follows grpo_vlm.py closely
- Easy to understand and debug

## Notes

- The pipeline internally calls `text_encoder.generate()` during training (GRPO handles this)
- Our reward function gets the completion text and does a forward pass
- No need to verify matches - we use GRPO's exact completions
- Much simpler than the previous approach!

