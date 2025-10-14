# GRPO Training for Qwen Vision Models

Clean, standalone implementations for training Qwen2.5-VL with GRPO:
- **Image Editing**: Train on Complex-Edit dataset (`train_grpo_qwen_image_edit.py`)
- **Text-to-Image**: Train on MMMG dataset (`train_grpo_qwen_image.py`)

## Pipeline

### Image Editing (Qwen-Image-Edit)
1. **GRPO generates VLM completions** (text descriptions)
2. **Extract embeddings** from completion text via forward pass
3. **Generate edited images** using embeddings in diffusion model
4. **Evaluate** with GPT-4V (alignment + quality)
5. **Update VLM** via GRPO (DiT stays frozen)

### Text-to-Image (Qwen-Image)
1. **GRPO generates VLM completions** (text descriptions)
2. **Extract embeddings** from completion text via forward pass
3. **Generate images** using embeddings in diffusion model
4. **Evaluate** with GPT-4V using **MMMG protocol** (knowledge graph evaluation)
5. **Update VLM** via GRPO (DiT stays frozen)

**Note**: Uses official MMMG evaluation protocol that evaluates if elements and dependencies from knowledge graphs are correctly visualized.

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

**For Image Editing (Complex-Edit dataset):**
```bash
bash run_8gpu.sh
# or for full fine-tuning:
bash run_8gpu_full_finetune.sh
```

**For Text-to-Image (MMMG dataset):**
```bash
bash run_8gpu_qwen_image.sh
```

### 3. Monitor

```bash
# For image editing:
tail -f outputs/drgrpo_qwenie/training.log

# For text-to-image:
tail -f outputs/grpo_qwen_image/training.log
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
├── train_grpo_qwen_image_edit.py  # Image editing training (Complex-Edit)
├── train_grpo_qwen_image.py       # Text-to-image training (MMMG)
├── deepspeed_zero2.yaml           # DeepSpeed config
├── deepspeed_zero3.yaml           # DeepSpeed ZeRO-3 config
├── run_8gpu.sh                    # Image editing launcher (LoRA)
├── run_8gpu_full_finetune.sh      # Image editing launcher (full)
├── run_8gpu_qwen_image.sh         # Text-to-image launcher
├── monitor_and_run.sh             # Monitoring helper
└── README.md                      # This file
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

## Dataset Details

### Complex-Edit (Image Editing)
- Multi-complexity image editing dataset
- Complexity levels 1-8 (atomic to compound instructions)
- Two types: `real` and `syn` (synthetic)
- Contains input images + edit instructions

### MMMG (Text-to-Image)
- Multi-modal multi-granularity generation benchmark
- Education levels: preschool, primaryschool, secondaryschool, highschool, undergraduate, PhD
- Disciplines: Biology, Chemistry, Economics, Engineering, Geography, History, Literature, Math, Philosophy, Sociology
- Contains text prompts for image generation
- **Includes knowledge graphs** with elements and dependencies for evaluation
- **Evaluation Protocol**: GPT-4V judges if elements/dependencies are correctly visualized (yes/no for each)

## Notes

- The pipeline internally calls `text_encoder.generate()` during training (GRPO handles this)
- Our reward function gets the completion text and does a forward pass
- No need to verify matches - we use GRPO's exact completions
- Much simpler than the previous approach!
- For text-to-image, no input image is needed - pure text-to-image generation

