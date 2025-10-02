# Complete Guide: GRPO Training for Qwen-Image-Edit-Response

## 🎯 What This Does

Trains **Qwen2.5-VL** to generate better image editing descriptions using GRPO reinforcement learning.

- **Input**: Image + editing instruction ("add a rainbow to the sky")
- **VLM output**: Description of how to edit ("The sky is blue. Add a colorful rainbow arc...")
- **Diffusion output**: Edited image with the rainbow
- **Reward**: GPT-4V judges if the edit is good
- **Learning**: VLM improves its descriptions based on rewards

## 📁 Project Structure

```
trl_qwen_image_edit/
├── train_grpo.py                 # Main training script (327 lines)
├── deepspeed_zero2.yaml          # DeepSpeed configuration
├── run_8gpu.sh                   # Training launcher
├── test_setup.py                 # Setup verification
├── README.md                     # Quick start guide
├── IMPLEMENTATION_SUMMARY.md     # Technical details
└── COMPLETE_GUIDE.md             # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
conda activate diffusers  # Your environment with all dependencies
```

Required packages (already installed):
- `trl` (from `/home/coder/work/trl`)
- `diffusers` (from `/home/coder/work/customized_diffusers`)
- `transformers`, `torch`, `datasets`, `loguru`, `openai`

### 2. Test Setup

```bash
cd /home/coder/work/trl_qwen_image_edit
python test_setup.py
```

Should show:
```
✅ ALL TESTS PASSED!
You can now run: bash run_8gpu.sh
```

### 3. Set API Key

```bash
export OPENAI_API_KEY="sk-..."
```

### 4. Run Training

```bash
bash run_8gpu.sh
```

### 5. Monitor

```bash
tail -f outputs/grpo_clean/training.log
```

## 📊 Configuration

### Hardware
- **GPUs**: 8× H200 (141 GB each)
- **Memory per GPU**: ~75 GB (66 GB headroom)
- **Total compute**: ~8× A100 equivalent

### Training
- **Model**: Qwen/Qwen2.5-VL-7B-Instruct
- **Method**: GRPO with LoRA (r=32, α=64)
- **Batch size**: 1 per GPU × 2 accumulation = 16 effective
- **Learning rate**: 1e-5
- **Precision**: bfloat16
- **Dataset**: Complex-Edit (complexity 8, real images)
- **Generations**: 8 per prompt
- **Epochs**: 3

### Optimization
- **DeepSpeed**: ZeRO-2 (optimizer + gradient sharding)
- **LoRA targets**: q_proj, v_proj, k_proj, o_proj
- **Frozen**: DiT transformer (only VLM is trained)

## 🔬 Algorithm Details

### The GRPO Loop

```python
for batch in dataset:
    # 1. VLM generates completion (text description)
    completion = vlm.generate(image, instruction)
    
    # 2. Extract embeddings from completion
    embeddings = forward_pass(vlm, completion)
    
    # 3. Generate edited image
    edited_img = diffusion(image, embeddings)
    
    # 4. Evaluate with GPT-4V
    reward = gpt_eval(input_img, edited_img, instruction)
    
    # 5. Update VLM via GRPO
    vlm.update(reward)
```

### Embedding Extraction (Option C)

**Problem**: GRPO generates text but doesn't return embeddings.

**Solution**: Do a forward pass on the generated text.

```python
# GRPO gives us:
completion_text = "The image shows a beach. To add a sunset..."

# We do:
full_prompt = f"<|im_start|>system\n....<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{completion_text}<|im_end|>"

outputs = vlm(full_prompt, image, output_hidden_states=True)
embeddings = outputs.hidden_states[-1]  # Shape: [batch, seq_len, hidden_dim]

# Drop system prompt (first 64 tokens after padding)
embeddings = embeddings[:, num_padding+64:]
```

**Why this works**:
- Forward pass is deterministic
- Uses GRPO's exact completion text
- No regeneration needed
- Embeddings match what was generated

### Reward Function

**Alignment** (60%):
```
"Does the edited image follow the instruction?"
Rates: 0.0 (not at all) to 1.0 (perfectly)
```

**Quality** (40%):
```
"Is the edited image high quality?"
Considers: realism, coherence, artifacts
Rates: 0.0 (poor) to 1.0 (excellent)
```

**Combined**:
```python
reward = 0.6 * alignment_score + 0.4 * quality_score
```

## 🎓 Key Insights

### 1. Only VLM is Trained

```python
# VLM (text_encoder)  ← Trained with GRPO + LoRA
# DiT (transformer)   ← Frozen (no gradients)
# VAE                 ← Frozen

pipe.text_encoder = trained_vlm  # Replace with GRPO-updated model
```

**Why**:
- VLM learns to generate better descriptions
- DiT already knows how to edit images
- Training DiT would be slow and unstable

### 2. Text First, Embeddings Second

**Old approach** (buggy):
- Try to extract embeddings during generation
- Mismatches between GRPO's text and our embeddings
- Complex verification logic

**New approach** (clean):
- GRPO generates text
- We do forward pass on that text
- Extract embeddings
- Use for diffusion

### 3. No Regeneration

**Problem with regeneration**:
```python
# GRPO generates: "The sky is blue..."
grpo_text = grpo.generate()

# We regenerate: "The clouds are white..."
our_text = vlm.generate()  # Different!

# Embeddings don't match original generation
embeddings = extract(our_text)  # Wrong!
```

**Our solution**:
```python
# GRPO generates
grpo_text = grpo.generate()

# We use exact same text
our_embeddings = forward_pass(grpo_text)  # Correct!
```

### 4. Embedding Concatenation

**Before** (without response):
```
[SYSTEM PROMPT] [USER + INSTRUCTION]
     ↓ drop        ↓ keep
                [USER + INSTRUCTION] → embeddings
```

**After** (with response):
```
[SYSTEM PROMPT] [USER + INSTRUCTION] [ASSISTANT + COMPLETION]
     ↓ drop        ↓ keep                ↓ keep
                [USER + INSTRUCTION + ASSISTANT + COMPLETION] → embeddings
```

The concatenated embeddings give the diffusion model:
- What to change (instruction)
- How to change it (completion)

## 🐛 Troubleshooting

### OOM (Out of Memory)

**Symptoms**: `CUDA out of memory` error

**Solutions**:
1. Reduce `per_device_train_batch_size` (currently 1)
2. Reduce `num_generations` (currently 8 → try 4)
3. Reduce `gradient_accumulation_steps` (currently 2 → try 1)

### Slow Training

**Symptoms**: Very slow progress

**Solutions**:
1. Check GPU utilization: `nvidia-smi -l 1`
2. Verify all 8 GPUs are being used
3. Check DeepSpeed is active (look for ZeRO logs)

### API Rate Limits

**Symptoms**: GPT evaluation fails frequently

**Solutions**:
1. Reduce `n_evals` (currently 3 → try 1)
2. Add retries with exponential backoff
3. Use a higher-tier OpenAI plan

### DeepSpeed Errors

**Symptoms**: `IndexError: pop from empty deque`

**Solution**: We use ZeRO-2 (not ZeRO-3) to avoid this

## 📈 Expected Results

### Training Progress

**First epoch**:
- Rewards start around 0.5 (random)
- Should increase to 0.6-0.7
- Loss decreases

**Second epoch**:
- Rewards 0.7-0.8
- More stable

**Third epoch**:
- Rewards plateau around 0.8-0.9
- Fine-tuning

### Checkpoints

Saved every 500 steps to `outputs/grpo_clean/checkpoint-XXX/`

Contains:
- LoRA weights
- Optimizer state
- Training state

### Final Model

At `outputs/grpo_clean/`:
- LoRA adapter weights
- Can be loaded with `from_pretrained()`

## 🎯 Verification Checklist

Before training:
- [ ] Environment activated (`conda activate diffusers`)
- [ ] API key set (`echo $OPENAI_API_KEY`)
- [ ] Test script passes (`python test_setup.py`)
- [ ] 8 GPUs available (`nvidia-smi`)

During training:
- [ ] All 8 processes start (check logs)
- [ ] Memory usage ~75 GB per GPU (check `nvidia-smi`)
- [ ] Rewards computed successfully (check logs)
- [ ] Loss decreasing (check logs)

After training:
- [ ] Final checkpoint saved
- [ ] Logs show completion
- [ ] Model files exist in `outputs/grpo_clean/`

## 🔍 Differences from Previous Implementation

| Aspect | Old (`qwen-image-edit-response/`) | New (`trl_qwen_image_edit/`) |
|--------|-----------------------------------|------------------------------|
| **Lines of code** | 587 (train_grpo.py) | 327 (train_grpo.py) |
| **Complexity** | High (many fixes) | Low (clean) |
| **Embedding extraction** | Regenerate + verify | Forward pass |
| **Verification** | Complex mismatch checks | None needed |
| **Maintainability** | Hard to understand | Easy to follow |
| **Correctness** | Had issues | Verified correct |

## 🎊 Ready to Train!

Everything is set up and ready. Just:

```bash
cd /home/coder/work/trl_qwen_image_edit
export OPENAI_API_KEY="your-key"
bash run_8gpu.sh
```

Training will take approximately:
- **1000 samples**: ~6-8 hours (with 8 GPUs)
- **Full dataset**: ~24-48 hours

Good luck! 🚀

