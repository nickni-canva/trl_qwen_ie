# Checkpoint Management

This document explains how checkpoints are saved and managed during GRPO training.

## 📦 **Checkpoint Configuration**

### **LoRA Training** (`run_8gpu.sh`)

```bash
--save_strategy steps          # Save based on training steps (not epochs)
--save_steps 100               # Save every 100 steps
--save_total_limit 10          # Keep last 10 checkpoints (auto-delete older ones)
--save_only_model True         # Only save model weights (not optimizer states)
```

**Checkpoint frequency:**
- Total steps: 1000
- Saves at: step 100, 200, 300, ..., 1000
- **10 checkpoints total**
- Disk space: ~3-5 GB per checkpoint (LoRA adapters only)

### **Full Fine-Tuning** (`run_8gpu_full_finetune.sh`)

```bash
--save_strategy steps          # Save based on training steps
--save_steps 50                # Save every 50 steps (more frequent!)
--save_total_limit 20          # Keep last 20 checkpoints
--save_only_model True         # Only save model weights
```

**Checkpoint frequency:**
- Total steps: 1000
- Saves at: step 50, 100, 150, ..., 1000
- **20 checkpoints total**
- Disk space: ~15-20 GB per checkpoint (full model)

---

## 📂 **Checkpoint Directory Structure**

### LoRA Training:
```
outputs/drgrpo_qwenie/
├── checkpoint-100/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── trainer_state.json
├── checkpoint-200/
│   └── ...
├── checkpoint-1000/  # Final checkpoint
│   └── ...
└── training.log
```

### Full Fine-Tuning:
```
outputs/grpo_full_finetune/
├── checkpoint-50/
│   ├── config.json
│   ├── model.safetensors
│   ├── generation_config.json
│   └── trainer_state.json
├── checkpoint-100/
│   └── ...
├── checkpoint-1000/  # Final checkpoint
│   └── ...
└── training.log
```

---

## 🔄 **Checkpoint Rotation**

Both scripts use `--save_total_limit` to automatically manage disk space:

- **Older checkpoints are automatically deleted** when the limit is reached
- Only the **most recent N checkpoints** are kept
- The **final checkpoint is always saved**

**Example (LoRA with limit=10):**
```
Step 100 → Save checkpoint-100
Step 200 → Save checkpoint-200
...
Step 1000 → Save checkpoint-1000

Step 1100 → Save checkpoint-1100, DELETE checkpoint-100
Step 1200 → Save checkpoint-1200, DELETE checkpoint-200
```

---

## 💾 **Manual Checkpoint Saving**

To save a checkpoint at any point, send `SIGUSR1` signal:
```bash
# Find the training process
ps aux | grep train_grpo.py

# Send save signal
kill -SIGUSR1 <PID>
```

---

## 🔍 **Checkpoint Contents**

### LoRA Checkpoints (smaller):
- `adapter_model.safetensors` - LoRA adapter weights (~500MB)
- `adapter_config.json` - LoRA configuration
- `trainer_state.json` - Training state (step, epoch, best metric)

### Full Model Checkpoints (larger):
- `model.safetensors` - Full model weights (~15GB for 7B model)
- `config.json` - Model configuration
- `generation_config.json` - Generation settings
- `trainer_state.json` - Training state

---

## 🚀 **Loading Checkpoints**

### Resume Training:
```bash
# Add to your run script:
--resume_from_checkpoint outputs/drgrpo_qwenie/checkpoint-500
```

### Load for Inference (LoRA):
```python
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)
model = PeftModel.from_pretrained(
    base_model,
    "outputs/drgrpo_qwenie/checkpoint-1000"
)
```

### Load for Inference (Full):
```python
from transformers import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "outputs/grpo_full_finetune/checkpoint-1000"
)
```

---

## ⚠️ **Important Notes**

1. **Disk Space Requirements:**
   - LoRA: ~50 GB for 10 checkpoints
   - Full: ~300-400 GB for 20 checkpoints

2. **`save_only_model=True`** means:
   - ✅ Saves model weights
   - ❌ Does NOT save optimizer states
   - ❌ Does NOT save gradient states
   - Result: **Checkpoints are smaller, but you cannot resume training from them with the exact optimizer state**

3. **To enable full resumption** (including optimizer states):
   - Remove `--save_only_model True`
   - Checkpoints will be **3-4x larger** but training can be resumed exactly

4. **DeepSpeed Checkpoints:**
   - With DeepSpeed ZeRO-2/3, checkpoints may be sharded across GPUs
   - Use `--save_only_model True` to consolidate into a single file

---

## 📊 **Best Practices**

1. **Monitor disk space:**
   ```bash
   watch -n 60 "du -sh outputs/*"
   ```

2. **Keep important checkpoints:**
   ```bash
   # Copy best checkpoint to permanent location
   cp -r outputs/drgrpo_qwenie/checkpoint-500 saved_models/best_model
   ```

3. **Cleanup old experiments:**
   ```bash
   # Remove old checkpoints manually if needed
   rm -rf outputs/old_experiment/checkpoint-*
   ```

---

## 🎯 **Checkpoint Verification**

To verify a checkpoint is valid:
```bash
# Check LoRA checkpoint
python -c "
from peft import PeftModel
from transformers import AutoModel
model = AutoModel.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
model = PeftModel.from_pretrained(model, 'outputs/drgrpo_qwenie/checkpoint-100')
print('✅ Checkpoint valid!')
"

# Check full checkpoint
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('outputs/grpo_full_finetune/checkpoint-100')
print('✅ Checkpoint valid!')
"
```

