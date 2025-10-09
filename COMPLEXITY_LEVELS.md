# Complexity Levels in Complex-Edit Dataset

This document explains how to configure complexity levels for training with the Complex-Edit dataset.

---

## 📊 **What are Complexity Levels?**

The **Complex-Edit** dataset contains images with editing instructions at **8 different complexity levels** (1-8):

- **Level 1**: Simple single-object edits (e.g., "Change the color of the car to red")
- **Level 2-3**: Simple multi-object edits
- **Level 4-5**: Moderate complexity with spatial relationships
- **Level 6-7**: Complex edits with multiple constraints
- **Level 8**: Most complex instructions with many atomic edits combined

Each image in the dataset has **8 compound instructions** (one per complexity level), stored in `ex["edit"]["compound"][0-7]`.

---

## ⚙️ **Configuring Complexity Levels**

You can specify which complexity levels to use during training with the `--complexity` argument:

### **1. Train on All Levels (Default)**
```bash
--complexity "1-8"
```
- Loads **all 8 complexity levels**
- Each image is expanded to **8 training samples** (one per level)
- **Dataset size = original_size × 8**
- Best for **general-purpose** editing models

### **2. Train on Specific Range**
```bash
# Only simple to moderate edits
--complexity "1-4"

# Only complex edits
--complexity "6-8"

# Medium complexity
--complexity "3-6"
```

### **3. Train on Specific Levels**
```bash
# Only levels 1, 4, and 8
--complexity "1,4,8"

# Only extreme ends
--complexity "1,8"
```

### **4. Train on Single Level**
```bash
# Only the most complex instructions
--complexity "8"

# Only simple instructions
--complexity "1"
```

---

## 📈 **Dataset Size Impact**

The complexity setting directly affects your dataset size:

| Complexity | Original Images | Training Samples | Expansion |
|------------|----------------|------------------|-----------|
| `"8"` | 532 | 532 | 1x |
| `"1,8"` | 532 | 1,064 | 2x |
| `"1-4"` | 532 | 2,128 | 4x |
| `"1-8"` | 532 | 4,256 | 8x |

**Example with `max_samples=1000`:**
- `--complexity "8"`: 1000 samples (limited by max_samples)
- `--complexity "1-8"`: 1000 samples (limited by max_samples, not 4256)

---

## 💡 **Training Strategies**

### **Strategy 1: Curriculum Learning**
Train progressively on increasing complexity:

```bash
# Stage 1: Simple edits (warm-up)
--complexity "1-3" --max_steps 300

# Stage 2: Medium complexity
--complexity "4-6" --max_steps 400

# Stage 3: All levels
--complexity "1-8" --max_steps 1000
```

### **Strategy 2: Difficulty-Specific Models**
Train separate models for different use cases:

```bash
# Model for simple edits (faster inference)
--complexity "1-3"

# Model for complex creative edits
--complexity "6-8"
```

### **Strategy 3: Balanced Sampling**
Use all levels but with balanced sampling:

```bash
# Default: all levels
--complexity "1-8"
```

---

## 📂 **Dataset Structure**

Each training sample includes:
```python
{
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Change the sky to sunset and add birds"}
    ],
    "image": PIL.Image,
    "instruction": "Change the sky to sunset and add birds",
    "complexity_level": 5  # Which level this sample came from
}
```

The `complexity_level` field is included for tracking and analysis.

---

## 🎯 **Recommended Settings**

### **For General-Purpose Training:**
```bash
--complexity "1-8"
--max_samples 4000  # To get good coverage across all levels
```

### **For Fast Prototyping:**
```bash
--complexity "8"  # Only hardest examples
--max_samples 500
```

### **For Production Models:**
```bash
--complexity "1-8"
--max_samples -1  # Use entire dataset
```

### **For Research on Specific Difficulty:**
```bash
--complexity "4,5,6"  # Mid-range complexity
```

---

## 📊 **Monitoring Complexity Distribution**

During training, you can track which complexity levels are being processed:
```bash
# Check the training logs
grep "complexity_level" outputs/drgrpo_qwenie/training.log
```

---

## 🔍 **Example Instructions by Complexity**

### **Level 1 (Simple):**
> "Change the color of the car to blue"

### **Level 3 (Moderate):**
> "Change the color of the car to blue and add a tree in the background"

### **Level 5 (Complex):**
> "Change the color of the car to blue, add a tree in the background, make it sunset, and add reflections on the car"

### **Level 8 (Most Complex):**
> "Change the color of the car to blue, add a tree in the background, make it sunset, add reflections on the car, change the road to cobblestone, add street lamps, include shadows, and adjust lighting to golden hour"

---

## ⚠️ **Important Notes**

1. **Memory Usage**: More complexity levels = larger dataset = more memory
   - With `"1-8"`, expect 8x more samples than with `"8"`
   - Use `--max_samples` to limit dataset size if needed

2. **Training Time**: More samples = longer training
   - Consider adjusting `--max_steps` proportionally
   - Or use `--num_train_epochs` to automatically scale

3. **Batch Size**: With larger datasets, you may need to adjust:
   ```bash
   --per_device_train_batch_size 1  # If OOM
   --gradient_accumulation_steps 4  # To compensate
   ```

4. **Evaluation**: The eval set will also be expanded by the same factor
   - Consider separate complexity settings for train vs eval if needed

---

## 🚀 **Quick Start Examples**

### **Train on all complexity levels (recommended):**
```bash
bash run_8gpu.sh  # Default is --complexity "1-8"
```

### **Train only on complex edits:**
```bash
# Modify run_8gpu.sh:
--complexity "6-8" \
```

### **Train on specific levels:**
```bash
# Modify run_8gpu.sh:
--complexity "1,3,5,7" \
```

