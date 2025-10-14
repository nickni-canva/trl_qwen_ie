"""
Test to ensure embedding extraction is consistent between 
train_grpo_qwen_image.py and pipeline_qwenimage_response.py
"""

import torch
import sys
sys.path.insert(0, '/home/coder/work/customized_diffusers/src')

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from diffusers.pipelines.qwenimage.pipeline_qwenimage_response import QwenImageResponsePipeline

# Initialize model and tokenizer
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"Loading model: {model_id}")

tokenizer = Qwen2Tokenizer.from_pretrained(model_id)
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Use CPU for testing
)

print("Model loaded successfully!\n")

# Test prompts
test_prompts = [
    "A red cat sitting on a blue chair",
    "Generate an educational diagram showing the water cycle with labeled arrows and text",
]

print("="*80)
print("Testing embedding extraction consistency")
print("="*80)

# ============================================================================
# METHOD 1: Pipeline approach (from pipeline_qwenimage_response.py)
# ============================================================================
print("\n📦 METHOD 1: Pipeline Implementation")
print("-" * 80)

prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
prompt_template_encode_start_idx = 34

def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    """From pipeline"""
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
    return split_result

# Format prompts
txt = [prompt_template_encode.format(p) for p in test_prompts]
print(f"Formatted prompt 1: {txt[0][:100]}...")
print(f"Formatted prompt 2: {txt[1][:100]}...")

# Tokenize
txt_tokens = tokenizer(
    txt, 
    max_length=1024 + prompt_template_encode_start_idx, 
    padding=True, 
    truncation=True, 
    return_tensors="pt"
)

print(f"\nTokenized shapes:")
print(f"  input_ids: {txt_tokens.input_ids.shape}")
print(f"  attention_mask: {txt_tokens.attention_mask.shape}")
print(f"  Drop index: {prompt_template_encode_start_idx}")

# Get hidden states
with torch.no_grad():
    outputs = vlm_model(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )

hidden_states = outputs.hidden_states[-1]
print(f"  hidden_states: {hidden_states.shape}")

# Extract using pipeline method
split_hidden_states = _extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
print(f"\nAfter _extract_masked_hidden:")
for i, h in enumerate(split_hidden_states):
    print(f"  Sequence {i}: {h.shape}")

# Drop tokens uniformly
drop_idx = prompt_template_encode_start_idx
split_hidden_states_dropped = [e[drop_idx:] for e in split_hidden_states]
print(f"\nAfter dropping {drop_idx} tokens:")
for i, h in enumerate(split_hidden_states_dropped):
    print(f"  Sequence {i}: {h.shape}")

# Pad to max length
attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states_dropped]
max_seq_len = max([e.size(0) for e in split_hidden_states_dropped])
prompt_embeds_pipeline = torch.stack(
    [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states_dropped]
)
prompt_masks_pipeline = torch.stack(
    [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
)

print(f"\nFinal pipeline embeddings: {prompt_embeds_pipeline.shape}")
print(f"Final pipeline masks: {prompt_masks_pipeline.shape}")
print(f"Pipeline mask sum per sequence: {prompt_masks_pipeline.sum(dim=1).tolist()}")

# ============================================================================
# METHOD 2: Training script approach (from train_grpo_qwen_image.py)
# ============================================================================
print("\n\n🔧 METHOD 2: Training Script Implementation")
print("-" * 80)

# Use the same system template
SYSTEM_TEMPLATE = "You are assisting an image generation model. Given the user's text, your task is to plan and describe what should be presented in the output image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"

# Format prompts
formatted_prompts = []
for p in test_prompts:
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": p},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted_prompts.append(text)

print(f"Formatted prompt 1: {formatted_prompts[0][:100]}...")
print(f"Formatted prompt 2: {formatted_prompts[1][:100]}...")

# Tokenize
inputs = tokenizer(
    formatted_prompts,
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors="pt",
)

print(f"\nTokenized shapes:")
print(f"  input_ids: {inputs.input_ids.shape}")
print(f"  attention_mask: {inputs.attention_mask.shape}")

# Get hidden states
with torch.no_grad():
    outputs2 = vlm_model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
    )

hidden_states2 = outputs2.hidden_states[-1]
print(f"  hidden_states: {hidden_states2.shape}")

# Extract using same method
split_hidden_states2 = _extract_masked_hidden(hidden_states2, inputs.attention_mask)
print(f"\nAfter _extract_masked_hidden:")
for i, h in enumerate(split_hidden_states2):
    print(f"  Sequence {i}: {h.shape}")

# Drop with per-sequence padding calculation (OLD APPROACH)
processed_hidden_states_old = []
for i, hidden in enumerate(split_hidden_states2):
    mask = inputs.attention_mask[i]
    num_padding = (mask == 0).sum().item()
    drop_idx_old = num_padding + 64
    
    if drop_idx_old < len(hidden):
        processed_hidden_states_old.append(hidden[drop_idx_old:])
    else:
        print(f"  WARNING: Sequence {i} - drop_idx {drop_idx_old} >= length {len(hidden)}")
        processed_hidden_states_old.append(hidden)

print(f"\nAfter OLD approach (padding + 64 per sequence):")
for i, h in enumerate(processed_hidden_states_old):
    print(f"  Sequence {i}: {h.shape}")

# Drop uniformly (NEW APPROACH - matching pipeline)
drop_idx_uniform = 64  # System prompt tokens
split_hidden_states_dropped2 = [e[drop_idx_uniform:] for e in split_hidden_states2]
print(f"\nAfter NEW approach (uniform drop of {drop_idx_uniform}):")
for i, h in enumerate(split_hidden_states_dropped2):
    print(f"  Sequence {i}: {h.shape}")

# Pad to max length (NEW)
max_len_new = max(h.size(0) for h in split_hidden_states_dropped2)
prompt_embeds_training_new = torch.stack([
    torch.cat([h, h.new_zeros(max_len_new - h.size(0), h.size(1))])
    for h in split_hidden_states_dropped2
])
prompt_masks_training_new = torch.stack([
    torch.cat([
        torch.ones(h.size(0), dtype=torch.long, device=h.device),
        torch.zeros(max_len_new - h.size(0), dtype=torch.long, device=h.device)
    ])
    for h in split_hidden_states_dropped2
])

print(f"\nFinal training embeddings (NEW): {prompt_embeds_training_new.shape}")
print(f"Final training masks (NEW): {prompt_masks_training_new.shape}")
print(f"Training mask sum per sequence (NEW): {prompt_masks_training_new.sum(dim=1).tolist()}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n\n" + "="*80)
print("COMPARISON & RECOMMENDATIONS")
print("="*80)

print("\n📊 Pipeline vs Training (NEW approach):")
print(f"  Pipeline embeddings shape: {prompt_embeds_pipeline.shape}")
print(f"  Training embeddings shape (NEW): {prompt_embeds_training_new.shape}")
print(f"  Pipeline masks shape: {prompt_masks_pipeline.shape}")
print(f"  Training masks shape (NEW): {prompt_masks_training_new.shape}")

print("\n🔍 Key Insight:")
print("  The pipeline uses a FIXED template with a known drop_idx.")
print("  The training script uses chat templates which may vary.")
print("  Both should use _extract_masked_hidden() then drop a FIXED number of tokens.")

print("\n✅ RECOMMENDATION:")
print("  1. Both implementations should extract masked hidden states first")
print("  2. Then drop a FIXED number of system prompt tokens uniformly")
print("  3. Do NOT calculate per-sequence padding offsets - padding is already handled")
print("  4. The drop_idx should correspond to the system prompt token count")

print("\n📝 System prompt analysis:")
system_tokens = tokenizer.encode(SYSTEM_TEMPLATE, add_special_tokens=False)
print(f"  System template tokens: {len(system_tokens)}")
print(f"  With special tokens (estimated): ~{len(system_tokens) + 10}")
print(f"  Pipeline uses drop_idx=34 (for its specific template)")
print(f"  Training should measure its own system prompt length")

# Calculate actual system prompt length for training
test_system_formatted = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_TEMPLATE}],
    tokenize=True,
    add_generation_prompt=False,
)
print(f"  Actual system prompt tokens in training: {len(test_system_formatted)}")

print("\n" + "="*80)
print("✅ Test complete! Check the shapes and token counts above.")
print("="*80)

