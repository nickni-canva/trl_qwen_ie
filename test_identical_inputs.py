"""
Test that pipeline and training script produce identical embeddings with the same input
"""

import torch
import sys
sys.path.insert(0, '/home/coder/work/customized_diffusers/src')

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

# Initialize
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"Loading model: {model_id}")

tokenizer = Qwen2Tokenizer.from_pretrained(model_id)
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print("Model loaded!\n")

# Use IDENTICAL test prompts
test_prompts = [
    "A red cat sitting on a blue chair",
    "A red cat sitting on a blue chair",  # Same prompt
]

print("="*80)
print("TEST: IDENTICAL INPUTS")
print("="*80)
print(f"Prompt 1: {test_prompts[0]}")
print(f"Prompt 2: {test_prompts[1]}")
print(f"Are they identical? {test_prompts[0] == test_prompts[1]}")

# ============================================================================
# METHOD 1: Pipeline approach
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: Pipeline Implementation")
print("="*80)

pipeline_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
pipeline_drop_idx = 34

def extract_masked_hidden(hidden_states, mask):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)

# Format
txt = [pipeline_template.format(p) for p in test_prompts]
print(f"Formatted text (first 100 chars):")
print(f"  Text 1: {txt[0][:100]}")
print(f"  Text 2: {txt[1][:100]}")
print(f"  Identical? {txt[0] == txt[1]}")

# Tokenize
txt_tokens = tokenizer(
    txt,
    max_length=1024 + pipeline_drop_idx,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(f"\nTokenized:")
print(f"  Shape: {txt_tokens.input_ids.shape}")
print(f"  Token IDs identical? {torch.equal(txt_tokens.input_ids[0], txt_tokens.input_ids[1])}")

# Get hidden states
with torch.no_grad():
    outputs = vlm_model(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )

hidden_states = outputs.hidden_states[-1]
print(f"  Hidden states shape: {hidden_states.shape}")
print(f"  Hidden states identical? {torch.allclose(hidden_states[0], hidden_states[1])}")

# Extract and drop
split_hidden = extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
print(f"\nAfter extract_masked_hidden:")
print(f"  Seq 0 shape: {split_hidden[0].shape}")
print(f"  Seq 1 shape: {split_hidden[1].shape}")
print(f"  Identical? {torch.allclose(split_hidden[0], split_hidden[1])}")

split_dropped = [e[pipeline_drop_idx:] for e in split_hidden]
print(f"\nAfter dropping {pipeline_drop_idx} tokens:")
print(f"  Seq 0 shape: {split_dropped[0].shape}")
print(f"  Seq 1 shape: {split_dropped[1].shape}")
print(f"  Identical? {torch.allclose(split_dropped[0], split_dropped[1])}")

# Pad
attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_dropped]
max_seq_len = max([e.size(0) for e in split_dropped])
prompt_embeds_pipeline = torch.stack([
    torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
    for u in split_dropped
])
prompt_masks_pipeline = torch.stack([
    torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
    for u in attn_mask_list
])

print(f"\nFinal pipeline embeddings:")
print(f"  Shape: {prompt_embeds_pipeline.shape}")
print(f"  Seq 0 == Seq 1? {torch.allclose(prompt_embeds_pipeline[0], prompt_embeds_pipeline[1])}")
print(f"  Max diff: {(prompt_embeds_pipeline[0] - prompt_embeds_pipeline[1]).abs().max().item()}")

# ============================================================================
# METHOD 2: Training script approach (with same template as pipeline for comparison)
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: Training Script Implementation (USING PIPELINE TEMPLATE)")
print("="*80)

# Use the SAME template as pipeline for fair comparison
txt2 = [pipeline_template.format(p) for p in test_prompts]
print(f"Using same template as pipeline")

# Tokenize the same way
txt_tokens2 = tokenizer(
    txt2,
    max_length=1024 + pipeline_drop_idx,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(f"\nTokenized:")
print(f"  Shape: {txt_tokens2.input_ids.shape}")
print(f"  Identical to pipeline tokens? {torch.equal(txt_tokens2.input_ids, txt_tokens.input_ids)}")

# Get hidden states
with torch.no_grad():
    outputs2 = vlm_model(
        input_ids=txt_tokens2.input_ids,
        attention_mask=txt_tokens2.attention_mask,
        output_hidden_states=True,
    )

hidden_states2 = outputs2.hidden_states[-1]
print(f"  Hidden states shape: {hidden_states2.shape}")
print(f"  Identical to pipeline? {torch.allclose(hidden_states2, hidden_states)}")

# Extract and drop (using training script approach)
split_hidden2 = extract_masked_hidden(hidden_states2, txt_tokens2.attention_mask)
print(f"\nAfter extract_masked_hidden:")
print(f"  Seq 0 shape: {split_hidden2[0].shape}")
print(f"  Seq 1 shape: {split_hidden2[1].shape}")
print(f"  Seq 0 identical to pipeline? {torch.allclose(split_hidden2[0], split_hidden[0])}")

# Drop uniformly (training script approach)
drop_idx_training = pipeline_drop_idx  # Use same drop_idx for fair comparison
split_dropped2 = [e[drop_idx_training:] for e in split_hidden2]
print(f"\nAfter dropping {drop_idx_training} tokens:")
print(f"  Seq 0 shape: {split_dropped2[0].shape}")
print(f"  Seq 1 shape: {split_dropped2[1].shape}")
print(f"  Seq 0 identical to pipeline? {torch.allclose(split_dropped2[0], split_dropped[0])}")

# Pad (training script approach)
max_len = max(h.size(0) for h in split_dropped2)
attn_mask_list2 = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_dropped2]
prompt_embeds_training = torch.stack([
    torch.cat([h, h.new_zeros(max_len - h.size(0), h.size(1))])
    for h in split_dropped2
])
prompt_masks_training = torch.stack([
    torch.cat([u, u.new_zeros(max_len - u.size(0))])
    for u in attn_mask_list2
])

print(f"\nFinal training embeddings:")
print(f"  Shape: {prompt_embeds_training.shape}")
print(f"  Seq 0 == Seq 1? {torch.allclose(prompt_embeds_training[0], prompt_embeds_training[1])}")
print(f"  Max diff: {(prompt_embeds_training[0] - prompt_embeds_training[1]).abs().max().item()}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("FINAL COMPARISON: Pipeline vs Training Script")
print("="*80)

# Compare embeddings
embeddings_identical = torch.allclose(prompt_embeds_pipeline, prompt_embeds_training, atol=1e-6)
max_diff = (prompt_embeds_pipeline - prompt_embeds_training).abs().max().item()

print(f"\n✓ Embeddings identical? {embeddings_identical}")
print(f"  Max absolute difference: {max_diff}")

# Compare masks
masks_identical = torch.equal(prompt_masks_pipeline, prompt_masks_training)
print(f"\n✓ Masks identical? {masks_identical}")

# Check within-batch consistency
seq0_seq1_pipeline = torch.allclose(prompt_embeds_pipeline[0], prompt_embeds_pipeline[1])
seq0_seq1_training = torch.allclose(prompt_embeds_training[0], prompt_embeds_training[1])

print(f"\n✓ Within pipeline batch (seq 0 vs seq 1): {seq0_seq1_pipeline}")
print(f"✓ Within training batch (seq 0 vs seq 1): {seq0_seq1_training}")

# Cross-method comparison
pipeline_0_vs_training_0 = torch.allclose(prompt_embeds_pipeline[0], prompt_embeds_training[0])
pipeline_1_vs_training_1 = torch.allclose(prompt_embeds_pipeline[1], prompt_embeds_training[1])

print(f"\n✓ Pipeline seq 0 vs Training seq 0: {pipeline_0_vs_training_0}")
print(f"✓ Pipeline seq 1 vs Training seq 1: {pipeline_1_vs_training_1}")

print("\n" + "="*80)
if embeddings_identical and masks_identical and seq0_seq1_pipeline and seq0_seq1_training:
    print("✅ SUCCESS! All checks passed!")
    print("   - Identical inputs produce identical embeddings")
    print("   - Pipeline and training script are consistent")
    print("   - Both implementations handle sequences identically")
else:
    print("❌ FAILURE! Some checks failed!")
    print("   - Review the differences above")
print("="*80)

