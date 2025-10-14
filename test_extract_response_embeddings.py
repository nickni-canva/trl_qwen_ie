"""
Test the actual extract_response_embeddings function from train_grpo_qwen_image.py
with identical inputs to verify consistency
"""

import torch
import sys
sys.path.insert(0, '/home/coder/work/customized_diffusers/src')
sys.path.insert(0, '/home/coder/work/trl_qwen_image_edit')

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

# Import the actual functions from training script
from train_grpo_qwen_image import extract_response_embeddings, extract_masked_hidden, SYSTEM_TEMPLATE_PROMPT

print("="*80)
print("TEST: extract_response_embeddings() with IDENTICAL inputs")
print("="*80)

# Test with identical prompts and completions
test_prompts = [
    "A red cat sitting on a blue chair",
    "A red cat sitting on a blue chair",  # Identical
]

test_completions = [
    "To create this image, I'll depict a vibrant red cat with soft fur texture sitting gracefully on a blue wooden chair...",
    "To create this image, I'll depict a vibrant red cat with soft fur texture sitting gracefully on a blue wooden chair...",  # Identical
]

print(f"\nPrompts:")
print(f"  Prompt 1: {test_prompts[0]}")
print(f"  Prompt 2: {test_prompts[1]}")
print(f"  Identical? {test_prompts[0] == test_prompts[1]}")

print(f"\nCompletions:")
print(f"  Completion 1: {test_completions[0][:80]}...")
print(f"  Completion 2: {test_completions[1][:80]}...")
print(f"  Identical? {test_completions[0] == test_completions[1]}")

# ============================================================================
# Call the actual extract_response_embeddings function
# ============================================================================
print("\n" + "="*80)
print("Calling extract_response_embeddings() from training script")
print("="*80)

prompt_embeds, prompt_masks = extract_response_embeddings(
    vlm_model=vlm_model,
    vlm_processor=tokenizer,
    prompts=test_prompts,
    completion_texts=test_completions,
)

print(f"\nResults:")
print(f"  Embeddings shape: {prompt_embeds.shape}")
print(f"  Masks shape: {prompt_masks.shape}")
print(f"  Mask sums: {prompt_masks.sum(dim=1).tolist()}")

# Check if identical inputs produce identical outputs
seq0_seq1_identical = torch.allclose(prompt_embeds[0], prompt_embeds[1])
max_diff = (prompt_embeds[0] - prompt_embeds[1]).abs().max().item()

print(f"\n✓ Sequence 0 vs Sequence 1:")
print(f"  Identical? {seq0_seq1_identical}")
print(f"  Max diff: {max_diff}")

# ============================================================================
# Manually trace through to compare with pipeline
# ============================================================================
print("\n" + "="*80)
print("Manual step-by-step comparison with pipeline approach")
print("="*80)

# Build full prompts (same as in the function)
print(f"\nSystem prompt used:")
print(f"  {SYSTEM_TEMPLATE_PROMPT[:100]}...")

full_prompts = []
for prompt, completion in zip(test_prompts, test_completions):
    full_prompt = (
        f"<|im_start|>system\n{SYSTEM_TEMPLATE_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion}<|im_end|>"
    )
    full_prompts.append(full_prompt)

print(f"\nFull prompt (first 150 chars):")
print(f"  {full_prompts[0][:150]}...")
print(f"  Prompts identical? {full_prompts[0] == full_prompts[1]}")

# Tokenize
inputs = tokenizer(
    text=full_prompts,
    padding=True,
    return_tensors="pt",
)

print(f"\nTokenized:")
print(f"  input_ids shape: {inputs.input_ids.shape}")
print(f"  Token IDs identical? {torch.equal(inputs.input_ids[0], inputs.input_ids[1])}")

# Get hidden states
with torch.no_grad():
    outputs = vlm_model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
    )

hidden_states = outputs.hidden_states[-1]
print(f"  hidden_states shape: {hidden_states.shape}")
print(f"  Hidden states identical? {torch.allclose(hidden_states[0], hidden_states[1])}")

# Extract masked
split_hidden = extract_masked_hidden(hidden_states, inputs.attention_mask)
print(f"\nAfter extract_masked_hidden:")
print(f"  Seq 0 shape: {split_hidden[0].shape}")
print(f"  Seq 1 shape: {split_hidden[1].shape}")
print(f"  Identical? {torch.allclose(split_hidden[0], split_hidden[1])}")

# Calculate drop_idx
system_message = [{"role": "system", "content": SYSTEM_TEMPLATE_PROMPT}]
system_tokens = tokenizer.apply_chat_template(
    system_message,
    tokenize=True,
    add_generation_prompt=False
)
drop_idx = len(system_tokens)
print(f"\nSystem prompt tokens: {drop_idx}")

# Drop uniformly
processed_hidden = [e[drop_idx:] for e in split_hidden]
print(f"\nAfter dropping {drop_idx} tokens:")
print(f"  Seq 0 shape: {processed_hidden[0].shape}")
print(f"  Seq 1 shape: {processed_hidden[1].shape}")
print(f"  Identical? {torch.allclose(processed_hidden[0], processed_hidden[1])}")

# Pad
max_len = max(h.size(0) for h in processed_hidden)
attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in processed_hidden]

manual_prompt_embeds = torch.stack([
    torch.cat([h, h.new_zeros(max_len - h.size(0), h.size(1))])
    for h in processed_hidden
])

manual_prompt_masks = torch.stack([
    torch.cat([u, u.new_zeros(max_len - u.size(0))])
    for u in attn_mask_list
])

print(f"\nManual embeddings:")
print(f"  Shape: {manual_prompt_embeds.shape}")
print(f"  Seq 0 vs Seq 1 identical? {torch.allclose(manual_prompt_embeds[0], manual_prompt_embeds[1])}")

# ============================================================================
# Compare function output vs manual computation
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION: Function output vs Manual computation")
print("="*80)

embeddings_match = torch.allclose(prompt_embeds, manual_prompt_embeds)
masks_match = torch.equal(prompt_masks, manual_prompt_masks)
max_embed_diff = (prompt_embeds - manual_prompt_embeds).abs().max().item()

print(f"\n✓ Embeddings match? {embeddings_match}")
print(f"  Max difference: {max_embed_diff}")
print(f"\n✓ Masks match? {masks_match}")

# ============================================================================
# Final checks
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

all_checks = [
    ("Identical inputs produce identical embeddings", seq0_seq1_identical),
    ("Function output matches manual computation", embeddings_match and masks_match),
    ("Max diff = 0 (perfect match)", max_diff == 0.0),
    ("Max embed diff = 0", max_embed_diff == 0.0),
]

print()
all_passed = True
for check_name, result in all_checks:
    status = "✅" if result else "❌"
    print(f"{status} {check_name}: {result}")
    all_passed = all_passed and result

print("\n" + "="*80)
if all_passed:
    print("✅ ALL TESTS PASSED!")
    print("   extract_response_embeddings() works correctly:")
    print("   - Identical inputs → identical outputs")
    print("   - Implementation matches expected behavior")
    print("   - Consistent with pipeline approach")
else:
    print("❌ SOME TESTS FAILED")
    print("   Review the output above for details")
print("="*80)

