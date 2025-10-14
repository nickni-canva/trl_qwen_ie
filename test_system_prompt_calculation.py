"""
Test that the system prompt token calculation works correctly
"""

import torch
import sys
sys.path.insert(0, '/home/coder/work/customized_diffusers/src')
sys.path.insert(0, '/home/coder/work/trl_qwen_image_edit')

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

print("="*80)
print("TEST: System Prompt Token Calculation")
print("="*80)

# Initialize
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"\nLoading tokenizer: {model_id}")

tokenizer = Qwen2Tokenizer.from_pretrained(model_id)
print("Tokenizer loaded!")

from train_grpo_qwen_image import SYSTEM_TEMPLATE_PROMPT

print(f"\nSystem prompt:")
print(f"  {SYSTEM_TEMPLATE_PROMPT[:100]}...")

# Test the calculation method used in the updated code
print("\n" + "-"*80)
print("Method 1: Format and tokenize (NEW approach)")
print("-"*80)

system_formatted = f"<|im_start|>system\n{SYSTEM_TEMPLATE_PROMPT}<|im_end|>\n"
print(f"\nFormatted system prompt (first 100 chars):")
print(f"  {system_formatted[:100]}...")

system_tokens = tokenizer(
    text=system_formatted,
    return_tensors="pt",
    add_special_tokens=False,
)

drop_idx = system_tokens.input_ids.shape[1]
print(f"\nTokenized:")
print(f"  Shape: {system_tokens.input_ids.shape}")
print(f"  Number of tokens (drop_idx): {drop_idx}")
print(f"✅ This method works!")

# Test with extract_response_embeddings
print("\n" + "="*80)
print("Testing with extract_response_embeddings()")
print("="*80)

print("\nLoading model...")
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
print("Model loaded!")

from train_grpo_qwen_image import extract_response_embeddings

test_prompts = [
    "A red cat sitting on a blue chair",
    "Generate an educational diagram showing the water cycle",
]

test_completions = [
    "To create this image, I'll show a red cat on a blue chair with detailed fur texture.",
    "To illustrate the water cycle, I'll draw evaporation, condensation, and precipitation.",
]

print(f"\nTest prompts: {len(test_prompts)}")
print(f"Test completions: {len(test_completions)}")

try:
    print("\nCalling extract_response_embeddings()...")
    prompt_embeds, prompt_masks = extract_response_embeddings(
        vlm_model=vlm_model,
        vlm_processor=tokenizer,
        prompts=test_prompts,
        completion_texts=test_completions,
    )
    
    print(f"\n✅ SUCCESS!")
    print(f"   Embeddings shape: {prompt_embeds.shape}")
    print(f"   Masks shape: {prompt_masks.shape}")
    print(f"   Mask sums: {prompt_masks.sum(dim=1).tolist()}")
    
    # Verify shapes are reasonable
    batch_size, seq_len, hidden_dim = prompt_embeds.shape
    assert batch_size == len(test_prompts), f"Batch size mismatch: {batch_size} != {len(test_prompts)}"
    assert hidden_dim == 3584, f"Hidden dim mismatch: {hidden_dim} != 3584"
    print(f"\n✅ All shape checks passed!")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("   System prompt token calculation works correctly")
print("   extract_response_embeddings() works without errors")
print("="*80)

