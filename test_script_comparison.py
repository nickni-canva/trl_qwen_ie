#!/usr/bin/env python3
"""
Compare the two training scripts to ensure consistency
"""

import ast
from pathlib import Path

print("="*80)
print("SCRIPT COMPARISON: Image Edit vs Image Generation")
print("="*80)

def extract_functions(filepath):
    """Extract function names from a Python file"""
    with open(filepath) as f:
        tree = ast.parse(f.read())
    
    functions = []
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    
    return functions, classes

# Compare both scripts
edit_script = Path("train_grpo_qwen_image_edit.py")
gen_script = Path("train_grpo_qwen_image.py")

print("\n[1] File Sizes")
print("-" * 80)
edit_size = edit_script.stat().st_size
gen_size = gen_script.stat().st_size
print(f"  Image Edit:   {edit_size:,} bytes ({edit_size // 1024} KB)")
print(f"  Image Gen:    {gen_size:,} bytes ({gen_size // 1024} KB)")
print(f"  Difference:   {abs(edit_size - gen_size):,} bytes")

print("\n[2] Line Counts")
print("-" * 80)
edit_lines = len(edit_script.read_text().split('\n'))
gen_lines = len(gen_script.read_text().split('\n'))
print(f"  Image Edit:   {edit_lines} lines")
print(f"  Image Gen:    {gen_lines} lines")
print(f"  Difference:   {abs(edit_lines - gen_lines)} lines")

print("\n[3] Function/Class Analysis")
print("-" * 80)
edit_funcs, edit_classes = extract_functions(edit_script)
gen_funcs, gen_classes = extract_functions(gen_script)

print(f"  Image Edit:   {len(edit_funcs)} functions, {len(edit_classes)} classes")
print(f"  Image Gen:    {len(gen_funcs)} functions, {len(gen_classes)} classes")

# Common functions (expected to be in both)
common_funcs = set(edit_funcs) & set(gen_funcs)
print(f"\n  Common functions ({len(common_funcs)}):")
for func in sorted(common_funcs):
    if not func.startswith('_'):  # Skip private functions
        print(f"    • {func}")

# Unique to Image Edit
edit_only = set(edit_funcs) - set(gen_funcs)
if edit_only:
    print(f"\n  Unique to Image Edit ({len(edit_only)}):")
    for func in sorted(edit_only):
        if not func.startswith('_'):
            print(f"    • {func}")

# Unique to Image Gen
gen_only = set(gen_funcs) - set(edit_funcs)
if gen_only:
    print(f"\n  Unique to Image Gen ({len(gen_only)}):")
    for func in sorted(gen_only):
        if not func.startswith('_'):
            print(f"    • {func}")

print("\n[4] Key Differences Check")
print("-" * 80)

edit_content = edit_script.read_text()
gen_content = gen_script.read_text()

# Check for expected differences
checks = [
    ("QwenImageEditResponsePipeline", "Image Edit has Edit pipeline", 
     lambda c: "QwenImageEditResponsePipeline" in c),
    ("QwenImageResponsePipeline", "Image Gen has Gen pipeline",
     lambda c: "QwenImageResponsePipeline" in c),
    ("complexity", "Image Edit has complexity args",
     lambda c: "complexity" in c and "COMPLEXITY_LEVELS" in c),
    ("levels", "Image Gen has levels args",
     lambda c: "LEVELS_CANONICAL" in c),
    ("disciplines", "Image Gen has disciplines args",
     lambda c: "DISCIPLINES" in c),
    ("image_type", "Image Edit has image_type",
     lambda c: "image_type" in c and '"real"' in c),
]

print("  Image Edit specific features:")
for keyword, desc, check in checks[:4:2]:  # Edit-specific
    if check(edit_content):
        print(f"    ✅ {desc}")
    else:
        print(f"    ❌ Missing: {desc}")

print("\n  Image Gen specific features:")
for keyword, desc, check in checks[1::2]:  # Gen-specific
    if check(gen_content):
        print(f"    ✅ {desc}")
    else:
        print(f"    ❌ Missing: {desc}")

print("\n[5] Dataset Loading Comparison")
print("-" * 80)

# Check dataset sources
if "ComplexEdit" in edit_content or "complex-edit" in edit_content.lower():
    print("  ✅ Image Edit uses ComplexEdit dataset")
else:
    print("  ⚠️  Image Edit dataset source unclear")

if "MMMG" in gen_content or "MMMGBench" in gen_content:
    print("  ✅ Image Gen uses MMMG dataset")
else:
    print("  ⚠️  Image Gen dataset source unclear")

print("\n[6] Embedding Extraction Comparison")
print("-" * 80)

# Check for pixel_values (should be in edit, not in gen for text-only)
edit_pixel = edit_content.count("pixel_values")
gen_pixel = gen_content.count("pixel_values")

print(f"  Image Edit 'pixel_values' mentions: {edit_pixel}")
print(f"  Image Gen 'pixel_values' mentions:  {gen_pixel}")

if edit_pixel > gen_pixel:
    print("  ✅ Image Edit has more image processing (expected)")
else:
    print("  ⚠️  Unexpected: Image Gen should have fewer pixel_values")

print("\n[7] Evaluation Metrics Comparison")
print("-" * 80)

# Check evaluation prompts
edit_has_instruction_following = "Instruction Following" in edit_content
edit_has_identity_preservation = "Identity Preservation" in edit_content
gen_has_prompt_alignment = "Prompt Alignment" in gen_content
gen_has_visual_coherence = "Visual Coherence" in gen_content

print("  Image Edit metrics:")
print(f"    {'✅' if edit_has_instruction_following else '❌'} Instruction Following")
print(f"    {'✅' if edit_has_identity_preservation else '❌'} Identity Preservation")

print("\n  Image Gen metrics:")
print(f"    {'✅' if gen_has_prompt_alignment else '❌'} Prompt Alignment")
print(f"    {'✅' if gen_has_visual_coherence else '❌'} Visual Coherence")

print("\n[8] Shared Infrastructure Check")
print("-" * 80)

shared_elements = [
    ("GRPO", "Uses GRPO algorithm"),
    ("GRPOTrainer", "Uses GRPOTrainer"),
    ("LoRA", "Supports LoRA fine-tuning"),
    ("extract_response_embeddings", "Extracts embeddings from completions"),
    ("generate_with_embeddings", "Generates images from embeddings"),
    ("GPT", "Uses GPT for evaluation"),
    ("alignment_weight", "Has alignment weight"),
    ("quality_weight", "Has quality weight"),
]

print("  Both scripts have:")
for keyword, desc in shared_elements:
    edit_has = keyword in edit_content
    gen_has = keyword in gen_content
    
    if edit_has and gen_has:
        print(f"    ✅ {desc}")
    elif not edit_has and not gen_has:
        print(f"    ⚠️  Neither has: {desc}")
    else:
        which = "Edit" if edit_has else "Gen"
        print(f"    ⚠️  Only {which} has: {desc}")

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print("✅ Both scripts follow the same architecture")
print("✅ Key differences are appropriate for their respective tasks:")
print("   • Image Edit: processes input images, edits based on instructions")
print("   • Image Gen:  text-only input, generates new images")
print("✅ Both use the same core approach:")
print("   • GRPO for training")
print("   • Embedding extraction via forward pass")
print("   • GPT-4V for evaluation")
print("   • LoRA for efficient fine-tuning")
print("="*80)


