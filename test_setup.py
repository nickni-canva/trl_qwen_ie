#!/usr/bin/env python3
"""
Quick test to verify the setup before training.
"""

import torch
from transformers import Qwen2VLProcessor
from loguru import logger

logger.info("Testing setup...")

# 1. Test VLM loading
logger.info("1. Testing VLM loading...")
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    logger.info("✅ VLM loaded successfully")
except Exception as e:
    logger.error(f"❌ VLM loading failed: {e}")
    exit(1)

# 2. Test processor
logger.info("2. Testing processor...")
try:
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    logger.info("✅ Processor loaded successfully")
except Exception as e:
    logger.error(f"❌ Processor loading failed: {e}")
    exit(1)

# 3. Test diffusion pipeline
logger.info("3. Testing diffusion pipeline...")
try:
    from diffusers import QwenImageEditResponsePipeline
    pipe = QwenImageEditResponsePipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    logger.info("✅ Diffusion pipeline loaded successfully")
except Exception as e:
    logger.error(f"❌ Pipeline loading failed: {e}")
    exit(1)

# 4. Test replacing text_encoder
logger.info("4. Testing text_encoder replacement...")
try:
    pipe.text_encoder = vlm
    logger.info("✅ text_encoder replaced with VLM")
except Exception as e:
    logger.error(f"❌ text_encoder replacement failed: {e}")
    exit(1)

# 5. Test dataset loading
logger.info("5. Testing dataset loading...")
try:
    from datasets import load_dataset
    ds = load_dataset("UCSC-VLAA/Complex-Edit")
    
    # Complex-Edit has splits like 'test_real', 'test_syn'
    if "test_real" in ds:
        test_ds = ds["test_real"]
        logger.info(f"✅ Dataset loaded: {len(test_ds)} samples in 'test_real' split")
    else:
        logger.error(f"Available splits: {list(ds.keys())}")
        raise ValueError("Expected 'test_real' split not found")
except Exception as e:
    logger.error(f"❌ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. Test TRL imports
logger.info("6. Testing TRL imports...")
try:
    from trl import GRPOConfig, GRPOTrainer, get_peft_config
    logger.info("✅ TRL imports successful")
except Exception as e:
    logger.error(f"❌ TRL imports failed: {e}")
    exit(1)

# 7. Test embedding extraction
logger.info("7. Testing embedding extraction...")
try:
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # Test prompt WITH image placeholder tokens
    test_prompt = (
        "<|im_start|>system\nTest<|im_end|>\n"
        "<|im_start|>user\n"
        "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        "Test instruction<|im_end|>\n"
        "<|im_start|>assistant\nTest response<|im_end|>"
    )
    
    # Process
    inputs = processor(
        text=[test_prompt],
        images=[dummy_img],
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")
    
    # Forward pass
    with torch.no_grad():
        outputs = vlm(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            output_hidden_states=True,
        )
    
    hidden = outputs.hidden_states[-1]
    logger.info(f"✅ Embedding extraction works. Shape: {hidden.shape}")
    
except Exception as e:
    logger.error(f"❌ Embedding extraction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

logger.info("="*80)
logger.info("✅ ALL TESTS PASSED!")
logger.info("="*80)
logger.info("You can now run: bash run_8gpu.sh")

