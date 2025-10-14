#!/usr/bin/env python3
"""
GRPO Training for Qwen-Image-Edit-Response

Clean implementation following grpo_vlm.py pattern.

Pipeline:
1. GRPO generates VLM completion TEXT (no embeddings yet)
2. Take completion text → forward pass → extract response embeddings
3. Use embeddings to generate edited image via diffusion
4. Evaluate edited image with GPT → reward
5. GRPO updates VLM based on reward

Only the VLM (text_encoder) is trained. DiT is frozen.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_dataset
from transformers import Qwen2VLProcessor
from loguru import logger

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward


@dataclass
class ImageEditArgs:
    """Arguments specific to image editing"""
    edit_model_path: str = field(
        default="Qwen/Qwen-Image-Edit-2509",
        metadata={"help": "Diffusion model path"}
    )
    complexity: str = field(
        default="1-8",
        metadata={"help": "Dataset complexity level(s). Use '1-8' for range, '1,3,5,8' for specific levels, or '8' for single level"}
    )
    image_type: str = field(
        default="real",
        metadata={"help": "Image type: real or syn"}
    )
    openai_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "OpenAI API key for evaluation"}
    )
    alignment_weight: float = field(default=0.6)
    quality_weight: float = field(default=0.4)
    n_evals: int = field(default=3)
    use_think_format_reward: bool = field(default=False)


################
# Dataset
################

def parse_complexity(complexity_str: str) -> List[int]:
    """Parse complexity string into list of levels
    
    Examples:
        "1-8" → [1, 2, 3, 4, 5, 6, 7, 8]
        "1,3,5,8" → [1, 3, 5, 8]
        "8" → [8]
    """
    complexity_str = complexity_str.strip()
    
    # Check if it's a range
    if '-' in complexity_str:
        parts = complexity_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid complexity range: {complexity_str}")
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))
    
    # Check if it's comma-separated
    if ',' in complexity_str:
        return [int(x.strip()) for x in complexity_str.split(',')]
    
    # Single value
    return [int(complexity_str)]


def prepare_dataset(complexity: str, image_type: str, max_samples: int = 1000):
    """Load and format Complex-Edit dataset for GRPO
    
    Dataset structure:
    - 'image': PIL Image
    - 'edit': dict with 'compound' key containing list of compound instructions
    
    We use the compound instructions which combine multiple atomic edits.
    Complexity parameter can specify multiple levels (e.g., "1-8", "1,3,5,8", or "8").
    Each image will be expanded to create one sample per complexity level.
    """
    
    complexity_levels = parse_complexity(complexity)
    logger.info(f"Loading Complex-Edit (complexity levels={complexity_levels}, type={image_type})")
    
    dataset = load_dataset("UCSC-VLAA/Complex-Edit")
    
    # Complex-Edit has splits like 'test_real', 'test_syn'
    split_name = f"test_{image_type}"
    
    if split_name not in dataset:
        logger.error(f"Split '{split_name}' not found. Available: {list(dataset.keys())}")
        raise ValueError(f"Split '{split_name}' not in dataset")
    
    logger.info(f"Using split: {split_name}")
    dataset = dataset[split_name]
    logger.info(f"Original dataset size: {len(dataset)} samples")
    logger.info(f"Dataset columns: {dataset.column_names}")
    
    # Convert to RGB
    def to_rgb(ex):
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        ex["image"] = img
        return ex
    
    dataset = dataset.map(to_rgb)
    
    # Format for GRPO - expand each image to multiple complexity levels
    SYSTEM_PROMPT = (
        "Describe the key features of the input image (color, shape, size, texture, "
        "objects, background), then explain how the user's text instruction should "
        "alter or modify the image. Generate a new image that meets the user's "
        "requirements while maintaining consistency with the original input where appropriate."
    )
    
    # Expand dataset: create one sample per (image, complexity_level) pair
    expanded_samples = []
    
    for ex in dataset:
        for complexity_level in complexity_levels:
            # Extract compound instruction at the specified complexity level
            # complexity 1-8 maps to compound indices 0-7
            compound_idx = min(complexity_level - 1, len(ex["edit"]["compound"]) - 1)
            compound_idx = max(0, compound_idx)  # Ensure non-negative
            
            instruction = ex["edit"]["compound"][compound_idx]["compound_instruction"]
            
            expanded_samples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                "image": ex["image"],
                "instruction": instruction,
                "complexity_level": complexity_level,  # Track which level this is
            })
    
    # Convert back to HF dataset
    from datasets import Dataset
    dataset = Dataset.from_list(expanded_samples)
    
    logger.info(f"Expanded dataset size: {len(dataset)} samples (from {len(dataset) // len(complexity_levels)} images × {len(complexity_levels)} complexity levels)")
    
    # Limit samples
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {len(dataset)} samples")
    
    logger.info(f"Dataset ready: {len(dataset)} total samples")
    logger.info(f"Complexity levels used: {complexity_levels}")
    
    # Split into train and eval
    if len(dataset) > 100:
        train_size = len(dataset) - 100
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        logger.info(f"Split into train: {len(train_dataset)}, eval: {len(eval_dataset)}")
        return train_dataset, eval_dataset
    else:
        return dataset, None


################
# Reward Function
################
def create_reward_function(
    edit_model_path: str,
    vlm_model,
    vlm_processor,
    api_key: str,
    alignment_weight: float,
    quality_weight: float,
    n_evals: int,
):
    """
    Create reward function that:
    1. Extracts embeddings from GRPO's completion text (Option C)
    2. Generates edited images
    3. Evaluates with GPT
    """
    
    # Lazy load
    edit_pipeline = None
    gpt_eval_fn = None
    
    def get_pipeline():
        nonlocal edit_pipeline
        if edit_pipeline is None:
            from diffusers import QwenImageEditResponsePipeline
            import torch.distributed as dist
            
            # Get device for this process
            if dist.is_initialized():
                device = f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Loading pipeline on {device}")
            
            edit_pipeline = QwenImageEditResponsePipeline.from_pretrained(
                edit_model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            
            # CRITICAL: Replace text_encoder with GRPO-trained VLM
            edit_pipeline.text_encoder = vlm_model[0]
            logger.info("Pipeline loaded, text_encoder = trained VLM")
        
        return edit_pipeline
    
    def get_gpt_evaluator():
        nonlocal gpt_eval_fn
        if gpt_eval_fn is None:
            from openai import OpenAI
            from pydantic import BaseModel
            from typing import Literal
            
            client = OpenAI(api_key=api_key)
            
            # EXACT prompts from Complex-Edit eval (WITH rubric, WITH CoT for alignment, WITHOUT CoT for quality)
            ALIGNMENT_SYSTEM_PROMPT = """You are required to evaluate the result of an instruction-based image editing model.
Given an input image, an output image and a text instruction, you are required to access the output image based on whether the changes made to the input image align with the text instruction.

You are required to give two integer scores in [0, 10] based on the following criteria:
1. Instruction Following: whether the required changes occur in the output image, regardless of whether unnecessary changes are also made. 10 means that all the changes required by the instruction occur in the output image, 0 means that no changes required by the instruction occur in the output image.
2. Identity Preservation: whether elements that should not be changed stay the same in the output image, regardless of whether required changes occur. 10 means that no unnecessary changes occur in the output image, 0 means that all elements in the input image that should be kept the same are changed in the output image.

Here is the detailed rubric for Instruction Following:
* 10 (Perfect Instruction Following): All the required changes occur in the output image.
* 9 (Near Perfect Instruction Following with negligible deviations): Almost all instructed changes are present but negligible deviations exist (e.g., a tiny color variation such as the cat in the image is now black but the ears are grey).
* 7-8 (Strong Instruction Following with minor deviations): Most required changes are applied accurately. Minor deviations exist but do not substantially alter the intended modification (e.g., a car is changed to blue as instructed, but the reflection on its surface still contains a red tint).
* 5-6 (Moderate Instruction Following with noticeable deviations): The output reflects an attempt to follow instructions but with moderate errors (e.g., adding a required element but with incorrect attributes like color or shape).
* 3-4 (Weak Instruction Following with major deviations): Most required modifications are missing, incorrect, or only vaguely implemented. Significant elements from the instruction are misrepresented (e.g., when instructed to add a hat, a small, barely visible accessory is added to the head, but it does not resemble a proper hat).
* 1-2 (Minimal Instruction Following with severe deviations): A vague attempt is made, but the required modifications are either incorrect or so minimal that they do not fulfill the instruction (e.g., the instruction asks to remove a person from the image, but they are still visible, just slightly blurred or faded instead of being properly erased.).
* 0 (Complete failed Instruction Following): The output image is entirely unrelated to the instruction.

Here is the detailed rubric for Identity Preservation:
* 10 (Perfect Identity Preservation): All key elements that should remain unchanged are completely preserved and indistinguishable from the input (e.g., a person's face, expression, and proportions remain completely unchanged except for the required edits).
* 9 (Near Perfect Identity Preservation with negligible distortion): Key elements that should remain unchanged are preserved with negligible distortion (e.g., A person's face is identical except for a tiny, imperceptible variation in hair texture).
* 7-8 (Strong Identity Preservation with minor distortion): Small details of the key elements may have changed, but they do not significantly disrupt the overall identity (e.g., a pet's fur pattern remains mostly accurate, but a minor detail like a stripe or spot is different).
* 5-6 (Moderate Identity Preservation with noticeable distortion): Most of the key elements remain recognizable but with noticeable distortions (e.g., the instruction asks to change a car's color, but the car's shape or size is modified along with the color).
* 3-4 (Weak Identity Preservation with major distortion): Key elements maintain a general resemblance but noticeable changes are present (e.g., the instruction asks to brighten the sky, but additional buildings in the background appear or disappear).
* 1-2 (Minimal Identity Preservation with severe distortion): Most key elements are significantly altered or replaced. The key elements in the output retain only minor aspects of the original, but major features are incorrect (e.g., a person's face is still a human face, but it no longer resembles the original person at all).
* 0 (Complete failed Identity Preservation): All key elements that should remain unchanged are altered, distorted, or missing.

Note that these two scores should be graded independently, and a low score for one criterion should not affect the score for the other criterion.
For example, an output image that is identical to the input image should have an Instruction Following score of 0, but an Identity Preservation score of 10. Also, an output image that has no relevance with the input image should have an Identity Preservation score of 0 unless the instruction specifically orders the model to create a whole different image, but it should not affect the Instruction Following score as long as changes required by the instruction occur in the output.

If the instruction contains several atomic operations, evaluate the Instruction Following for each atomic operation separately and then average the scores as the assessment for Instruction Following."""

            QUALITY_SYSTEM_PROMPT = """You are required to evaluate the result of an instruction-based image editing model.
Given an output image and a text instruction, you are required to access the output image's "Perceptual Quality".

You are required to give one integer score in [0, 10] with 0 indicating extreme disharmony characterized by numerous conflicting or clashing elements, and 10 indicating perfect harmony with all components blending effortlessly.

These are the criteria:
1. Consistency in lighting and shadows: The light source and corresponding shadows are consistent across various elements, with no discrepancies in direction or intensity.
2. Element cohesion: Every item in the image should logically fit within the scene's context, without appearing misplaced or extraneous.
3. Integration and edge smoothness: Objects should blend seamlessly into their surroundings, with edges that do not appear artificially inserted or poorly integrated.
4. Aesthetic uniformity and visual flow: The image should not only be aesthetically pleasing but also facilitate a natural visual journey, without abrupt interruptions caused by disharmonious elements.

Here is the detailed rubric:
* 10 (Perfect Perceptual Quality): The image appears flawlessly natural, and all objects are seamlessly integrated into the environment with consistent lighting and shadows. There is no visual artifact at all.
* 9 (Near Perfect Perceptual Quality with negligible incoherence): The image is very close to perfect, but a tiny, almost imperceptible inconsistency exists. Seamless integration, but one might notice an extremely subtle flaw. (e.g., a person added to a group photo blends in perfectly, but upon close examination, their shadow is slightly softer than others.)
* 7-8 (Strong Perceptual Quality with minor incoherence): Minor incoherence and artifacts are present but they do not significantly detract from the overall harmony. (e.g., a sunset scene where the added reflections on water are slightly off in intensity, but the image still looks highly realistic.)
* 5-6 (Moderate Perceptual Quality with noticeable incoherence): There is noticeable visual artifacts affecting the image's harmony. Lighting and shadows may be misaligned or inconsistent. (e.g., an animal is distorted in size or shape, making it appear out of place in the scene.)
* 3-4 (Weak Perceptual Quality with major incoherence): Disharmonious elements are prominent, greatly disturbing the visual harmony. (e.g., an animal's shape or a person's face is greatly distorted, only showing some resemblance of the animal species or a human face.)
* 1-2 (Minimal Perceptual Quality with severe incoherence): The whole scene is distorted, making it difficult to recognize the objects or subjects in the image.
* 0 (Complete failed Perceptual Quality): The image is completely random and makes no sense at all.

Note that if something unrealistic is requested in the instruction, such as the motion blur of the background or the sci-fi style of an object, then it is not considered "unrealistic". Yet you are not here to evaluate whether the output image follows the instruction but to evaluate the perceptual quality of the output image based on the instruction."""
            
            # Define structured output schemas (matching Complex-Edit benchmark)
            class CoTAlignment(BaseModel):
                reasoning: str
                instruction_following: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                identity_preservation: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            
            class Quality(BaseModel):
                perceptual_quality: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            
            def evaluate_batch(input_images, edited_images, instructions):
                """Evaluate with GPT-4o following EXACT Complex-Edit eval format with structured outputs"""
                import base64
                from io import BytesIO
                import sys
                
                print(f"\n🔥 GPT EVALUATOR CALLED with {len(input_images)} images", flush=True)
                sys.stdout.flush()
                
                rewards = []
                for idx, (inp_img, edit_img, instr) in enumerate(zip(input_images, edited_images, instructions)):
                    print(f"   📊 Evaluating sample {idx+1}/{len(input_images)}...", flush=True)
                    # Convert images to base64
                    def img_to_b64(img):
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        return base64.b64encode(buffered.getvalue()).decode()
                    
                    inp_b64 = img_to_b64(inp_img)
                    edit_b64 = img_to_b64(edit_img)
                    
                    try:
                        # Alignment evaluation WITH CoT (Chain of Thought) + rubric
                        alignment_user_prompt = f"""The first image is the input image and the second image is the output image.
The text instruction is:
{instr}

If the instruction contains several atomic operations, evaluate the Instruction Following for each atomic operation separately and then average the scores as the assessment for Instruction Following."""

                        align_resp = client.beta.chat.completions.parse(
                            model="gpt-4o-2024-11-20",
                            messages=[
                                {"role": "system", "content": ALIGNMENT_SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{inp_b64}"}},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edit_b64}"}},
                                        {"type": "text", "text": alignment_user_prompt},
                                    ]
                                }
                            ],
                            response_format=CoTAlignment,
                        )
                        
                        # Parse alignment scores from structured output
                        align_result = align_resp.choices[0].message.parsed
                        instruction_following = int(align_result.instruction_following) / 10.0
                        identity_preservation = int(align_result.identity_preservation) / 10.0
                        reasoning = align_result.reasoning
                        
                        # Quality evaluation WITHOUT CoT + rubric
                        quality_user_prompt = f"""The corresponding text instruction is:
{instr}"""

                        quality_resp = client.beta.chat.completions.parse(
                            model="gpt-4o-2024-11-20",
                            messages=[
                                {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edit_b64}"}},
                                        {"type": "text", "text": quality_user_prompt},
                                    ]
                                }
                            ],
                            response_format=Quality,
                        )
                        
                        quality_score = int(quality_resp.choices[0].message.parsed.perceptual_quality) / 10.0
                        
                        # Combined reward (matching your original weights)
                        reward = alignment_weight * instruction_following + quality_weight * quality_score
                        rewards.append(reward)
                        
                        print(f"      ✅ Sample {idx+1}: IF={instruction_following:.2f}, IP={identity_preservation:.2f}, Q={quality_score:.2f}, Reward={reward:.2f}", flush=True)
                        logger.info(f"Eval scores - IF: {instruction_following:.2f}, IP: {identity_preservation:.2f}, Q: {quality_score:.2f}, Reward: {reward:.2f}")
                        if idx == 0:
                            logger.info(f"Reasoning (sample 1): {reasoning[:200]}...")
                        
                    except Exception as e:
                        print(f"      ❌ Sample {idx+1} failed: {e}", flush=True)
                        logger.error(f"GPT eval error: {e}")
                        import traceback
                        traceback.print_exc()
                        rewards.append(0.5)
                
                print(f"🎁 Final rewards: {rewards}", flush=True)
                return rewards
            
            gpt_eval_fn = evaluate_batch
        
        return gpt_eval_fn
    
    def reward_fn(completions, image: List, instruction: List, **kwargs):
        """
        Main reward function called by GRPO.
        
        Args:
            completions: List[List[Dict]] - GRPO's generated completions
                Format: [[{"role": "assistant", "content": "..."}], ...]
            image: List[PIL.Image] - Input images
            instruction: List[str] - Edit instructions
        
        Returns:
            List[float] - Reward scores
        """
        try:
            # Force flush logs immediately
            import sys
            print("=" * 100, flush=True)
            print("🎯 REWARD FUNCTION CALLED!", flush=True)
            print(f"   Image batch size: {len(image)}", flush=True)
            print(f"   Completions: {len(completions)}", flush=True)
            print("=" * 100, flush=True)
            sys.stdout.flush()
            
            batch_size = len(image)
            logger.info(f"=== Reward computation for batch of {batch_size} ===")
            
            # Extract completion texts from GRPO
            completion_texts = [comp[0]["content"] for comp in completions]
            logger.info(f"Completion text sample: {completion_texts[0]}")
            print(f"📝 First completion: {completion_texts[0][100:]}...", flush=True)
            
            # Get pipeline
            pipe = get_pipeline()
            
            # STEP 1: Extract embeddings from completion text
            # We need to get embeddings for: [system + user + assistant + completion_text]
            # Then drop the system part, keep [user + assistant + completion]
            
            logger.info("Extracting response embeddings from completion text...")
            response_embeds, response_masks = extract_response_embeddings(
                vlm_model=vlm_model[0],
                vlm_processor=vlm_processor,
                images=image,
                instructions=instruction,
                completion_texts=completion_texts,
            )
            
            # STEP 2: Generate edited images using these embeddings
            logger.info("Generating edited images...")
            
            # Call pipeline's internal method to generate with embeddings
            # We bypass the text generation and use pre-computed embeddings
            edited_images = generate_with_embeddings(
                pipe=pipe,
                images=image,
                prompt_embeds=response_embeds,
                prompt_embeds_mask=response_masks,
            )
            
            # STEP 3: Evaluate with GPT
            print("🔍 STEP 3: Calling GPT evaluator...", flush=True)
            logger.info("Evaluating with GPT...")
            evaluator = get_gpt_evaluator()
            print(f"   Evaluator initialized, calling with {len(image)} samples...", flush=True)
            rewards = evaluator(image, edited_images, instruction)
            
            print(f"✅ GPT Evaluation complete! Rewards: {rewards}", flush=True)
            logger.info(f"Rewards: {rewards}")
            return rewards
            
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            import traceback
            traceback.print_exc()
            return [0.5] * len(image)
    
    return reward_fn


def extract_response_embeddings(
    vlm_model,
    vlm_processor,
    images,
    instructions,
    completion_texts,
):
    """
    Extract embeddings from completion text via forward pass (Option C).
    
    Process:
    1. Format input: [system + user + instruction] + [assistant + completion_text]
    2. Forward pass to get hidden states
    3. Drop system prompt part
    4. Return embeddings for [user + instruction + assistant + completion]
    """
    
    # Format prompts with completions
    SYSTEM_PROMPT = (
        "Describe the key features of the input image (color, shape, size, texture, "
        "objects, background), then explain how the user's text instruction should "
        "alter or modify the image. Generate a new image that meets the user's "
        "requirements while maintaining consistency with the original input where appropriate."
    )
    
    img_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    
    # Build full prompts with completions
    full_prompts = []
    for instr, completion in zip(instructions, completion_texts):
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{img_template.format(1)}{instr}<|im_end|>\n"
            f"<|im_start|>assistant\n{completion}<|im_end|>"
        )
        full_prompts.append(prompt)
    
    # Process with VLM
    inputs = vlm_processor(
        text=full_prompts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device)
    
    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = vlm_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            output_hidden_states=True,
        )
    
    hidden_states = outputs.hidden_states[-1]  # Last layer
    
    # Extract masked hidden states
    split_hidden_states = extract_masked_hidden(hidden_states, inputs.attention_mask)
    
    # Drop system prompt tokens (first 64 tokens after padding)
    # With left-padding, we need to account for padding offset
    processed_hidden_states = []
    for i, hidden in enumerate(split_hidden_states):
        # Count padding
        mask = inputs.attention_mask[i]
        num_padding = (mask == 0).sum().item()
        
        # Drop: padding + system prompt (64 tokens)
        drop_idx = num_padding + 64
        
        if drop_idx < len(hidden):
            processed_hidden_states.append(hidden[drop_idx:])
        else:
            logger.warning(f"Drop idx {drop_idx} >= length {len(hidden)}, keeping all")
            processed_hidden_states.append(hidden)
    
    # Pad to max length
    max_len = max(h.size(0) for h in processed_hidden_states)
    
    prompt_embeds = torch.stack([
        torch.cat([h, h.new_zeros(max_len - h.size(0), h.size(1))])
        for h in processed_hidden_states
    ])
    
    prompt_masks = torch.stack([
        torch.cat([
            torch.ones(h.size(0), dtype=torch.long, device=h.device),
            torch.zeros(max_len - h.size(0), dtype=torch.long, device=h.device)
        ])
        for h in processed_hidden_states
    ])
    
    return prompt_embeds, prompt_masks


def extract_masked_hidden(hidden_states, mask):
    """Extract hidden states using attention mask"""
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def generate_with_embeddings(pipe, images, prompt_embeds, prompt_embeds_mask):
    """Generate images using pre-computed embeddings"""
    
    # Use pipeline with pre-computed embeddings instead of text prompts
    # Pipeline accepts either prompt OR prompt_embeds (not both)
    output = pipe(
        image=images,
        prompt=None,  # Must be None when using prompt_embeds
        negative_prompt=" ",
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        num_inference_steps=50,
        true_cfg_scale=4.0,
    )
    
    return output.images


################
# Main
################
if __name__ == "__main__":
    # Parse args
    parser = TrlParser((ImageEditArgs, GRPOConfig, ModelConfig))
    img_args, training_args, model_args = parser.parse_args_and_config()
    
    # Setup
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.add(
        Path(training_args.output_dir) / "training.log",
        level="INFO",
    )
    
    logger.info("="*80)
    logger.info("GRPO Training: Qwen-Image-Edit-Response")
    logger.info("="*80)
    
    # Get API key
    api_key = img_args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")
    
    # Model config
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    
    quant_config = get_quantization_config(model_args)
    if quant_config:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quant_config
    
    # Load dataset
    train_ds, eval_ds = prepare_dataset(
        img_args.complexity,
        img_args.image_type,
        max_samples=1000,
    )
    
    # Calculate max_steps if not set
    if training_args.max_steps <= 0:
        # Calculate based on dataset size and batch size
        num_samples = len(train_ds)
        batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        num_devices = training_args.world_size if hasattr(training_args, 'world_size') else 1
        effective_batch_size = batch_size * num_devices
        
        steps_per_epoch = num_samples // effective_batch_size
        training_args.max_steps = steps_per_epoch * training_args.num_train_epochs
        
        logger.info(f"Calculated max_steps: {training_args.max_steps}")
        logger.info(f"  num_samples: {num_samples}")
        logger.info(f"  effective_batch_size: {effective_batch_size}")
        logger.info(f"  steps_per_epoch: {steps_per_epoch}")
    
    # Setup processor FIRST (needed for reward function)
    logger.info("Setting up VLM processor...")
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = 'left'
    
    # Create reward function BEFORE trainer
    # NOTE: vlm_model will be set inside the reward function when first called
    logger.info("Creating reward function...")
    vlm_model_ref = [None]  # Use mutable reference to share model
    
    def set_vlm_model(model):
        """Called by trainer to inject the model into reward function"""
        vlm_model_ref[0] = model
        logger.info("✅ VLM model injected into reward function")
    
    reward_fn = create_reward_function(
        edit_model_path=img_args.edit_model_path,
        vlm_model=vlm_model_ref,  # Pass mutable reference
        vlm_processor=processor,
        api_key=api_key,
        alignment_weight=img_args.alignment_weight,
        quality_weight=img_args.quality_weight,
        n_evals=img_args.n_evals,
    )
    
    # Initialize GRPO trainer with reward function
    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[reward_fn] + ([think_format_reward] if img_args.use_think_format_reward else []),  # ✅ Pass during init
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
    )
    
    # Inject model into reward function
    set_vlm_model(trainer.model)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    logger.info("Saving model...")
    trainer.save_model(training_args.output_dir)
    
    logger.info("✅ Training complete!")

