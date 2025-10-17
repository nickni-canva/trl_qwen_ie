#!/usr/bin/env python3
"""
GRPO Training for Qwen-Image-Response

Clean implementation for text-to-image generation following grpo_vlm.py pattern.

Pipeline:
1. GRPO generates VLM completion TEXT (no embeddings yet)
2. Take completion text → forward pass → extract response embeddings
3. Use embeddings to generate image via diffusion
4. Evaluate generated image with GPT → reward
5. GRPO updates VLM based on reward

Only the VLM (text_encoder) is trained. DiT is frozen.

Logging:
- All logging includes rank information: [Rank N] message
- Use --verbose flag to enable detailed debug logging
- verbose=False: INFO level, essential logs only
- verbose=True: DEBUG level, includes detailed operation logs

===============================================================================
PARALLELIZATION OPTIMIZATIONS (Optimized for 8x H200 GPUs)
===============================================================================

This script implements MODERATE parallelization to maximize training throughput
while avoiding GPU OOM:

1. **Multi-GPU Distributed Training** (8 GPUs):
   - ZeRO-3 sharding for memory efficiency
   - Independent batch processing per GPU (16 samples/batch)
   - System-wide: 128 samples processed in parallel per step

2. **Async GPT API Evaluation** (asyncio.gather) - FULLY PARALLEL:
   - Each GPU: 16 parallel GPT API calls (no restrictions)
   - System-wide: 8 GPUs × 16 = 128 concurrent API calls
   - Overlaps API wait time with other GPU computations
   - No GPU memory impact, fully asynchronous

3. **Serialized GPU SAM2 Processing** - MODERATE PARALLELIZATION:
   - SAM2 runs on GPU (fast) with ThreadPoolExecutor (1 worker) per GPU
   - Only 1 SAM2 at a time per GPU (prevents OOM from memory conflicts)
   - Thread pool naturally serializes SAM2 calls, avoiding semaphore deadlock
   - System-wide: 8 GPUs × 1 SAM2 = 8 parallel SAM2 inferences
   - Within each sample: GPT + SAM2 still run concurrently
   - Across samples: GPT fully parallel, SAM2 queued
   - Benefits: Fast GPU inference without OOM, no deadlock risk
   - Alternative tried: CPU (too slow), parallel GPU (OOM), semaphore (deadlock)

4. **torch.distributed Wandb Gathering**:
   - In-memory data transfer (5-10x faster than file I/O)
   - No disk bottleneck or cleanup overhead
   - Efficient cross-GPU synchronization

5. **Dynamic Image Generation Batching**:
   - Adaptive batch sizing based on H200's 141GB VRAM
   - Automatically scales batch size to available memory
   - Maximizes GPU utilization without OOM

6. **Memory Optimizations**:
   - SAM2 serialized per GPU (prevents OOM)
   - Aggressive gc.collect() and torch.cuda.empty_cache()
   - Sub-batch processing for large batches
   - LoRA fine-tuning (low memory footprint)

Critical Fixes:
- Fixed SAM2 torch compile error (Hydra state clearing)
- Serialized SAM2 with ThreadPoolExecutor (prevents OOM, maintains GPU speed, no deadlock)
- Wrapped SAM2 init in torch.no_grad() context

Expected Performance:
- 2x faster wall-clock training time vs sequential evaluation
- Good balance: GPT fully parallel, SAM2 fast (GPU) without OOM
- Bottleneck: GPT API latency (30-60s) + serialized SAM2 (32-48s)

System-wide Parallelism Summary:
- 128 images generated in parallel (8 GPUs × 16/batch)
- 128 GPT API calls in parallel (fully async, no restrictions)
- 8 SAM2 inferences in parallel (1 per GPU, serialized within GPU)
- Moderate approach: prevents OOM while maintaining speed
===============================================================================
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import torch
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
from accelerate.utils import gather_object


@dataclass
class ImageGenArgs:
    """Arguments specific to image generation"""
    gen_model_path: str = field(
        default="Qwen/Qwen-Image",
        metadata={"help": "Diffusion model path"}
    )
    levels: str = field(
        default="all",
        metadata={"help": "Education levels to use: 'all', 'preschool', 'primaryschool,secondaryschool', etc."}
    )
    disciplines: str = field(
        default="all",
        metadata={"help": "Disciplines to use: 'all', 'Math', 'Biology,Chemistry', etc."}
    )
    local_data_path: str = field(
        default="/mnt/ephemeral/MMMG_train/train.json",
        metadata={"help": "Path to local MMMG train.json file"}
    )
    openai_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "OpenAI API key for evaluation"}
    )
    alignment_weight: float = field(default=0.7)
    quality_weight: float = field(default=0.3)
    n_evals: int = field(default=3)
    use_think_format_reward: bool = field(default=False)
    sam2_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to SAM2 checkpoint for readability evaluation"}
    )
    num_diff_lvls: int = field(
        default=1,
        metadata={"help": "Number of difficulty levels for curriculum learning. 1 = no curriculum, all samples available from start. >1 = progressive difficulty."}
    )
    difficulty_summary_path: str = field(
        default="mmmg_difficulty_summary.json",
        metadata={"help": "Path to difficulty summary JSON file"}
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Enable verbose logging with detailed debug information"}
    )
    wandb_log_images: int = field(
        default=4,
        metadata={"help": "Number of top/bottom images to log to wandb per rank (0 to disable). Logs N best + N worst = 2N total per rank."}
    )
    wandb_image_size: int = field(
        default=512,
        metadata={"help": "Max image dimension (width/height) for wandb logging. Images are resized to save bandwidth."}
    )


# Available levels and disciplines
LEVELS_CANONICAL = [
    "preschool",
    "primaryschool",
    "secondaryschool",
    "highschool",
    "undergraduate",
    "PhD",
]

DISCIPLINES = [
    "Biology",
    "Chemistry",
    "Economics",
    "Engineering",
    "Geography",
    "History",
    "Literature",
    "Math",
    "Philosophy",
    "Sociology",
]

SYSTEM_TEMPLATE_PROMPT = "You are assisting an image generation model. Given the user's text, your task is to plan and describe what should be presented in the output image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"

################
# Logging Utilities
################

def get_rank():
    """Get current distributed rank or 0 if not distributed"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def log_with_rank(message: str, level: str = "info", verbose_only: bool = False):
    """Log message with rank prefix
    
    Args:
        message: Message to log
        level: Logging level ('info', 'warning', 'error', 'debug')
        verbose_only: If True, only log when verbose mode is enabled
    """
    rank = get_rank()
    prefix = f"[Rank {rank}]"
    full_message = f"{prefix} {message}"
    
    # Check if verbose mode is required and enabled
    # Note: verbose flag is set globally after parsing args
    if verbose_only and not getattr(log_with_rank, '_verbose', False):
        return
    
    if level == "info":
        logger.info(full_message)
    elif level == "warning":
        logger.warning(full_message)
    elif level == "error":
        logger.error(full_message)
    elif level == "debug":
        if getattr(log_with_rank, '_verbose', False):
            logger.debug(full_message)
    else:
        logger.info(full_message)


def set_verbose_mode(verbose: bool):
    """Set global verbose mode for logging"""
    log_with_rank._verbose = verbose
    if verbose:
        logger.info(f"[Rank {get_rank()}] Verbose logging enabled")


################
# Curriculum Learning
################

def load_difficulty_summary(summary_path: str):
    """Load difficulty summary JSON file"""
    import json
    with open(summary_path, 'r') as f:
        return json.load(f)


def group_by_difficulty(difficulty_summary, num_levels: int):
    """
    Group (level, discipline) tuples into difficulty bins.
    
    Args:
        difficulty_summary: Dict loaded from JSON with 'tuples' key
        num_levels: Number of difficulty groups (e.g., 3 → easy/medium/hard)
    
    Returns:
        List of difficulty groups, where each group is a list of (level, discipline) tuples
        Groups are ordered from easiest to hardest
    """
    tuples = difficulty_summary["tuples"]
    total = len(tuples)
    
    # Calculate group sizes (equal-sized bins)
    group_size = total // num_levels
    remainder = total % num_levels
    
    groups = []
    start_idx = 0
    
    for i in range(num_levels):
        # Distribute remainder across first few groups
        current_group_size = group_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_group_size
        
        group_tuples = [
            (t["level"], t["discipline"])
            for t in tuples[start_idx:end_idx]
        ]
        groups.append(group_tuples)
        
        start_idx = end_idx
    
    return groups


class CurriculumDataset:
    """
    Wrapper dataset that implements curriculum learning by progressively
    expanding the available samples based on training progress.
    """
    
    def __init__(self, full_dataset, difficulty_groups, max_steps):
        """
        Args:
            full_dataset: Complete HF dataset with all samples
            difficulty_groups: List of difficulty groups (list of (level, discipline) tuples)
            max_steps: Total training steps for scheduling
        """
        import random
        
        self.full_dataset = full_dataset
        self.difficulty_groups = difficulty_groups
        self.num_groups = len(difficulty_groups)
        self.max_steps = max_steps
        self.rng = random.Random(42)  # Fixed seed for reproducibility
        
        # Build reverse mapping: (level, discipline) → list of indices in full_dataset
        self.tuple_to_indices = {}
        for idx, sample in enumerate(full_dataset):
            key = (sample["level"], sample["discipline"])
            if key not in self.tuple_to_indices:
                self.tuple_to_indices[key] = []
            self.tuple_to_indices[key].append(idx)
        
        # Precompute indices for each cumulative difficulty level
        self.cumulative_indices = []
        for i in range(self.num_groups):
            # Include all groups up to and including group i
            indices = []
            for j in range(i + 1):
                for level, discipline in self.difficulty_groups[j]:
                    key = (level, discipline)
                    if key in self.tuple_to_indices:
                        indices.extend(self.tuple_to_indices[key])
            # Shuffle indices to ensure diverse sampling
            shuffled_indices = indices.copy()
            self.rng.shuffle(shuffled_indices)
            self.cumulative_indices.append(shuffled_indices)
        
        # Current difficulty level
        self.current_difficulty_level = 0
        self.current_indices = self.cumulative_indices[0]
        
        log_with_rank(f"📚 Curriculum Learning Initialized:")
        log_with_rank(f"   Total difficulty groups: {self.num_groups}")
        for i, group in enumerate(difficulty_groups):
            n_samples = len(self.cumulative_indices[i])
            log_with_rank(f"   Level {i+1}: {len(group)} tuples, {n_samples} cumulative samples")
    
    def update_difficulty(self, current_step):
        """
        Update available samples based on current training step.
        
        Curriculum schedule: Linear progression through difficulty levels
        """
        # Calculate which difficulty level should be active
        progress = current_step / self.max_steps
        new_level = min(int(progress * self.num_groups), self.num_groups - 1)
        
        if new_level != self.current_difficulty_level:
            self.current_difficulty_level = new_level
            self.current_indices = self.cumulative_indices[new_level]
            
            log_with_rank(f"📈 Curriculum Update at Step {current_step}:")
            log_with_rank(f"   Progress: {progress*100:.1f}%")
            log_with_rank(f"   Difficulty level: {new_level + 1}/{self.num_groups}")
            log_with_rank(f"   Available samples: {len(self.current_indices)}")
    
    def get_current_dataset(self):
        """Return a filtered dataset with only currently available samples"""
        return self.full_dataset.select(self.current_indices)
    
    def __len__(self):
        # Return full dataset length so dataloader doesn't need to be recreated
        # We filter samples dynamically in __getitem__
        return len(self.full_dataset)
    
    def __getitem__(self, idx):
        # Sample from currently available indices (curriculum-aware)
        # Use modulo to map any index to available samples
        curriculum_idx = idx % len(self.current_indices)
        full_idx = self.current_indices[curriculum_idx]
        return self.full_dataset[full_idx]


################
# Dataset
################

def parse_list_arg(arg: str, available: List[str]) -> List[str]:
    """Parse comma-separated string or 'all' into list
    
    Examples:
        "all" → all items
        "Math,Biology" → ["Math", "Biology"]
        "preschool" → ["preschool"]
    """
    arg = arg.strip()
    
    if arg.lower() == "all":
        return available
    
    if ',' in arg:
        items = [x.strip() for x in arg.split(',')]
        # Validate
        invalid = [x for x in items if x not in available]
        if invalid:
            raise ValueError(f"Invalid items: {invalid}. Available: {available}")
        return items
    
    # Single value
    if arg not in available:
        raise ValueError(f"Invalid item: {arg}. Available: {available}")
    return [arg]


def prepare_dataset(levels: str, disciplines: str, max_samples: int = 1000, local_data_path: str = "/mnt/ephemeral/MMMG_train/train.json"):
    """Load and format MMMG dataset for GRPO from local file
    
    Dataset structure:
    - Local JSON file with structure: {level_id: {discipline: [samples]}}
    - Each sample has 'question' (prompt), 'key', 'Visual Components' (knowledge graph), 'Key Knowledge' (annotation)
    
    We load all specified level/discipline combinations and format for GRPO.
    """
    
    levels_list = parse_list_arg(levels, LEVELS_CANONICAL)
    disciplines_list = parse_list_arg(disciplines, DISCIPLINES)
    
    log_with_rank(f"Loading MMMG from local file: {local_data_path}")
    log_with_rank(f"  Levels: {levels_list}")
    log_with_rank(f"  Disciplines: {disciplines_list}")
    
    # Load the local JSON file
    import json
    with open(local_data_path, 'r') as f:
        data = json.load(f)
    
    # Map canonical level names to JSON keys
    # JSON uses format: "0_preschool", "1_primaryschool", etc.
    level_to_json_key = {
        "preschool": "0_preschool",
        "primaryschool": "1_primaryschool",
        "secondaryschool": "2_secondaryschool",
        "highschool": "3_highschool",
        "undergraduate": "4_undergraduate",
        "PhD": "5_PhD",
    }
    
    # Collect all samples from specified configs
    all_samples = []
    
    for level in levels_list:
        json_key = level_to_json_key.get(level)
        if not json_key or json_key not in data:
            log_with_rank(f"Level {level} not found in data (looking for key: {json_key})", level="warning")
            continue
            
        level_data = data[json_key]
        
        for discipline in disciplines_list:
            if discipline not in level_data:
                log_with_rank(f"Discipline {discipline} not found in level {level}", level="warning")
                continue
            
            discipline_samples = level_data[discipline]
            log_with_rank(f"Loading {level}/{discipline}: {len(discipline_samples)} samples")
            
            # Convert to samples - extract knowledge graph and annotation
            for row in discipline_samples:
                prompt = row.get("question")  # Changed from 'prompt' to 'question'
                key = row.get("key")
                visual_components = row.get("Visual Components", {})
                key_knowledge = row.get("Key Knowledge", {})
                img_path = row.get("img_path", "")  # Ground truth image path
                
                if not prompt or not key:
                    continue
                
                # Convert Visual Components to knowledge graph JSON string
                knowledge_graph = json.dumps(visual_components)
                
                # Use Key Knowledge as annotation
                annotation = json.dumps(key_knowledge)
                
                all_samples.append({
                    "prompt_text": prompt,
                    "key": key,
                    "level": level,
                    "discipline": discipline,
                    "knowledge_graph": knowledge_graph,
                    "annotation": annotation,
                    "gt_img_path": img_path,  # Ground truth image path
                })
            
            log_with_rank(f"  Loaded {len(discipline_samples)} samples from {level}_{discipline}")
    
    log_with_rank(f"Total collected samples: {len(all_samples)}")
    
    if len(all_samples) == 0:
        raise ValueError("No samples loaded! Check your level/discipline settings.")
    
    # Format for GRPO
    # SYSTEM_PROMPT = (
    #     "You are an expert at generating detailed, accurate images based on text descriptions. "
    #     "Analyze the user's prompt carefully and create an image that precisely matches the requirements, "
    #     "paying attention to all specified details, objects, styles, colors, and compositions."
    # )
    SYSTEM_PROMPT = SYSTEM_TEMPLATE_PROMPT
    
    formatted_samples = []
    for sample in all_samples:
        formatted_samples.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["prompt_text"]},
            ],
            "prompt_text": sample["prompt_text"],
            "key": sample["key"],
            "level": sample["level"],
            "discipline": sample["discipline"],
            "knowledge_graph": sample["knowledge_graph"],
            "annotation": sample["annotation"],
            "gt_img_path": sample["gt_img_path"],  # Include ground truth image path
        })
    
    # Convert to HF dataset
    from datasets import Dataset
    dataset = Dataset.from_list(formatted_samples)
    
    log_with_rank(f"Dataset formatted: {len(dataset)} samples")
    
    # Limit samples
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        log_with_rank(f"Limited to {len(dataset)} samples")
    
    log_with_rank(f"Dataset ready: {len(dataset)} total samples")
    log_with_rank(f"  Levels: {levels_list}")
    log_with_rank(f"  Disciplines: {disciplines_list}")
    
    # Split into train and eval
    if len(dataset) > 100:
        train_size = len(dataset) - 100
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        log_with_rank(f"Split into train: {len(train_dataset)}, eval: {len(eval_dataset)}")
        return train_dataset, eval_dataset
    else:
        return dataset, None


################
# Reward Function - MMMG Evaluation Protocol
################

# MMMG Evaluation Prompt Template
MMMG_INSTRUCTION_PROMPT = '''
This evaluation is part of a research study on visual grounding of abstract concepts. No jailbreak or prompt injection is intended.

Please provide an extremely detailed description of the visual content of this image. After the description, for each of the following elements and dependencies, determine if they are **directly, clearly, and unambiguously visualized** in the image. Output "yes" or "no" for each. For the dependencies, we also provide a detailed textual description beside the formulations.

# Important Instructions:

* **Base your judgment solely on what is explicitly visible in the image.** Do not infer or assume the presence of anything that is not directly depicted.
* **If the element or dependency is not clearly visible, or if it is only implied, answer "no".**

* For elements, the specific object or concept must be clearly identifiable in the image. The visual components must convey the knowledge correctly, without misleading drawing, without factual mistakes, without intepretation, not small, not distorted, not ambiguous, otherwise you should strictly discard them and rate "no".

* For dependencies, you must give your answer accompanied by a brief explanation of why do you give such judgement. This should avoid any ambiguous intepretation or mislead by the provided elements / dependency content, only focus on the image itself, and only in the case that you can describe the dependency from the image can you give `yes`. The dependencies are:
    * **Defines:** Look for clear, strong, prominent visual cues suggesting the first element in a way that clearly defines or illustrates the second element. Any ambiguous or inferential patterns should lead to "no".
    * **Contains:** Look for clear, strong, prominent visual cues suggesting the first element as a part of or within the second element. Any ambiguous or inferential patterns should lead to "no".
    * **Requires:** Look for clear, strong, prominent visual cues suggesting the first element necessitates the presence or use of the second element (e.g., a boiler visibly connected to or interacting with a working fluid).
    * **Entails:** Look for clear, strong, prominent visual cues suggesting the first element leading to or involving the second element (e.g., a boiler clearly connected to a turbine).
    * **Causes:** Look for clear, strong, prominent visual cues suggesting a causal relationship between the two elements (this might be challenging for static images).
    * **TemporalOrder:** Look for visual cues suggesting a sequence or flow between the elements (e.g., pipes or connections implying a direction). If no clear visual cue for temporal order exists, answer "no".

* **Exclude any entity or dependency that is absent, unclear, or based on factual artifacts or external knowledge not directly shown.**
* For abstract concepts only answer "yes" if the key visual components and their interactions characteristic of these concepts are clearly and directly depicted.

The elements and dependencies are as follows, where there are no offensive or inappropriate elements, just educational ones:
[ELEM_DEPEND]

For the output format, please use the following structure:
**Image Description:**
[IMAGE_DESCRIPTION]
**Element and Dependency Analysis:**
** Element Evaluation: **
*   [ELEMENT_1]: [yes/no] 
*   [ELEMENT_2]: [yes/no]
...
** Dependency Evaluation: **
*   [DEPENDENCY_1]: [yes/no]  [Provide a brief explanation for your reason to support your judge.]
*   [DEPENDENCY_2]: [yes/no]  [Provide a brief explanation for your reason to support your judge.]
...
'''

def create_reward_function(
    gen_model_path: str,
    vlm_model,
    trainer_ref,
    vlm_processor,
    api_key: str,
    sam2_checkpoint: str,
    alignment_weight: float,
    quality_weight: float,
    n_evals: int,
    logging_steps: int = 1,
    training_args=None,
    img_args=None,
):
    """
    Create reward function using MMMG evaluation protocol that:
    1. Extracts embeddings from GRPO's completion text
    2. Generates images via diffusion
    3. Evaluates with GPT using MMMG knowledge graph evaluation
    
    PARALLELIZATION OPTIMIZATIONS (for 8x H200 GPUs):
    ====================================================
    
    1. **Multi-GPU Distributed Evaluation**:
       - 8 GPUs process batches independently in parallel
       - Each GPU: 16 samples/batch → 128 samples system-wide
    
    2. **Async GPT API Calls** (asyncio.gather):
       - Within each GPU: 16 parallel API calls (no restrictions)
       - System-wide: 8 GPUs × 16 = 128 parallel API calls
       - Fully parallel - GPT has no GPU memory impact
    
    3. **Serialized GPU SAM2 Processing** (MODERATE PARALLELIZATION):
       - SAM2 runs on GPU (fast) with ThreadPoolExecutor (1 worker) per GPU
       - Thread pool executor naturally serializes without deadlock risk
       - Only 1 SAM2 at a time per GPU (prevents OOM from memory conflicts)
       - System-wide: 8 GPUs × 1 SAM2 = 8 parallel SAM2 inferences
       - Within each sample: GPT + SAM2 still run concurrently
       - Across samples: GPT fully parallel, SAM2 queued
       - Benefits: Fast GPU inference without OOM, no deadlock
    
    4. **torch.distributed Wandb Gathering**:
       - In-memory data transfer vs file I/O (5-10x faster)
       - No disk bottleneck, no cleanup overhead
    
    5. **Dynamic Image Generation Batching**:
       - Adaptive batch sizing based on H200's 141GB VRAM
       - Maximizes throughput without OOM
    
    6. **Memory Optimizations**:
       - SAM2 serialized per GPU (prevents OOM)
       - Aggressive gc.collect() after each phase
       - Sub-batch processing for large batches
    
    CRITICAL FIXES:
    ===============
    - Fixed SAM2 torch compile error by clearing Hydra state
    - Serialized SAM2 with ThreadPoolExecutor (prevents OOM, maintains GPU speed, no deadlock)
    - Wrapped SAM2 init in torch.no_grad()
    
    PARALLELISM SUMMARY:
    ====================
    - 128 parallel GPT API calls (fully concurrent)
    - 8 parallel SAM2 inferences (1 per GPU, serialized within GPU)
    - Best of both: GPT fully parallel, SAM2 fast on GPU without OOM
    
    EXPECTED SPEEDUP: 2x faster wall-clock training time (vs sequential evaluation)
    - Bottleneck: GPT API latency (30-60s) + serialized SAM2 (2-3s each × 16 = 32-48s)
    - Total per batch: ~60-100s (same as semaphore but without deadlock)
    """
    
    # Lazy load
    gen_pipeline = None
    gpt_eval_fn = None
    
    # Step counter for custom image logging (separate from trainer's global_step)
    # This counter increments with each reward function call
    image_log_step = [0]
    
    def get_pipeline():
        nonlocal gen_pipeline
        if gen_pipeline is None:
            from diffusers import QwenImageResponsePipeline
            import torch.distributed as dist
            import gc
            
            # Get device for this process
            if dist.is_initialized():
                device = f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            log_with_rank(f"Loading pipeline on {device}", verbose_only=True)
            
            gen_pipeline = QwenImageResponsePipeline.from_pretrained(
                gen_model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            log_with_rank("Loading LoRA weights...", verbose_only=True)
            gen_pipeline.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
            )
            
            # MEMORY OPTIMIZATION: Free original text_encoder before replacement
            if hasattr(gen_pipeline, 'text_encoder') and gen_pipeline.text_encoder is not None:
                original_text_encoder = gen_pipeline.text_encoder
                del original_text_encoder
            
            # CRITICAL: Replace text_encoder with GRPO-trained VLM
            gen_pipeline.text_encoder = vlm_model[0]
            
            # MEMORY OPTIMIZATION: Enable memory-efficient attention mechanisms
            gen_pipeline.enable_attention_slicing(slice_size="auto")
            if hasattr(gen_pipeline, 'enable_vae_slicing'):
                gen_pipeline.enable_vae_slicing()
            
            # NOTE: We do NOT enable model_cpu_offload here because:
            # 1. text_encoder IS the training model (shared with GRPO)
            # 2. CPU offload would interfere with training forward/backward passes
            # 3. The model needs to stay on GPU for gradient updates
            
            # Clear cache after setup
            gc.collect()
            torch.cuda.empty_cache()
            
            log_with_rank("Pipeline loaded, text_encoder = trained VLM (memory optimized)", verbose_only=True)
        
        return gen_pipeline
    
    def get_gpt_evaluator():
        nonlocal gpt_eval_fn
        if gpt_eval_fn is None:
            from openai import AsyncOpenAI
            import json
            import re
            import asyncio
            import threading
            
            client = AsyncOpenAI(api_key=api_key)
            
            # DEBUG: Track active GPT API calls to diagnose hanging
            gpt_active_counter = threading.Lock()
            gpt_active_count = [0]  # Use list for mutability in nested functions
            
            # ========== MMMG Readability Evaluation (SAM2 + OCR) ==========
            # Initialize SAM2 and OCR for region counting
            import sys
            import pathlib
            import numpy as np
            import torch
            
            # Setup SAM2
            mmmg_eval_root = pathlib.Path("/home/coder/work/MMMG/mmmg_eval")
            sam2_root = mmmg_eval_root / "sam2" / "sam2"
            sys.path.insert(0, str(sam2_root))
            
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
            from paddleocr import PaddleOCR
            
            # Initialize models once
            sam2_model = None
            mask_generator = None
            ocr_engine = None
            
            def get_sam2_mask_generator():
                nonlocal sam2_model, mask_generator
                if mask_generator is None:
                    from hydra.core.global_hydra import GlobalHydra
                    from hydra import initialize_config_dir
                    
                    import torch.distributed as dist
                    if dist.is_initialized():
                        device = f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
                    else:
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    log_with_rank(f"Loading SAM2 on {device}...", verbose_only=True)
                    
                    # CRITICAL FIX: Properly manage Hydra state for SAM2
                    # Clear existing Hydra instance to avoid conflicts
                    if GlobalHydra.instance().is_initialized():
                        log_with_rank("Clearing existing Hydra instance...", verbose_only=True)
                        GlobalHydra.instance().clear()
                    
                    # Initialize Hydra with SAM2's config directory
                    # SAM2 root is already in sys.path (line 720)
                    sam2_config_dir = str(sam2_root / "configs")
                    log_with_rank(f"Initializing Hydra with config_dir: {sam2_config_dir}", verbose_only=True)
                    
                    try:
                        # Initialize Hydra with SAM2's config directory
                        with initialize_config_dir(config_dir=sam2_config_dir, version_base=None):
                            # Use relative path from config directory
                            model_cfg = "sam2.1/sam2.1_hiera_l.yaml"
                            
                            # Wrap in no_grad to prevent any gradient tracking
                            with torch.no_grad():
                                sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
                                mask_generator = SAM2AutomaticMaskGenerator(
                                    model=sam2_model,
                                    points_per_side=32,
                                    points_per_batch=256,
                                    pred_iou_thresh=0.7,
                                    stability_score_thresh=0.95,
                                    stability_score_offset=0.7,
                                    crop_n_layers=1,
                                    box_nms_thresh=0.6,
                                    crop_n_points_downscale_factor=2,
                                    min_mask_region_area=0.0,
                                    use_m2m=True,
                                )
                        
                        # Clear Hydra again after loading to prevent conflicts with future calls
                        if GlobalHydra.instance().is_initialized():
                            GlobalHydra.instance().clear()
                        
                        log_with_rank("SAM2 loaded successfully", verbose_only=True)
                    except Exception as e:
                        log_with_rank(f"Failed to load SAM2 with Hydra context: {e}", level="error")
                        # Clean up on error
                        if GlobalHydra.instance().is_initialized():
                            GlobalHydra.instance().clear()
                        raise
                
                return mask_generator
            
            def get_ocr_engine():
                nonlocal ocr_engine
                if ocr_engine is None:
                    log_with_rank("Loading PaddleOCR (CPU mode to avoid CUDNN issues)...", verbose_only=True)
                    # Use CPU mode to avoid CUDNN version mismatch errors
                    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, use_gpu=False)
                    log_with_rank("PaddleOCR loaded successfully", verbose_only=True)
                return ocr_engine
            
            # MODERATE PARALLELIZATION: Serialize SAM2 per GPU to prevent OOM
            # - SAM2 runs on GPU (fast) but only 1 at a time per GPU (prevents memory conflicts)
            # - GPT API calls remain fully parallel (no GPU memory impact)
            # - System-wide: 8 GPUs × 1 SAM2 = 8 parallel SAM2 inferences
            # - System-wide: 8 GPUs × 16 GPT calls = 128 parallel GPT API calls
            # - ThreadPoolExecutor with 1 worker naturally queues SAM2 calls without deadlock
            from concurrent.futures import ThreadPoolExecutor
            sam2_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SAM2")
            log_with_rank("SAM2 ThreadPoolExecutor initialized (1 worker per GPU)", verbose_only=True)
            
            def call_ppocr(image_pil, confidence_threshold=0.85, box_short_edge_thresh=20, line_merge_y_thresh=20):
                """
                Perform OCR using PaddleOCR and return merged line-level text boxes.
                Follows MMMG step2_readability.py lines 26-108
                """
                ocr = get_ocr_engine()
                result = ocr.ocr(np.array(image_pil), cls=True)
                
                filtered_results = []
                if result is None:
                    return [], []
                
                for line in result:
                    if line is None:
                        continue
                    for res in line:
                        box = res[0]
                        text = res[1][0]
                        conf = res[1][1]
                        
                        edge_lengths = [
                            np.linalg.norm(np.array(box[i]) - np.array(box[(i + 1) % 4]))
                            for i in range(4)
                        ]
                        min_edge = min(edge_lengths)
                        
                        if conf >= confidence_threshold and min_edge >= box_short_edge_thresh:
                            y_center = np.mean([pt[1] for pt in box])
                            filtered_results.append({
                                "text": text,
                                "box": box,
                                "y_center": y_center
                            })
                
                # Group lines by vertical center
                filtered_results.sort(key=lambda x: x["y_center"])
                merged_lines = []
                current_line = []
                current_y = None
                
                for item in filtered_results:
                    if current_y is None or abs(item["y_center"] - current_y) <= line_merge_y_thresh:
                        current_line.append(item)
                        current_y = item["y_center"] if current_y is None else (current_y + item["y_center"]) / 2
                    else:
                        merged_lines.append(current_line)
                        current_line = [item]
                        current_y = item["y_center"]
                if current_line:
                    merged_lines.append(current_line)
                
                # Convert to bounding boxes
                final_boxes = []
                for line in merged_lines:
                    all_points = np.concatenate([x["box"] for x in line], axis=0)
                    x1 = int(np.min(all_points[:, 0]))
                    y1 = int(np.min(all_points[:, 1]))
                    x2 = int(np.max(all_points[:, 0]))
                    y2 = int(np.max(all_points[:, 1]))
                    final_boxes.append((x1, y1, x2, y2))
                
                return final_boxes
            
            def merge_sam_masks_with_ocr(sam_masks, ocr_boxes, image_size, iou_thresh=0.8, min_side=10):
                """
                Merge SAM masks with OCR text regions using IoU.
                Follows MMMG step2_readability.py lines 112-171
                """
                def compute_iou(box1, box2):
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])
                    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                    if inter_area == 0:
                        return 0.0
                    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    union_area = min(box1_area, box2_area)
                    return inter_area / union_area
                
                final_masks = []
                for mask in sam_masks:
                    seg = mask['segmentation']
                    y_indices, x_indices = np.where(seg)
                    if len(x_indices) == 0 or len(y_indices) == 0:
                        continue
                    box_mask = [min(x_indices), min(y_indices), max(x_indices), max(y_indices)]
                    
                    overlaps_ocr = any(compute_iou(box_mask, ocr_box) > iou_thresh for ocr_box in ocr_boxes)
                    if not overlaps_ocr:
                        final_masks.append(mask)
                
                # Add OCR boxes as masks
                for ocr_box in ocr_boxes:
                    x1, y1, x2, y2 = map(int, ocr_box)
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width < min_side or height < min_side:
                        continue
                    
                    mask_array = np.zeros((image_size[1], image_size[0]), dtype=bool)
                    mask_array[y1:y2+1, x1:x2+1] = True
                    area = (y2 - y1 + 1) * (x2 - x1 + 1)
                    
                    final_masks.append({
                        'segmentation': mask_array,
                        'area': area,
                        'source': 'ocr'
                    })
                
                return final_masks
            
            def calc_weighting(region_count):
                """
                Calculate weighting based on region count.
                Follows MMMG step3_stat.py lines 158-163
                """
                if region_count <= 70:
                    return 1.0
                else:
                    return max((160 - region_count) / 90, 0.0)
            
            def evaluate_readability(image_pil):
                """
                Evaluate image readability using SAM2 + OCR to count visual regions.
                Follows MMMG step2_readability.py logic.
                
                Returns:
                    region_count: Number of visual regions
                    R_score: Weighting based on region count
                """
                try:
                    image_width, image_height = image_pil.size
                    
                    # Generate SAM2 masks
                    mask_gen = get_sam2_mask_generator()
                    masks = mask_gen.generate(np.array(image_pil))
                    
                    # Run OCR
                    ocr_boxes = call_ppocr(image_pil)
                    
                    # Merge masks with OCR
                    final_masks = merge_sam_masks_with_ocr(masks, ocr_boxes, (image_width, image_height))
                    
                    region_count = len(final_masks)
                    R_score = calc_weighting(region_count)
                    
                    return region_count, R_score
                    
                except Exception as e:
                    log_with_rank(f"Readability evaluation error: {e}", level="error")
                    # Default values if evaluation fails
                    return 0, 0.0

            def clean_key(k):
                """Clean element/dependency key - from MMMG stat_knowledge.py lines 10-16"""
                s = k.strip()
                s = re.sub(r"^[\*`]+", "", s)   
                s = re.sub(r"[\*`]+$", "", s)   
                s = re.sub(r"[:：]$", "", s)     
                return s.strip()
            
            def parse_mmmg_response(text, elements, dependencies):
                """
                Parse GPT response to extract yes/no for elements and dependencies.
                Exact implementation from MMMG stat_knowledge.py lines 18-65.
                
                Args:
                    text: GPT response text
                    elements: List of element names from ground truth
                    dependencies: List of dependency names from ground truth
                
                Returns:
                    element_dict: Dict[str, bool] - element presence
                    dependency_dict: Dict[str, bool] - dependency presence
                """
                all_keys = elements + dependencies
                
                escaped_keys = [re.escape(k) for k in all_keys]
                
                # Clean text (MMMG lines 23-25)
                text = text.replace("**", "")
                text = text.replace("[yes]", "yes")
                text = text.replace("[no]", "no")
                
                # Regex pattern (MMMG line 26)
                pattern = rf"^\s*(?:[*\-•]|\d+\.)?\s*`*\s*({'|'.join(escaped_keys)})`*\s*[:：]\s*(yes|no|[yes]|[no]|YES|NO|Yes|No)\b"
                matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
                
                # If no matches, return all False (MMMG lines 29-34)
                if len(matches) == 0:
                    return {key: False for key in elements}, {key: False for key in dependencies}
                
                # Initialize dicts (MMMG lines 36-37)
                element_dict = {key: False for key in elements}
                dependency_dict = {key: False for key in dependencies}
                
                # Parse matches (MMMG lines 39-58)
                for raw_k, v in matches:
                    k_clean = clean_key(raw_k)
                    v_bool = v.strip().lower() == "yes"
                    
                    if k_clean in element_dict:
                        element_dict[k_clean] = v_bool
                    elif k_clean in dependency_dict:
                        dependency_dict[k_clean] = v_bool
                    else:
                        # Fuzzy matching with case-insensitive comparison
                        find = False
                        for ele in element_dict.keys():
                            if k_clean.lower() == ele.lower():
                                element_dict[ele] = v_bool
                                find = True
                                break
                        if not find:
                            for dep in dependency_dict.keys():
                                if k_clean.lower() == dep.lower():
                                    dependency_dict[dep] = v_bool
                                    find = True
                                    break
                
                return element_dict, dependency_dict
            
            def parse_dependencies(dependencies):
                """
                Parse dependencies from MMMG format into edges.
                Follows MMMG step3_stat.py logic exactly (lines 12-75).
                """
                edges = []
                for dep, exists in dependencies.items():
                    if exists:
                        try:
                            # Parse "Relation(source, target)"
                            relation, nodes = dep.split('(', 1)
                            
                            # Handle closing bracket
                            try:
                                if nodes[-1] == ")":
                                    nodes = nodes[:-1]
                                else:
                                    log_with_rank(f"Error parsing dependency BRACKETS: {dep}")
                                
                                # Try splitting with ", " first
                                source, target = nodes.split(', ', 1)
                            except ValueError:
                                # Fallback: try splitting with "," only
                                if len(nodes.split(', ', 1)) == 1 and len(nodes.split(',', 1)) == 2:
                                    source, target = nodes.split(',', 1)
                                else:
                                    log_with_rank(f"Error parsing dependency: {dep}")
                                    continue
                            
                            source = source.lower()
                            target = target.lower()
                            
                            # Handle change() wrapper (MMMG lines 46-49)
                            if "change(" in source.lower():
                                source = source.strip(")").split("change(")[-1].lower()
                            if "change(" in target.lower():
                                target = target.strip(")").split("change(")[-1].lower()
                            
                            # Handle different relation types (MMMG lines 51-72)
                            lower_relation = relation.lower()
                            if lower_relation == "requires":
                                # Requires: edge goes from target to source
                                if isinstance(target, list) or isinstance(target, tuple):
                                    for t in target:
                                        edges.append((t, source, relation))
                                else:
                                    edges.append((target, source, relation))
                            elif lower_relation == "defines":
                                # Defines: bidirectional edge
                                if isinstance(target, list) or isinstance(target, tuple):
                                    for t in target:
                                        edges.append((source, t, relation))
                                        edges.append((t, source, relation))
                                else:
                                    edges.append((source, target, relation))
                                    edges.append((target, source, relation))
                            else:
                                # Causes, Entails, TemporalOrder, Contains: source to target
                                if isinstance(target, list) or isinstance(target, tuple):
                                    for t in target:
                                        edges.append((source, t, relation))
                                else:
                                    edges.append((source, target, relation))
                        except ValueError:
                            continue  # Skip malformed items
                return edges
            
            def build_graph(elements, dependencies):
                """
                Build networkx graph from elements and dependencies.
                Follows MMMG step3_stat.py logic exactly (lines 78-121).
                """
                import networkx as nx
                
                G = nx.DiGraph()
                
                # Add nodes (only elements that exist)
                for node, exists in elements.items():
                    if exists:
                        G.add_node(node.lower())
                
                # Parse and add edges
                edges = parse_dependencies(dependencies)
                
                # Add edges with fuzzy matching (same as MMMG lines 86-119)
                for source, target, relation in edges:
                    match_src = False
                    match_tgt = False
                    
                    # Find candidates for fuzzy matching
                    src_candidate = [s for s in G.nodes if source in s and source != s] + \
                                   [s for s in G.nodes if s in source and s != source and " " in source]
                    tgt_candidate = [t for t in G.nodes if target in t and target != t] + \
                                   [t for t in G.nodes if t in target and t != target and " " in target]
                    
                    # Filter candidates (avoid false matches)
                    src_candidate = [s for s in src_candidate if "" in s]
                    tgt_candidate = [t for t in tgt_candidate if "" in t]
                    
                    # First try exact matching
                    for node in G.nodes:
                        if source == node:
                            match_src = True
                        if target == node:
                            match_tgt = True
                    
                    # If no exact match, try fuzzy matching
                    if not match_src:
                        if len(src_candidate) == 1:
                            match_src = True
                            source = src_candidate[0]
                    if not match_tgt:
                        if len(tgt_candidate) == 1:
                            match_tgt = True
                            target = tgt_candidate[0]
                    
                    # Add edge if both nodes matched
                    if match_src and match_tgt:
                        G.add_edge(source, target, label=relation)
                
                return G
            
            def compute_graph_edit_distance(predicted_elements, predicted_dependencies, gt_elements, gt_dependencies):
                """
                Compute normalized Graph Edit Distance using networkx.
                Follows MMMG step3_stat.py exactly: lines 193-196, 123-130, 198-202
                
                Args:
                    predicted_elements: Dict[str, bool] - Predicted element presence (from GPT parsing)
                    predicted_dependencies: Dict[str, bool] - Predicted dependency presence (from GPT parsing)
                    gt_elements: List[str] - List of ground truth element names
                    gt_dependencies: List[str] - List of ground truth dependency names
                
                Returns:
                    knowledge_fidelity: float in [0, 1], where 1 = perfect match, 0 = completely wrong
                    G_gt: Ground truth graph (for logging)
                    G_pred: Predicted graph (for logging)
                """
                import networkx as nx
                
                # Build GT graph with ALL elements/dependencies set to True
                # This follows MMMG step3_stat.py lines 193-195:
                #   all_elements = {n: True for n in branch_data["elements"].keys()}
                #   all_dependencies = {dep: True for dep in branch_data["dependencies"].keys()}
                #   G_gt = build_graph(all_elements, all_dependencies)
                all_elements_gt = {elem: True for elem in gt_elements}
                all_dependencies_gt = {dep: True for dep in gt_dependencies}
                G_gt = build_graph(all_elements_gt, all_dependencies_gt)
                
                # Build predicted graph with actual True/False values from GPT
                # This follows MMMG step3_stat.py line 196:
                #   G_data = build_graph(branch_data["elements"], branch_data["dependencies"])
                G_pred = build_graph(predicted_elements, predicted_dependencies)
                
                # Compute normalized GED (exact copy of MMMG lines 123-130)
                try:
                    ged = next(nx.optimize_graph_edit_distance(G_gt, G_pred))
                except StopIteration:
                    ged = 0  # No distance if graphs are identical
                
                max_size = (G_gt.number_of_nodes() + G_pred.number_of_nodes() + 
                           G_gt.number_of_edges() + G_pred.number_of_edges())
                
                # IMPORTANT: Match MMMG line 130 exactly - return 1.0 if max_size == 0
                normalized_ged = ged / max_size if max_size > 0 else 1.0
                
                # Knowledge fidelity = 1 - GED (same as MMMG line 202)
                knowledge_fidelity = 1.0 - normalized_ged
                
                return knowledge_fidelity, G_gt, G_pred
            
            def serialize_graph(G):
                """Convert networkx graph to a readable text format for logging"""
                lines = []
                lines.append(f"Nodes ({G.number_of_nodes()}): {sorted(G.nodes())}")
                if G.number_of_edges() > 0:
                    edges_with_labels = []
                    for u, v, data in G.edges(data=True):
                        label = data.get('label', '')
                        edges_with_labels.append(f"{label}({u}, {v})")
                    lines.append(f"Edges ({G.number_of_edges()}): {edges_with_labels}")
                else:
                    lines.append(f"Edges (0): []")
                return "\n".join(lines)
            
            def save_generated_images(images, prompts, completions, keys, rewards, knowledge_fidelities, region_counts, R_scores, gt_graphs, pred_graphs, gt_images, current_step, levels=None, disciplines=None, advantages=None):
                """Save generated images with metadata and log to wandb (synchronous, following TRL pattern)"""
                import json
                from datetime import datetime
                
                # Ensure levels and disciplines are lists
                levels_list = levels if levels else [None] * len(images)
                disciplines_list = disciplines if disciplines else [None] * len(images)
                advantages_list = advantages if advantages else [0.0] * len(images)
                
                # Get wandb config from training_args
                wandb_log_images = img_args.wandb_log_images
                wandb_image_size = img_args.wandb_image_size
                
                # Create output directory
                output_dir = Path(training_args.output_dir) / "generated_images"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get current rank
                try:
                    from torch.distributed import get_rank, is_initialized
                    if is_initialized():
                        current_rank = get_rank()
                    else:
                        current_rank = 0
                except Exception:
                    current_rank = 0
                
                rank = current_rank
                
                # Use the passed current_step directly for image logging
                # This is a separate counter from trainer's global_step to avoid conflicts
                training_step = current_step
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_dir = output_dir / f"step_{timestamp}_rank{rank}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                
                use_wandb = training_args.report_to and "wandb" in training_args.report_to
                should_log = use_wandb and (training_step % logging_steps == 0)
                
                # Save each image with metadata to disk
                for i, (img, prompt, completion, key, reward, kf, rc, rs, gt_graph, pred_graph, gt_img, level, discipline, advantage) in enumerate(
                    zip(images, prompts, completions, keys or [None]*len(images), 
                        rewards, knowledge_fidelities, region_counts, R_scores, gt_graphs, pred_graphs, gt_images,
                        levels_list, disciplines_list, advantages_list)
                ):
                    # Save generated image to disk
                    img_filename = f"image_{i:03d}.png"
                    img_path = batch_dir / img_filename
                    img.save(img_path)
                    
                    # Save ground truth image if available
                    gt_img_filename = None
                    if gt_img is not None:
                        gt_img_filename = f"gt_image_{i:03d}.png"
                        gt_img_path = batch_dir / gt_img_filename
                        gt_img.save(gt_img_path)
                    
                    # Save metadata
                    metadata = {
                        "image_filename": img_filename,
                        "gt_image_filename": gt_img_filename,
                        "key": key,
                        "prompt": prompt,
                        "completion": completion,
                        "level": level,
                        "discipline": discipline,
                        "rewards": {
                            "knowledge_fidelity": float(kf),
                            "region_count": int(rc),
                            "R_score": float(rs),
                            "k_score_w": float(reward),
                            "final_reward": float(reward),
                            "advantage": float(advantage),
                        },
                        "graphs": {
                            "ground_truth": gt_graph,
                            "predicted": pred_graph,
                        },
                        "timestamp": timestamp,
                    }
                    
                    metadata_filename = f"image_{i:03d}.json"
                    metadata_path = batch_dir / metadata_filename
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                # Save batch summary
                summary = {
                    "timestamp": timestamp,
                    "rank": rank,
                    "num_images": len(images),
                    "average_knowledge_fidelity": float(sum(knowledge_fidelities) / len(knowledge_fidelities)),
                    "average_region_count": float(sum(region_counts) / len(region_counts)),
                    "average_R_score": float(sum(R_scores) / len(R_scores)),
                    "average_reward": float(sum(rewards) / len(rewards)),
                    "reward_std": float(torch.tensor(rewards).std().item()) if len(rewards) > 1 else 0.0,
                }
                
                summary_path = batch_dir / "batch_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                log_with_rank(f"[Step {training_step}] Saved {len(images)} generated images to {batch_dir}", verbose_only=True)
                
                # Helper function to resize image
                def resize_image(img, max_size):
                    """Resize image to max dimension while maintaining aspect ratio"""
                    from PIL import Image as PILImage
                    if img.width <= max_size and img.height <= max_size:
                        return img
                    ratio = min(max_size / img.width, max_size / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    return img.resize(new_size, PILImage.Resampling.LANCZOS)
                
                # Helper function to select top/bottom samples by reward
                def sample_images_by_reward(n_samples):
                    """Select top N and bottom N samples by reward for logging (robust version)"""
                    if n_samples == 0 or len(rewards) == 0:
                        return []
                    
                    if len(rewards) <= n_samples * 2:
                        return list(range(len(rewards)))
                    
                    # Filter out failed samples (reward == 0.0) for selection
                    valid_indices = [i for i, r in enumerate(rewards) if r > 0.0]
                    
                    if len(valid_indices) == 0:
                        # All failed - just return first n_samples
                        return list(range(min(n_samples, len(rewards))))
                    
                    if len(valid_indices) <= n_samples * 2:
                        return sorted(valid_indices)
                    
                    # Get indices sorted by reward (only valid ones)
                    sorted_valid = sorted(valid_indices, key=lambda i: rewards[i])
                    
                    # Take bottom N (worst) and top N (best)
                    selected = sorted_valid[:n_samples] + sorted_valid[-n_samples:]
                    return sorted(selected)  # Sort to maintain order
                
                # ========== PREPARE DATA FOR CALLBACK LOGGING (No gather_object here!) ==========
                # Store data for ImageLoggingCallback to process later in on_step_end()
                # where all ranks are synchronized by the trainer
                if should_log and use_wandb and wandb_log_images > 0:
                    from PIL import Image as PILImage
                    import io
                    
                    # Sample top/bottom images by reward (all ranks do this for their data)
                    selected_indices = sample_images_by_reward(wandb_log_images)
                    
                    # Guard: Check if we have any samples to log
                    if len(selected_indices) == 0:
                        log_with_rank("No samples to log to wandb (empty batch or all failed)", level="warning", verbose_only=True)
                        rank_data = None
                    else:
                        log_with_rank(f"Preparing {len(selected_indices)} sampled images for wandb...", verbose_only=True)
                        
                        # Prepare sampled data with resized images (all ranks prepare their data)
                        sampled_data = []
                        for idx in selected_indices:
                            # Resize and compress image
                            resized_img = resize_image(images[idx], wandb_image_size)
                            
                            # Convert to JPEG bytes for faster upload
                            img_buffer = io.BytesIO()
                            resized_img.convert('RGB').save(img_buffer, format='JPEG', quality=85, optimize=True)
                            img_buffer.seek(0)
                            img_pil = PILImage.open(img_buffer)
                            img_pil.load()  # Force load image data into memory before buffer is closed
                            
                            # Resize GT image if available
                            gt_img_pil = None
                            if gt_images[idx] is not None:
                                resized_gt = resize_image(gt_images[idx], wandb_image_size)
                                gt_buffer = io.BytesIO()
                                resized_gt.convert('RGB').save(gt_buffer, format='JPEG', quality=85, optimize=True)
                                gt_buffer.seek(0)
                                gt_img_pil = PILImage.open(gt_buffer)
                                gt_img_pil.load()  # Force load image data into memory before buffer is closed
                            
                            sampled_data.append({
                                "idx": idx,
                                "image": img_pil,
                                "gt_image": gt_img_pil,
                                "prompt": prompts[idx], #[:200] + "..." if len(prompts[idx]) > 200 else prompts[idx],
                                "completion": completions[idx], #[:200] + "..." if len(completions[idx]) > 200 else completions[idx],
                                "key": keys[idx] if keys and keys[idx] else "N/A",
                                "level": levels_list[idx] if levels_list[idx] else "unknown",
                                "discipline": disciplines_list[idx] if disciplines_list[idx] else "unknown",
                                "knowledge_fidelity": float(knowledge_fidelities[idx]),
                                "region_count": int(region_counts[idx]),
                                "R_score": float(R_scores[idx]),
                                "reward": float(rewards[idx]),
                                "advantage": float(advantages_list[idx]),
                                "gt_graph": gt_graphs[idx],
                                "pred_graph": pred_graphs[idx],
                            })
                        
                        # Serialize sampled data with safe statistics
                        rank_data = {
                            "rank": rank,
                            "sampled_data": sampled_data,
                            "summary_stats": {
                                "num_total": len(images),
                                "num_sampled": len(selected_indices),
                                "avg_reward": float(sum(rewards) / len(rewards)) if len(rewards) > 0 else 0.0,
                                "avg_knowledge_fidelity": float(sum(knowledge_fidelities) / len(knowledge_fidelities)) if len(knowledge_fidelities) > 0 else 0.0,
                                "avg_region_count": float(sum(region_counts) / len(region_counts)) if len(region_counts) > 0 else 0.0,
                                "avg_R_score": float(sum(R_scores) / len(R_scores)) if len(R_scores) > 0 else 0.0,
                                "reward_std": float(torch.tensor(rewards).std().item()) if len(rewards) > 1 else 0.0,
                            },
                        }
                    
                    # Store data for callback to process later (NO gather_object here - avoids deadlock!)
                    # The callback will call gather_object in on_step_end() when ranks are synchronized
                    if trainer_ref and trainer_ref[0] and hasattr(trainer_ref[0], 'image_logging_callback'):
                        # DEBUG: Log what step we're storing with and what trainer's step is NOW
                        current_trainer_step = trainer_ref[0].state.global_step if trainer_ref[0] and hasattr(trainer_ref[0], 'state') else -1
                        
                        trainer_ref[0].image_logging_callback.store_data_for_step(training_step, rank_data)
                        
                        log_with_rank(f"📦 STORED {len(rank_data['sampled_data']) if rank_data and 'sampled_data' in rank_data else 0} samples")
                        log_with_rank(f"   Storage key: step={training_step}")
                        log_with_rank(f"   Trainer global_step NOW: {current_trainer_step}")
                        log_with_rank(f"   ⚠️ MISMATCH DETECTED!" if training_step != current_trainer_step else f"   ✓ Match confirmed", verbose_only=False if training_step != current_trainer_step else True)
                    else:
                        log_with_rank("Warning: image_logging_callback not found on trainer", level="warning", verbose_only=True)
                
            
            async def evaluate_single_sample_async(idx, gen_img, kg_str, annotation, user_prompt, gt_img_path, level, discipline):
                """Evaluate a single sample asynchronously"""
                import base64
                from io import BytesIO
                from PIL import Image
                import time
                
                sample_start_time = time.time()
                log_with_rank(f"   📊 Evaluating sample {idx+1} (async)...", verbose_only=True)
                
                # Load ground truth image if path is provided
                gt_img = None
                if gt_img_path:
                    try:
                        log_with_rank(f"      🔍 Attempting to load GT image: {gt_img_path}", verbose_only=True)
                        
                        from pathlib import Path
                        if not Path(gt_img_path).is_absolute() or not Path(gt_img_path).exists():
                            relative_path = gt_img_path.lstrip('/')
                            gt_img_path = f"/mnt/ephemeral/MMMG_train/{relative_path}"
                        
                        gt_img = Image.open(gt_img_path).convert("RGB")
                        log_with_rank(f"      ✅ Loaded GT image: {gt_img_path}", verbose_only=True)
                    except Exception as e:
                        log_with_rank(f"Failed to load GT image {gt_img_path}: {e}", level="warning", verbose_only=True)
                        log_with_rank(f"      ❌ Failed to load GT image {gt_img_path}: {e}", verbose_only=True)
                        gt_img = None
                else:
                    log_with_rank(f"      ⚠️ No GT image path provided for sample {idx+1}", verbose_only=True)
                
                # Convert image to base64
                def img_to_b64(img):
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    return base64.b64encode(buffered.getvalue()).decode()
                
                gen_b64 = img_to_b64(gen_img)
                
                try:
                    # Parse ground truth knowledge graph
                    kg = json.loads(kg_str)
                    gt_elements_data = kg.get("elements", {})
                    gt_dependencies_data = kg.get("dependencies", {})
                    
                    if isinstance(gt_elements_data, dict):
                        gt_elements = list(gt_elements_data.keys())
                    else:
                        gt_elements = gt_elements_data if isinstance(gt_elements_data, list) else []
                    
                    if isinstance(gt_dependencies_data, dict):
                        gt_dependencies = list(gt_dependencies_data.keys())
                    else:
                        gt_dependencies = gt_dependencies_data if isinstance(gt_dependencies_data, list) else []
                    
                    # ========== CONCURRENT EVALUATION: GPT + Readability ==========
                    elem_depend_text = str(kg) + "\n" + str(annotation)
                    full_prompt = MMMG_INSTRUCTION_PROMPT.replace("[ELEM_DEPEND]", elem_depend_text)
                    
                    log_with_rank("      🔍 Launching GPT evaluation (parallel) + SAM2 readability (serialized)...", verbose_only=True)
                    
                    # DEBUG: Log before GPT API call
                    from datetime import datetime
                    gpt_start_time = time.time()
                    log_with_rank(f"      ⏰ Sample {idx+1}: BEFORE GPT API call at {datetime.now().strftime('%d/%m/%y %H:%M:%S')}", verbose_only=True)
                    
                    # DEBUG: Track active GPT calls
                    with gpt_active_counter:
                        gpt_active_count[0] += 1
                        current_count = gpt_active_count[0]
                    log_with_rank(f"      📈 Sample {idx+1}: Starting GPT call (active: {current_count})", verbose_only=True)
                    
                    # Launch GPT API call (fully parallel across all samples)
                    kg_task = client.responses.create(
                        input=[{
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": full_prompt},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{gen_b64}"
                                }
                            ]
                        }],
                        model="o3",
                        reasoning={"effort": "high"}
                    )
                    
                    # DEBUG: Log after GPT task creation
                    gpt_submit_time = time.time() - gpt_start_time
                    log_with_rank(f"      🚀 Sample {idx+1}: GPT task CREATED at {datetime.now().strftime('%d/%m/%y %H:%M:%S')} (submit: {gpt_submit_time:.3f}s)", verbose_only=True)
                    
                    # MODERATE PARALLELIZATION: Serialize SAM2 per GPU using ThreadPoolExecutor
                    # - ThreadPoolExecutor with 1 worker naturally queues SAM2 calls
                    # - Prevents OOM from multiple concurrent SAM2 inferences on same GPU
                    # - GPT API calls remain fully parallel (no memory impact)
                    # - More robust than semaphore (no deadlock risk)
                    # - Within each sample: GPT + SAM2 still run concurrently
                    # - Across samples: GPT fully parallel, SAM2 queued (only 1 at a time per GPU)
                    sam2_start_time = time.time()
                    log_with_rank(f"      🔧 Sample {idx+1}: Submitting SAM2 to thread pool...", verbose_only=True)
                    loop = asyncio.get_event_loop()
                    readability_task = loop.run_in_executor(sam2_executor, evaluate_readability, gen_img)
                    sam2_submit_time = time.time() - sam2_start_time
                    
                    # DEBUG: Log before awaiting gather
                    log_with_rank(f"      ⏳ Sample {idx+1}: AWAITING gather(kg_task, readability_task)...", verbose_only=True)
                    
                    # Wait for both to complete (GPT parallel, SAM2 serialized per GPU)
                    gather_start_time = time.time()
                    kg_response, (region_count, R_score) = await asyncio.gather(kg_task, readability_task)
                    gather_time = time.time() - gather_start_time
                    
                    # DEBUG: Log after gather completes with timing breakdown
                    total_sample_time = time.time() - sample_start_time
                    log_with_rank(f"      ✅ Sample {idx+1}: GATHER COMPLETE at {datetime.now().strftime('%d/%m/%y %H:%M:%S')}", verbose_only=True)
                    log_with_rank(f"      ⏱️ Sample {idx+1} timing: submit={gpt_submit_time+sam2_submit_time:.3f}s, gather={gather_time:.2f}s, total={total_sample_time:.2f}s", verbose_only=True)
                    
                    # DEBUG: Decrement active GPT counter
                    with gpt_active_counter:
                        gpt_active_count[0] -= 1
                        current_count = gpt_active_count[0]
                    log_with_rank(f"      📉 Sample {idx+1}: GPT completed (active: {current_count})", verbose_only=True)
                    
                    log_with_rank("      ✅ Both evaluations completed concurrently!", verbose_only=True)
                    
                    # Process GPT response
                    class Message:
                        def __init__(self, content):
                            self.content = content

                    class Choice:
                        def __init__(self, message):
                            self.message = message

                    class FakeResponse:
                        def __init__(self, output_text):
                            self.choices = [Choice(Message(output_text))]

                    fake_response = FakeResponse(kg_response.output_text)
                    kg_response_text = fake_response.choices[0].message.content
                    
                    # Parse predicted knowledge graph
                    pred_elements, pred_dependencies = parse_mmmg_response(kg_response_text, gt_elements, gt_dependencies)
                    
                    # Debug logging
                    log_with_rank(f"         GT elements: {len(gt_elements)} total", verbose_only=True)
                    log_with_rank(f"         GT dependencies: {len(gt_dependencies)} total", verbose_only=True)
                    pred_elem_count = sum(pred_elements.values())
                    pred_dep_count = sum(pred_dependencies.values())
                    log_with_rank(f"         Predicted elements: {pred_elem_count}/{len(gt_elements)} marked as present", verbose_only=True)
                    log_with_rank(f"         Predicted dependencies: {pred_dep_count}/{len(gt_dependencies)} marked as present", verbose_only=True)
                    
                    # Compute Graph Edit Distance → Knowledge Fidelity
                    knowledge_fidelity, G_gt, G_pred = compute_graph_edit_distance(
                        pred_elements, pred_dependencies,
                        gt_elements, gt_dependencies
                    )
                    
                    gt_graph_str = serialize_graph(G_gt)
                    pred_graph_str = serialize_graph(G_pred)
                    
                    log_with_rank(f"         ✓ Knowledge Fidelity: {knowledge_fidelity:.3f}", verbose_only=True)
                    log_with_rank(f"         ✓ Region Count: {region_count}, R-score: {R_score:.3f}", verbose_only=True)
                    
                    # ========== MMMG Reward Formula: k_score_w = R * (1 - GED) ==========
                    reward = R_score * knowledge_fidelity
                    
                    # Summary logging
                    log_with_rank(f"      ✅ Sample {idx+1}: KF={knowledge_fidelity:.3f}, Regions={region_count}, R={R_score:.3f}, k_score_w={reward:.3f}", verbose_only=True)
                    log_with_rank(f"MMMG Eval - Knowledge Fidelity: {knowledge_fidelity:.3f}, Region Count: {region_count}, R-score: {R_score:.3f}, k_score_w: {reward:.3f}", verbose_only=True)
                    
                    # Clean up intermediate data
                    del gen_b64, kg_response, kg_response_text
                    del pred_elements, pred_dependencies, G_gt, G_pred
                    
                    return {
                        'reward': reward,
                        'knowledge_fidelity': knowledge_fidelity,
                        'region_count': region_count,
                        'R_score': R_score,
                        'gt_graph': gt_graph_str,
                        'pred_graph': pred_graph_str,
                        'gt_image': gt_img,
                        'success': True
                    }
                    
                except Exception as e:
                    log_with_rank(f"      ❌ Sample {idx+1} failed: {e}", verbose_only=True)
                    log_with_rank(f"      📍 Error type: {type(e).__name__}", verbose_only=True)
                    log_with_rank(f"      📍 Error details: {str(e)[:500]}", verbose_only=True)
                    log_with_rank(f"MMMG eval error: {e}", level="error")
                    log_with_rank(f"Error type: {type(e).__name__}", level="error")
                    import traceback
                    error_trace = traceback.format_exc()
                    log_with_rank(f"Full traceback: {error_trace}", level="error")
                    log_with_rank(f"      📍 Full traceback:\n{error_trace}", verbose_only=True)
                    
                    return {
                        'reward': 0.0,
                        'knowledge_fidelity': 0.0,
                        'region_count': 0,
                        'R_score': 0.0,
                        'gt_graph': "Error: evaluation failed",
                        'pred_graph': "Error: evaluation failed",
                        'gt_image': gt_img,
                        'success': False
                    }
            
            async def evaluate_batch_async(generated_images, knowledge_graphs, annotations, batch_step, prompts=None, completions=None, keys=None, gt_img_paths=None, levels=None, disciplines=None):
                """Evaluate using MMMG protocol with MODERATE parallelization
                
                PARALLELIZATION STRATEGY:
                - GPT API calls: FULLY PARALLEL (16 concurrent per GPU, 128 system-wide)
                - SAM2 inference: SERIALIZED per GPU via ThreadPoolExecutor (1 worker, 8 system-wide)
                - Within each sample: GPT + SAM2 run concurrently
                - Across samples: GPT fully parallel, SAM2 queued
                
                This prevents OOM while maintaining speed:
                - GPT has no GPU memory impact → safe to parallelize fully
                - SAM2 uses GPU memory → serialize with ThreadPoolExecutor to prevent OOM
                - Best of both worlds: fast GPU SAM2 without memory conflicts or deadlock
                """
                import sys
                import gc
                import torch.distributed as dist
                import time
                
                batch_eval_start_time = time.time()
                log_with_rank(f"\n🔥 MMMG EVALUATOR CALLED with {len(generated_images)} images (MODERATE PARALLEL MODE)", verbose_only=True)
                sys.stdout.flush()
                
                levels_list = levels if levels else [None] * len(generated_images)
                disciplines_list = disciplines if disciplines else [None] * len(generated_images)
                prompts_list = prompts if prompts else ["Generate an educational image"] * len(generated_images)
                gt_img_paths_list = gt_img_paths if gt_img_paths else [None] * len(generated_images)
                
                # MODERATE PARALLELIZATION: GPT fully parallel, SAM2 serialized
                # - GPT: 16 parallel per GPU (fully async, no memory impact)
                # - SAM2: 1 at a time per GPU (ThreadPoolExecutor-controlled, prevents OOM)
                # - System-wide: 128 parallel GPT calls, 8 parallel SAM2 inferences
                
                # Create tasks for evaluation (local to this GPU)
                log_with_rank(f"🚀 Launching {len(generated_images)} evaluations on this GPU...", verbose_only=True)
                log_with_rank(f"   GPT: {len(generated_images)} parallel calls (fully concurrent)", verbose_only=True)
                log_with_rank(f"   SAM2: 1 at a time (serialized via ThreadPoolExecutor)", verbose_only=True)
                log_with_rank(f"   System-wide: {len(generated_images)} × {dist.get_world_size() if dist.is_initialized() else 1} GPUs", verbose_only=True)
                
                tasks = []
                for idx, (gen_img, kg_str, annotation) in enumerate(zip(generated_images, knowledge_graphs, annotations)):
                    # DEBUG: Log task creation
                    log_with_rank(f"   🎯 Creating async task for sample {idx+1}", verbose_only=True)
                    
                    task = evaluate_single_sample_async(
                        idx=idx,
                        gen_img=gen_img,
                        kg_str=kg_str,
                        annotation=annotation,
                        user_prompt=prompts_list[idx],
                        gt_img_path=gt_img_paths_list[idx],
                        level=levels_list[idx],
                        discipline=disciplines_list[idx]
                    )
                    
                    # DEBUG: Log task created
                    log_with_rank(f"   ✓ Task created for sample {idx+1}", verbose_only=True)
                    tasks.append(task)
                
                # Execute all evaluations in parallel (within this GPU)
                log_with_rank("⏳ Waiting for all parallel evaluations to complete on this GPU...", verbose_only=True)
                results = await asyncio.gather(*tasks)
                log_with_rank("✅ All parallel evaluations completed on this GPU!", verbose_only=True)
                
                # Unpack results
                rewards = []
                knowledge_fidelities = []
                region_counts = []
                R_scores = []
                gt_graphs = []
                pred_graphs = []
                gt_images = []
                
                for result in results:
                    rewards.append(result['reward'])
                    knowledge_fidelities.append(result['knowledge_fidelity'])
                    region_counts.append(result['region_count'])
                    R_scores.append(result['R_score'])
                    gt_graphs.append(result['gt_graph'])
                    pred_graphs.append(result['pred_graph'])
                    gt_images.append(result['gt_image'])
                
                # ========== Final Statistics (robust) ==========
                log_with_rank(f"\n🎁 Final MMMG Rewards (k_score_w): {rewards}", verbose_only=True)
                
                # Guard against empty lists
                if len(knowledge_fidelities) > 0:
                    kf_min = min(knowledge_fidelities)
                    kf_max = max(knowledge_fidelities)
                    kf_mean = sum(knowledge_fidelities) / len(knowledge_fidelities)
                    log_with_rank(f"   Knowledge Fidelity: min={kf_min:.3f}, max={kf_max:.3f}, mean={kf_mean:.3f}", verbose_only=True)
                else:
                    log_with_rank("   Knowledge Fidelity: No successful evaluations", verbose_only=True)
                
                if len(region_counts) > 0:
                    rc_min = min(region_counts)
                    rc_max = max(region_counts)
                    rc_mean = sum(region_counts) / len(region_counts)
                    log_with_rank(f"   Region Counts: min={rc_min}, max={rc_max}, mean={rc_mean:.1f}", verbose_only=True)
                else:
                    log_with_rank("   Region Counts: No successful evaluations", verbose_only=True)
                
                if len(R_scores) > 0:
                    r_min = min(R_scores)
                    r_max = max(R_scores)
                    r_mean = sum(R_scores) / len(R_scores)
                    log_with_rank(f"   R-scores: min={r_min:.3f}, max={r_max:.3f}, mean={r_mean:.3f}", verbose_only=True)
                else:
                    log_with_rank("   R-scores: No successful evaluations", verbose_only=True)
                
                # Save generated images with metadata
                if prompts is not None and completions is not None:
                    # Compute simple advantages (reward - mean)
                    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                    advantages = [r - mean_reward for r in rewards]
                    
                    # NOTE: batch_step is passed from parent scope (captured at batch entry)
                    # This is CRITICAL to avoid race conditions where trainer advances mid-batch
                    save_generated_images(
                        generated_images, prompts, completions, keys,
                        rewards, knowledge_fidelities, region_counts,
                        R_scores, gt_graphs, pred_graphs, gt_images,
                        current_step=batch_step,  # Use batch_step from parent scope
                        levels=levels_list,
                        disciplines=disciplines_list,
                        advantages=advantages
                    )
                
                # MEMORY OPTIMIZATION: Clear evaluation data after saving
                gc.collect()
                torch.cuda.empty_cache()
                
                # DEBUG: Log batch evaluation completion time
                batch_eval_total_time = time.time() - batch_eval_start_time
                log_with_rank(f"⏱️ MMMG EVALUATION COMPLETE in {batch_eval_total_time:.2f}s ({len(generated_images)} samples)")
                
                return rewards
            
            def evaluate_batch(generated_images, knowledge_graphs, annotations, batch_step, prompts=None, completions=None, keys=None, gt_img_paths=None, levels=None, disciplines=None):
                """Synchronous wrapper for async batch evaluation"""
                # Run the async function in an event loop
                return asyncio.run(
                    evaluate_batch_async(
                        generated_images, knowledge_graphs, annotations, batch_step,
                        prompts, completions, keys, gt_img_paths, levels, disciplines
                    )
                )
            
            gpt_eval_fn = evaluate_batch
        
        return gpt_eval_fn
    
    def reward_fn(completions, prompt_text: List, knowledge_graph: List[str], annotation: List[str], **kwargs):
        """
        Main reward function called by GRPO using MMMG evaluation.
        
        Args:
            completions: List[List[Dict]] - GRPO's generated completions
                Format: [[{"role": "assistant", "content": "..."}], ...]
            prompt_text: List[str] - Text prompts for image generation
            knowledge_graph: List[str] - JSON strings of knowledge graphs
            annotation: List[str] - Annotation strings
        
        Returns:
            List[float] - Reward scores (percentage of correctly visualized elements+dependencies)
        """
        try:
            # CRITICAL FIX: Capture global_step at batch ENTRY, before any async work
            # This ensures all ranks use the SAME step number for this batch
            # even if trainer advances while slow ranks are still computing
            import time
            batch_entry_time = time.time()
            
            if trainer_ref[0] is not None and hasattr(trainer_ref[0], 'state'):
                batch_step = trainer_ref[0].state.global_step
            else:
                batch_step = image_log_step[0]
            
            # Force flush logs immediately
            import sys
            log_with_rank("=" * 100, verbose_only=True)
            log_with_rank("🎯 MMMG REWARD FUNCTION CALLED!")
            log_with_rank(f"   🔢 CAPTURED batch_step: {batch_step} (will use this for storage)")
            log_with_rank(f"   Prompt batch size: {len(prompt_text)}", verbose_only=True)
            log_with_rank(f"   Completions: {len(completions)}", verbose_only=True)
            
            # DEBUG: Log trainer step to understand increments
            if trainer_ref[0] is not None and hasattr(trainer_ref[0], 'state'):
                log_with_rank(f"   📊 Trainer global_step at entry: {trainer_ref[0].state.global_step}")
                log_with_rank(f"   📊 Trainer _step (internal): {trainer_ref[0]._step}", verbose_only=True)
            log_with_rank(f"   📊 Image log step (legacy): {image_log_step[0]}", verbose_only=True)
            
            # Check prompt duplication
            unique_prompts = list(set(prompt_text))
            log_with_rank(f"   📊 Unique prompts: {len(unique_prompts)}", verbose_only=True)
            if len(unique_prompts) != len(prompt_text):
                from collections import Counter
                prompt_counts = Counter(prompt_text)
                log_with_rank(f"   📊 Prompt duplication counts: {dict(prompt_counts)}", verbose_only=True)
            
            # Log curriculum info (level/discipline distribution)
            levels = kwargs.get('level', [])
            disciplines = kwargs.get('discipline', [])
            if levels and disciplines:
                if not isinstance(levels, list):
                    levels = [levels] * len(prompt_text)
                if not isinstance(disciplines, list):
                    disciplines = [disciplines] * len(prompt_text)
                
                from collections import Counter
                level_dist = Counter(levels)
                discipline_dist = Counter(disciplines)
                combo_dist = Counter(zip(levels, disciplines))
                
                log_with_rank(f"   📚 Batch composition:", verbose_only=True)
                log_with_rank(f"      Levels: {dict(level_dist)}", verbose_only=True)
                log_with_rank(f"      Disciplines: {dict(discipline_dist)}", verbose_only=True)
                log_with_rank(f"      Combinations: {dict(combo_dist)}", verbose_only=True)
            
            log_with_rank("=" * 100, verbose_only=True)
            sys.stdout.flush()
            
            batch_size = len(prompt_text)
            log_with_rank(f"=== MMMG Reward computation for batch of {batch_size} ===")
            
            # Extract completion texts from GRPO (handle different formats)
            log_with_rank(f"🔍 DEBUG: completions type: {type(completions)}", verbose_only=True)
            log_with_rank(f"🔍 DEBUG: completions[0] type: {type(completions[0])}", verbose_only=True)
            log_with_rank(f"🔍 DEBUG: completions[0] content: {str(completions[0])[:200]}", verbose_only=True)
            
            # Handle different completion formats
            completion_texts = []
            for i, comp in enumerate(completions):
                if isinstance(comp, str):
                    # Already a string
                    completion_texts.append(comp)
                elif isinstance(comp, list) and len(comp) > 0:
                    # List of messages
                    if isinstance(comp[0], dict) and "content" in comp[0]:
                        completion_texts.append(comp[0]["content"])
                    elif isinstance(comp[0], str):
                        completion_texts.append(comp[0])
                    else:
                        log_with_rank(f"Unexpected completion format at index {i}: {type(comp[0])}", level="warning", verbose_only=True)
                        completion_texts.append(str(comp))
                elif isinstance(comp, dict) and "content" in comp:
                    # Single message dict
                    completion_texts.append(comp["content"])
                else:
                    log_with_rank(f"Unexpected completion format at index {i}: {type(comp)}", level="warning", verbose_only=True)
                    completion_texts.append(str(comp))
            
            log_with_rank(f"Completion text sample: {completion_texts[0][:200]}...", verbose_only=True)
            log_with_rank(f"📝 First completion: {completion_texts[0][:100]}...", verbose_only=True)
            
            # Get pipeline
            pipe = get_pipeline()
            
            # STEP 1: Extract embeddings from completion text
            log_with_rank("Extracting response embeddings from completion text...", verbose_only=True)
            response_embeds, response_masks = extract_response_embeddings(
                vlm_model=vlm_model[0],
                vlm_processor=vlm_processor,
                prompts=prompt_text,
                completion_texts=completion_texts,
            )
            
            # STEP 2: Generate images using these embeddings
            log_with_rank("Generating images...", verbose_only=True)
            
            generated_images = generate_with_embeddings(
                pipe=pipe,
                prompt_embeds=response_embeds,
                prompt_embeds_mask=response_masks,
            )
            
            # MEMORY OPTIMIZATION: Clear embeddings after generation
            del response_embeds, response_masks
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # STEP 3: Evaluate with MMMG GPT protocol
            log_with_rank("🔍 STEP 3: Calling MMMG GPT evaluator...", verbose_only=True)
            log_with_rank("Evaluating with MMMG protocol...", verbose_only=True)
            evaluator = get_gpt_evaluator()
            log_with_rank(f"   Evaluator initialized, calling with {len(prompt_text)} samples...", verbose_only=True)
            
            # Extract keys if available in kwargs
            keys = kwargs.get('key', None)
            if keys is not None and not isinstance(keys, list):
                keys = [keys] * len(prompt_text)
            
            # Extract ground truth image paths if available
            gt_img_paths = kwargs.get('gt_img_path', None)
            if gt_img_paths is not None and not isinstance(gt_img_paths, list):
                gt_img_paths = [gt_img_paths] * len(prompt_text)
            
            log_with_rank(f"🖼️ GT image paths: {gt_img_paths if gt_img_paths else 'None'}", verbose_only=True)
            
            # Call evaluator with all metadata for logging
            rewards = evaluator(
                generated_images, 
                knowledge_graph, 
                annotation,
                batch_step,  # Pass captured batch_step
                prompts=prompt_text,
                completions=completion_texts,
                keys=keys,
                gt_img_paths=gt_img_paths,
                levels=levels,
                disciplines=disciplines
            )
            
            log_with_rank(f"✅ MMMG GPT Evaluation complete! Rewards: {rewards}", verbose_only=True)
            log_with_rank(f"Rewards: {rewards}", verbose_only=True)
            
            # MEMORY OPTIMIZATION: Clear generated images after evaluation
            del generated_images, completion_texts
            gc.collect()
            torch.cuda.empty_cache()
            
            # Increment image logging step counter
            image_log_step[0] += 1
            
            # DEBUG: Log completion timing and final step state
            batch_total_time = time.time() - batch_entry_time
            if trainer_ref[0] is not None and hasattr(trainer_ref[0], 'state'):
                final_trainer_step = trainer_ref[0].state.global_step
                log_with_rank(f"⏱️ BATCH COMPLETE in {batch_total_time:.2f}s")
                log_with_rank(f"   Batch entry step: {batch_step}")
                log_with_rank(f"   Trainer step NOW: {final_trainer_step}")
                log_with_rank(f"   ⚠️ STEP ADVANCED during batch!" if final_trainer_step != batch_step else f"   ✓ Step unchanged")
            
            return rewards
            
        except Exception as e:
            log_with_rank(f"Reward computation failed: {e}", level="error")
            import traceback
            traceback.log_with_rank_exc()
            
            # Clear any partial data
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Increment image logging step counter even on error
            image_log_step[0] += 1
            return [0.5] * len(prompt_text)
    
    return reward_fn


def extract_response_embeddings(
    vlm_model,
    vlm_processor,
    prompts,
    completion_texts,
):
    """
    Extract embeddings from completion text via forward pass.
    
    Process:
    1. Format input: [system + user + prompt] + [assistant + completion_text]
    2. Forward pass to get hidden states
    3. Drop system prompt part
    4. Return embeddings for [user + prompt + assistant + completion]
    """
    import gc
    
    # Format prompts with completions
    # SYSTEM_PROMPT = (
    #     "You are an expert at generating detailed, accurate images based on text descriptions. "
    #     "Analyze the user's prompt carefully and create an image that precisely matches the requirements, "
    #     "paying attention to all specified details, objects, styles, colors, and compositions."
    # )
    SYSTEM_PROMPT = SYSTEM_TEMPLATE_PROMPT
    
    # Build full prompts with completions
    full_prompts = []
    for prompt, completion in zip(prompts, completion_texts):
        full_prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{completion}<|im_end|>"
        )
        full_prompts.append(full_prompt)
    
    # Process with VLM (no images for text-to-image)
    inputs = vlm_processor(
        text=full_prompts,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device)
    
    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = vlm_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )
    
    hidden_states = outputs.hidden_states[-1]  # Last layer
    
    # Extract masked hidden states (padding is already handled by this function)
    split_hidden_states = extract_masked_hidden(hidden_states, inputs.attention_mask)
    
    # MEMORY OPTIMIZATION: Clear intermediate variables
    del outputs, hidden_states
    
    # Drop system prompt tokens uniformly from all sequences
    # This matches the pipeline implementation in pipeline_qwenimage_response.py
    # The drop_idx should correspond to the system prompt + special tokens
    # Calculate drop_idx based on the actual system prompt used
    # Format the system prompt the same way as in full_prompts
    system_formatted = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    system_tokens = vlm_processor(
        text=system_formatted,
        return_tensors="pt",
        add_special_tokens=False,
    )
    drop_idx = system_tokens.input_ids.shape[1]  # Number of tokens in system prompt
    
    # Drop uniformly from all sequences (same as pipeline line 231)
    processed_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    
    # MEMORY OPTIMIZATION: Clear intermediate variables
    del split_hidden_states, system_tokens
    
    # Pad to max length
    max_len = max(h.size(0) for h in processed_hidden_states)
    
    # Create attention masks for each processed sequence (same as pipeline lines 232-239)
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in processed_hidden_states]
    
    # Stack and pad embeddings and masks to max length
    prompt_embeds = torch.stack([
        torch.cat([h, h.new_zeros(max_len - h.size(0), h.size(1))])
        for h in processed_hidden_states
    ])
    
    prompt_masks = torch.stack([
        torch.cat([u, u.new_zeros(max_len - u.size(0))])
        for u in attn_mask_list
    ])
    
    # MEMORY OPTIMIZATION: Clear intermediate variables
    del processed_hidden_states, attn_mask_list, inputs
    gc.collect()
    
    return prompt_embeds, prompt_masks


def extract_masked_hidden(hidden_states, mask):
    """Extract hidden states using attention mask"""
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def generate_with_embeddings(pipe, prompt_embeds, prompt_embeds_mask):
    """Generate images using pre-computed embeddings"""
    import gc
    
    batch_size = prompt_embeds.shape[0]

    log_with_rank("="*80, verbose_only=True)
    log_with_rank(f"Generating {batch_size} images with embeddings...", verbose_only=True)
    log_with_rank("="*80, verbose_only=True)
    
    # OPTIMIZATION: Dynamic sub-batch sizing based on available memory
    # H200 has 141GB VRAM - we can process larger batches efficiently
    if torch.cuda.is_available():
        try:
            # Get available memory
            device_idx = pipe.device.index if hasattr(pipe.device, 'index') else 0
            mem_free, mem_total = torch.cuda.mem_get_info(device_idx)
            mem_free_gb = mem_free / (1024**3)
            
            # Heuristic: Each image generation needs ~2GB peak (DiT is large)
            # Conservative estimate to avoid OOM
            estimated_batch_size = max(8, int(mem_free_gb / 3.0))  # 3GB per image for safety
            max_subbatch_size = min(estimated_batch_size, 64)  # Cap at 64
            
            log_with_rank(f"Dynamic batching: {mem_free_gb:.1f}GB free, using sub-batch size {max_subbatch_size}", verbose_only=True)
        except Exception as e:
            log_with_rank(f"Failed to query GPU memory: {e}, using default batch size", verbose_only=True)
            max_subbatch_size = 16
    else:
        max_subbatch_size = 16
    
    if batch_size <= max_subbatch_size:
        # Small batch - process all at once
        with torch.no_grad():  # Ensure no gradients for generation
            output = pipe(
                prompt=None,  # Must be None when using prompt_embeds
                negative_prompt=" ",
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                true_cfg_scale=1.0,
                num_inference_steps=8, #50,
                generator=torch.Generator(device=pipe.device).manual_seed(0), # Same noise seed for all samples across the group to minimize reward variance
            )
        
        images = output.images
        
        # MEMORY OPTIMIZATION: Clear immediately after getting images
        del output, prompt_embeds, prompt_embeds_mask
        gc.collect()
        torch.cuda.empty_cache()
        
        return images
    else:
        # Large batch - process in sub-batches
        log_with_rank(f"Large batch ({batch_size}), processing in sub-batches of {max_subbatch_size}", verbose_only=True)
        all_images = []
        
        for i in range(0, batch_size, max_subbatch_size):
            end_idx = min(i + max_subbatch_size, batch_size)
            log_with_rank(f"  Processing sub-batch {i//max_subbatch_size + 1}/{(batch_size + max_subbatch_size - 1)//max_subbatch_size} (samples {i}-{end_idx-1})", verbose_only=True)
            
            sub_embeds = prompt_embeds[i:end_idx]
            sub_masks = prompt_embeds_mask[i:end_idx]
            
            with torch.no_grad():  # Ensure no gradients for generation
                output = pipe(
                    prompt=None,
                    negative_prompt=" ",
                    prompt_embeds=sub_embeds,
                    prompt_embeds_mask=sub_masks,
                    true_cfg_scale=1.0,
                    num_inference_steps=8,
                    generator=torch.Generator(device=pipe.device).manual_seed(0),
                )
            
            all_images.extend(output.images)
            
            # MEMORY OPTIMIZATION: Aggressive cleanup after each sub-batch
            del output, sub_embeds, sub_masks
            gc.collect()
            torch.cuda.empty_cache()
        
        # Clear the original embeddings tensor after all sub-batches
        del prompt_embeds, prompt_embeds_mask
        gc.collect()
        
        return all_images


################
# Main
################
if __name__ == "__main__":
    # Parse args
    parser = TrlParser((ImageGenArgs, GRPOConfig, ModelConfig))
    img_args, training_args, model_args = parser.parse_args_and_config()
    
    # Setup
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Set verbose mode globally
    set_verbose_mode(img_args.verbose)
    
    # Setup logging with appropriate level
    log_level = "DEBUG" if img_args.verbose else "INFO"
    logger.add(
        Path(training_args.output_dir) / "training.log",
        level=log_level,
    )
    
    log_with_rank("="*80)
    log_with_rank("GRPO Training: Qwen-Image-Response")
    log_with_rank("="*80)
    log_with_rank(f"Verbose mode: {img_args.verbose}")
    
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
        img_args.levels,
        img_args.disciplines,
        max_samples=1000,
        local_data_path=img_args.local_data_path,
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
        
        log_with_rank(f"Calculated max_steps: {training_args.max_steps}")
        log_with_rank(f"  num_samples: {num_samples}")
        log_with_rank(f"  effective_batch_size: {effective_batch_size}")
        log_with_rank(f"  steps_per_epoch: {steps_per_epoch}")
    
    # Setup curriculum learning if requested
    curriculum_dataset = None
    if img_args.num_diff_lvls > 1:
        log_with_rank(f"🎓 Setting up curriculum learning with {img_args.num_diff_lvls} difficulty levels")
        
        # Load difficulty summary
        difficulty_summary_path = Path(img_args.difficulty_summary_path)
        if not difficulty_summary_path.is_absolute():
            # Relative to script directory
            difficulty_summary_path = Path(__file__).parent / difficulty_summary_path
        
        if not difficulty_summary_path.exists():
            log_with_rank(f"Difficulty summary not found: {difficulty_summary_path}", level="warning")
            log_with_rank("Falling back to random sampling (no curriculum)", level="warning")
        else:
            difficulty_summary = load_difficulty_summary(str(difficulty_summary_path))
            difficulty_groups = group_by_difficulty(difficulty_summary, img_args.num_diff_lvls)
            
            # Create curriculum dataset wrapper
            curriculum_dataset = CurriculumDataset(
                full_dataset=train_ds,
                difficulty_groups=difficulty_groups,
                max_steps=training_args.max_steps
            )
            
            # Use the curriculum dataset wrapper directly for training
            # It will dynamically filter samples based on training progress
            train_ds = curriculum_dataset
            log_with_rank(f"✅ Curriculum learning initialized")
            log_with_rank(f"   Dataset reports length: {len(train_ds)} (full dataset)")
            log_with_rank(f"   Currently available: {len(curriculum_dataset.current_indices)} samples (easiest group)")
            log_with_rank(f"   Will expand to {len(curriculum_dataset.full_dataset)} samples over {training_args.max_steps} steps")
    else:
        log_with_rank("📚 No curriculum learning (using all samples from start)")
    
    # Setup processor FIRST (needed for reward function)
    log_with_rank("Setting up VLM processor...")
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = 'left'
    
    # Create reward function BEFORE trainer
    # NOTE: vlm_model will be set inside the reward function when first called
    log_with_rank("Creating reward function...")
    vlm_model_ref = [None]  # Use mutable reference to share model
    trainer_ref = [None]  # Use mutable reference to share trainer
    
    def set_vlm_model(model):
        """Called by trainer to inject the model into reward function"""
        vlm_model_ref[0] = model
        log_with_rank("✅ VLM model injected into reward function")
    
    def set_trainer(trainer):
        """Called after trainer initialization to inject trainer reference"""
        trainer_ref[0] = trainer
        log_with_rank("✅ Trainer injected into reward function")
    
    reward_fn = create_reward_function(
        gen_model_path=img_args.gen_model_path,
        vlm_model=vlm_model_ref,  # Pass mutable reference
        trainer_ref=trainer_ref,  # Pass trainer reference
        vlm_processor=processor,
        api_key=api_key,
        sam2_checkpoint=img_args.sam2_checkpoint,
        alignment_weight=img_args.alignment_weight,
        quality_weight=img_args.quality_weight,
        n_evals=img_args.n_evals,
        logging_steps=training_args.logging_steps,
        training_args=training_args,
        img_args=img_args,  # Pass for wandb configuration
    )
    
    # Initialize GRPO trainer with reward function
    log_with_rank("Initializing GRPO trainer...")
    
    # Build reward function list
    reward_funcs = [reward_fn]
    if img_args.use_think_format_reward:
        try:
            from trl.rewards import think_format_reward
            reward_funcs.append(think_format_reward)
            log_with_rank("Added think_format_reward to reward functions")
        except ImportError:
            log_with_rank("think_format_reward not available, skipping", level="warning")
    
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
    )
    
    # Inject model and trainer into reward function
    set_vlm_model(trainer.model)
    set_trainer(trainer)
    
    # DEBUG: Log GRPO configuration to understand step increments
    log_with_rank("=" * 80)
    log_with_rank("GRPO Trainer Configuration:")
    log_with_rank(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    log_with_rank(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    log_with_rank(f"  num_generations: {training_args.num_generations}")
    log_with_rank(f"  steps_per_generation: {training_args.steps_per_generation}")
    log_with_rank(f"  num_iterations: {training_args.num_iterations}")
    log_with_rank(f"  Expected global_step increment per reward call: {training_args.steps_per_generation * training_args.num_iterations}")
    log_with_rank("=" * 80)
    
    # Add curriculum learning callback if enabled
    if curriculum_dataset is not None:
        from transformers import TrainerCallback
        
        class CurriculumCallback(TrainerCallback):
            """Callback to update curriculum dataset during training"""
            
            def __init__(self, curriculum_dataset, trainer):
                self.curriculum_dataset = curriculum_dataset
                self.trainer = trainer
                self.last_update_step = -1
            
            def on_step_end(self, args, state, control, **kwargs):
                """Update curriculum after each step"""
                current_step = state.global_step
                
                # Update curriculum (this will only trigger changes at difficulty boundaries)
                old_level = self.curriculum_dataset.current_difficulty_level
                self.curriculum_dataset.update_difficulty(current_step)
                new_level = self.curriculum_dataset.current_difficulty_level
                
                # Log when difficulty level changes
                # Note: The dataset's __getitem__ will automatically use new current_indices
                # No need to update trainer.train_dataset - sampling happens dynamically
                if new_level != old_level and current_step != self.last_update_step:
                    self.last_update_step = current_step
                    n_available = len(self.curriculum_dataset.current_indices)
                    log_with_rank(f"🔄 Curriculum Level Changed!")
                    log_with_rank(f"   Step: {current_step}")
                    log_with_rank(f"   New difficulty level: {new_level + 1}/{self.curriculum_dataset.num_groups}")
                    log_with_rank(f"   Available samples: {n_available}/{len(self.curriculum_dataset.full_dataset)}")
                    log_with_rank(f"   Progress: {(current_step / args.max_steps)*100:.1f}%")
                
                return control
        
        callback = CurriculumCallback(curriculum_dataset, trainer)
        trainer.add_callback(callback)
        log_with_rank("✅ Curriculum callback registered")
    
    # Register image logging callback for wandb (avoids deadlock by calling gather_object in synchronized on_step_end)
    if img_args.wandb_log_images > 0:
        from transformers import TrainerCallback
        from pathlib import Path
        
        class ImageLoggingCallback(TrainerCallback):
            """
            Callback to log generated images to wandb in distributed training.
            
            Gathers data from all ranks in on_step_end() where ranks are synchronized,
            avoiding deadlock from calling gather_object() during reward computation.
            """
            
            def __init__(self, wandb_log_images, wandb_image_size, logging_steps):
                self.wandb_log_images = wandb_log_images
                self.wandb_image_size = wandb_image_size
                self.logging_steps = logging_steps
                self.pending_data = {}  # step -> rank_data mapping
            
            def store_data_for_step(self, step, rank_data):
                """Called by reward function to store data for later logging"""
                self.pending_data[step] = rank_data
            
            def on_step_end(self, args, state, control, **kwargs):
                """
                Called after each training step when all ranks are synchronized.
                This is where we safely call gather_object().
                """
                import torch.distributed as dist
                
                current_step = state.global_step
                
                # Only log every N steps - all ranks check same condition
                if current_step % self.logging_steps != 0:
                    return control
                
                log_with_rank(f"🔍 CALLBACK on_step_end called")
                log_with_rank(f"   Trainer global_step: {current_step}")
                log_with_rank(f"   Will check keys: {[current_step-1, current_step, current_step+1]}")
                log_with_rank(f"   Available in pending_data: {sorted(self.pending_data.keys())}")
                
                # Prepare data for this rank (wrap in list for gather_object)
                # gather_object expects iterables, so use [rank_data] or []
                # Note: Check all possible step keys since image_log_step might differ from global_step
                rank_data = []
                for step_key in [current_step, current_step - 1, current_step + 1]:
                    if step_key in self.pending_data:
                        rank_data = [self.pending_data.pop(step_key)]
                        log_with_rank(f"✅ FOUND data for step_key={step_key}")
                        break
                
                if not rank_data:
                    log_with_rank(f"❌ NO DATA FOUND!")
                    log_with_rank(f"   Searched for: {[current_step-1, current_step, current_step+1]}")
                    log_with_rank(f"   Available keys: {sorted(self.pending_data.keys())}")
                    log_with_rank(f"   ⚠️ This will cause 'No valid samples' warning!")
                else:
                    log_with_rank(f"   Retrieved {len(rank_data[0]['sampled_data']) if rank_data and len(rank_data) > 0 and 'sampled_data' in rank_data[0] else 0} samples", verbose_only=True)
                
                # CRITICAL: Barrier to ensure all ranks reach this point before gather
                # This prevents deadlock where some ranks call gather_object before others enter callback
                if dist.is_initialized():
                    log_with_rank(f"Waiting at barrier before gather (step {current_step})...", verbose_only=True)
                    dist.barrier()
                    log_with_rank(f"Barrier cleared, proceeding to gather (step {current_step})...", verbose_only=True)
                
                # CRITICAL: All ranks participate in gather (synchronized by barrier above)
                # Ranks without data pass empty list []
                log_with_rank(f"Gathering image data from all ranks via gather_object (step {current_step})...", verbose_only=True)
                all_ranks_data_nested = gather_object(rank_data)
                log_with_rank(f"Successfully gathered {len(all_ranks_data_nested)} items from all ranks", verbose_only=True)
                
                # gather_object returns dicts directly (not wrapped in lists!)
                # Structure: [dict, dict, None, None] where each dict is {'rank': X, 'sampled_data': [...], ...}
                all_ranks_data = all_ranks_data_nested  # No flattening needed!
                
                # DEBUG: Validate structure and count valid items
                valid_count = 0
                for i, item in enumerate(all_ranks_data):
                    if item is not None and isinstance(item, dict) and 'sampled_data' in item:
                        n_samples = len(item.get('sampled_data', []))
                        valid_count += 1
                        log_with_rank(f"  Rank {i}: {n_samples} samples", verbose_only=True)
                    else:
                        log_with_rank(f"  Rank {i}: None or invalid", verbose_only=True)
                
                log_with_rank(f"✅ Gathered data from {valid_count}/{len(all_ranks_data)} ranks")
                
                # Only rank 0 logs to wandb
                rank = get_rank()
                if rank == 0:
                    self._log_to_wandb(all_ranks_data, current_step)
                else:
                    log_with_rank("Gather complete, continuing training (non-rank-0)", verbose_only=True)
                
                return control
            
            def _log_to_wandb(self, all_ranks_data, step):
                """Rank 0 only: Process gathered data and log to wandb"""
                try:
                    import wandb
                    from PIL import Image as PILImage
                    
                    if wandb.run is None:
                        log_with_rank("wandb.run is None, skipping", level="warning", verbose_only=True)
                        return
                    
                    # Define custom step metric for image logging
                    try:
                        wandb.define_metric("reward_step")
                        wandb.define_metric("rank*/generation_table", step_metric="reward_step")
                        wandb.define_metric("rank*/avg_*", step_metric="reward_step")
                        wandb.define_metric("rank*/reward_*", step_metric="reward_step")
                        wandb.define_metric("rank*/num_*", step_metric="reward_step")
                    except Exception:
                        pass
                    
                    log_with_rank("Processing gathered data from all ranks...", verbose_only=True)
                    
                    # Filter valid data
                    valid_ranks_data = []
                    for i, data in enumerate(all_ranks_data):
                        # Skip None and invalid data
                        if data is None:
                            log_with_rank(f"Item {i} is None", verbose_only=True)
                        elif isinstance(data, dict) and data.get("sampled_data") and len(data["sampled_data"]) > 0:
                            valid_ranks_data.append(data)
                            log_with_rank(f"Item {i} has {len(data['sampled_data'])} samples", verbose_only=True)
                        else:
                            log_with_rank(f"Item {i} has no valid samples", verbose_only=True)
                    
                    # Guard: Check if we have any valid data from any rank
                    if len(valid_ranks_data) == 0:
                        log_with_rank("No valid samples from any rank, skipping wandb logging", level="warning")
                        return
                    
                    # Create compact wandb table with SAMPLED images
                    log_with_rank(f"Creating wandb tables for sampled images from {len(valid_ranks_data)} ranks...", verbose_only=True)
                    
                    # Comprehensive columns including prompts, completions, and graphs
                    columns = [
                        "rank", "idx", "image", "gt_image", 
                        "prompt", "completion",
                        "level", "discipline",
                        "KF", "regions", "R_score", "reward", "advantage",
                        "gt_graph", "pred_graph"
                    ]
                    
                    table_data = []
                    for rank_data_item in valid_ranks_data:
                        r = rank_data_item["rank"]
                        for sample in rank_data_item["sampled_data"]:
                            # Convert to wandb.Image (already resized and compressed)
                            wandb_img = wandb.Image(sample["image"])
                            if sample["gt_image"] is not None:
                                wandb_gt_img = wandb.Image(sample["gt_image"])
                            else:
                                blank = PILImage.new('RGB', (50, 50), color='gray')
                                wandb_gt_img = wandb.Image(blank, caption="No GT")
                            
                            table_data.append([
                                r,
                                sample["idx"],
                                wandb_img,
                                wandb_gt_img,
                                sample["prompt"],
                                sample["completion"],
                                sample["level"],
                                sample["discipline"],
                                sample["knowledge_fidelity"],
                                sample["region_count"],
                                sample["R_score"],
                                sample["reward"],
                                sample["advantage"],
                                sample["gt_graph"],
                                sample["pred_graph"],
                            ])
                    
                    # Single combined table for all ranks
                    log_with_rank(f"Creating wandb table with {len(table_data)} rows...", verbose_only=True)
                    table = wandb.Table(columns=columns, data=table_data)
                    
                    # Prepare log dict
                    log_dict = {"reward_step": step}
                    log_dict["generation_samples"] = table
                    log_with_rank(f"Prepared log_dict with {len(log_dict)} keys", verbose_only=True)
                    
                    # Log summary stats and reward distributions for each rank
                    for rank_data_item in valid_ranks_data:
                        r = rank_data_item["rank"]
                        stats = rank_data_item["summary_stats"]
                        log_dict[f"rank{r}/num_total"] = stats["num_total"]
                        log_dict[f"rank{r}/num_sampled"] = stats["num_sampled"]
                        log_dict[f"rank{r}/avg_reward"] = stats["avg_reward"]
                        log_dict[f"rank{r}/avg_kf"] = stats["avg_knowledge_fidelity"]
                        log_dict[f"rank{r}/avg_regions"] = stats["avg_region_count"]
                        log_dict[f"rank{r}/avg_R_score"] = stats["avg_R_score"]
                        log_dict[f"rank{r}/reward_std"] = stats["reward_std"]
                        
                        # Create distribution histograms for key metrics for this rank
                        rank_rewards = [sample["reward"] for sample in rank_data_item["sampled_data"]]
                        rank_kfs = [sample["knowledge_fidelity"] for sample in rank_data_item["sampled_data"]]
                        rank_regions = [sample["region_count"] for sample in rank_data_item["sampled_data"]]
                        rank_rscores = [sample["R_score"] for sample in rank_data_item["sampled_data"]]
                        
                        if len(rank_rewards) > 0:
                            log_dict[f"rank{r}/reward_distribution"] = wandb.Histogram(rank_rewards)
                            log_dict[f"rank{r}/kf_distribution"] = wandb.Histogram(rank_kfs)
                            log_dict[f"rank{r}/regions_distribution"] = wandb.Histogram(rank_regions)
                            log_dict[f"rank{r}/rscore_distribution"] = wandb.Histogram(rank_rscores)
                            log_with_rank(f"   Rank {r}: distributions with {len(rank_rewards)} samples", verbose_only=True)
                    
                    # Create combined metric distributions across all ranks
                    all_rewards = []
                    all_kfs = []
                    all_regions = []
                    all_rscores = []
                    for rank_data_item in valid_ranks_data:
                        all_rewards.extend([sample["reward"] for sample in rank_data_item["sampled_data"]])
                        all_kfs.extend([sample["knowledge_fidelity"] for sample in rank_data_item["sampled_data"]])
                        all_regions.extend([sample["region_count"] for sample in rank_data_item["sampled_data"]])
                        all_rscores.extend([sample["R_score"] for sample in rank_data_item["sampled_data"]])
                    
                    if len(all_rewards) > 0:
                        log_dict["all_ranks/reward_distribution"] = wandb.Histogram(all_rewards)
                        log_dict["all_ranks/kf_distribution"] = wandb.Histogram(all_kfs)
                        log_dict["all_ranks/regions_distribution"] = wandb.Histogram(all_regions)
                        log_dict["all_ranks/rscore_distribution"] = wandb.Histogram(all_rscores)
                        log_with_rank(f"   Combined distributions with {len(all_rewards)} total samples", verbose_only=True)
                    
                    # Single log call for all data
                    wandb.log(log_dict, commit=True)
                    log_with_rank(f"✅ Successfully logged {len(table_data)} samples to wandb table 'generation_samples' at step {step}")
                    log_with_rank(f"   Logged data from {len(valid_ranks_data)} ranks", verbose_only=True)
                
                except Exception as e:
                    log_with_rank(f"Failed to log to wandb: {e}", level="error")
                    import traceback
                    traceback.print_exc()
        
        image_callback = ImageLoggingCallback(
            wandb_log_images=img_args.wandb_log_images,
            wandb_image_size=img_args.wandb_image_size,
            logging_steps=training_args.logging_steps
        )
        trainer.add_callback(image_callback)
        trainer.image_logging_callback = image_callback  # Store reference for reward function
        log_with_rank("✅ Image logging callback registered")
    
    # Train
    log_with_rank("Starting training...")
    trainer.train()
    
    # Save
    log_with_rank("Saving model...")
    trainer.save_model(training_args.output_dir)
    
    log_with_rank("✅ Training complete!")

