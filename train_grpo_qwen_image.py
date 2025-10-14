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
    
    logger.info(f"Loading MMMG from local file: {local_data_path}")
    logger.info(f"  Levels: {levels_list}")
    logger.info(f"  Disciplines: {disciplines_list}")
    
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
            logger.warning(f"Level {level} not found in data (looking for key: {json_key})")
            continue
            
        level_data = data[json_key]
        
        for discipline in disciplines_list:
            if discipline not in level_data:
                logger.warning(f"Discipline {discipline} not found in level {level}")
                continue
            
            discipline_samples = level_data[discipline]
            logger.info(f"Loading {level}/{discipline}: {len(discipline_samples)} samples")
            
            # Convert to samples - extract knowledge graph and annotation
            for row in discipline_samples:
                prompt = row.get("question")  # Changed from 'prompt' to 'question'
                key = row.get("key")
                visual_components = row.get("Visual Components", {})
                key_knowledge = row.get("Key Knowledge", {})
                
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
                })
            
            logger.info(f"  Loaded {len(discipline_samples)} samples from {level}_{discipline}")
    
    logger.info(f"Total collected samples: {len(all_samples)}")
    
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
        })
    
    # Convert to HF dataset
    from datasets import Dataset
    dataset = Dataset.from_list(formatted_samples)
    
    logger.info(f"Dataset formatted: {len(dataset)} samples")
    
    # Limit samples
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {len(dataset)} samples")
    
    logger.info(f"Dataset ready: {len(dataset)} total samples")
    logger.info(f"  Levels: {levels_list}")
    logger.info(f"  Disciplines: {disciplines_list}")
    
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
    vlm_processor,
    api_key: str,
    alignment_weight: float,
    quality_weight: float,
    n_evals: int,
    logging_steps: int = 1,
    training_args=None,
):
    """
    Create reward function using MMMG evaluation protocol that:
    1. Extracts embeddings from GRPO's completion text
    2. Generates images via diffusion
    3. Evaluates with GPT using MMMG knowledge graph evaluation
    """
    
    # Lazy load
    gen_pipeline = None
    gpt_eval_fn = None
    
    # Step counter for logging
    reward_step_counter = [0]  # Use list to make it mutable in nested function
    
    def get_pipeline():
        nonlocal gen_pipeline
        if gen_pipeline is None:
            from diffusers import QwenImageResponsePipeline
            import torch.distributed as dist
            
            # Get device for this process
            if dist.is_initialized():
                device = f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Loading pipeline on {device}")
            
            gen_pipeline = QwenImageResponsePipeline.from_pretrained(
                gen_model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            
            # CRITICAL: Replace text_encoder with GRPO-trained VLM
            gen_pipeline.text_encoder = vlm_model[0]
            logger.info("Pipeline loaded, text_encoder = trained VLM")
        
        return gen_pipeline
    
    def get_gpt_evaluator():
        nonlocal gpt_eval_fn
        if gpt_eval_fn is None:
            from openai import OpenAI
            import json
            import re
            
            client = OpenAI(api_key=api_key)
            
            # Visual Quality Evaluation Prompt
            VISUAL_QUALITY_PROMPT = """Please evaluate the visual quality and readability of this educational image across the following dimensions:

1. **Visual Clarity** (0-10): Is the image clear, sharp, and in focus? Are there any blur, noise, or artifacts?

2. **Layout & Composition** (0-10): Is the layout well-organized? Are elements properly arranged and balanced? Is the composition aesthetically pleasing?

3. **Color & Contrast** (0-10): Are colors appropriate and harmonious? Is there sufficient contrast for readability? Are visual elements distinguishable?

4. **Text Readability** (0-10): If text is present, is it legible and well-sized? Is the font appropriate? Is text positioning good?

5. **Visual Coherence** (0-10): Do all visual elements work together coherently? Are there any conflicting or contradictory visual cues?

Please provide your evaluation in JSON format with scores (0-10) and brief justifications.

**IMPORTANT: Your response must be a valid JSON object with this exact structure:**

```json
{
  "visual_clarity": {"score": 0-10, "reason": "brief explanation"},
  "layout_composition": {"score": 0-10, "reason": "brief explanation"},
  "color_contrast": {"score": 0-10, "reason": "brief explanation"},
  "text_readability": {"score": 0-10, "reason": "brief explanation"},
  "visual_coherence": {"score": 0-10, "reason": "brief explanation"},
  "overall_score": 0-10
}
```

Return ONLY the JSON object, no additional text."""

            def parse_mmmg_response(text, elements, dependencies):
                """Parse GPT response to extract yes/no for elements and dependencies"""
                all_keys = elements + dependencies
                
                # Clean text
                text = text.replace("**", "")
                text = text.replace("[yes]", "yes")
                text = text.replace("[no]", "no")
                
                # Regex pattern to match "element: yes/no"
                escaped_keys = [re.escape(k) for k in all_keys]
                pattern = rf"^\s*(?:[*\-•]|\d+\.)?\s*`*\s*({'|'.join(escaped_keys)})`*\s*[:：]\s*(yes|no|YES|NO|Yes|No)\b"
                matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
                
                # Initialize dicts
                element_dict = {key: False for key in elements}
                dependency_dict = {key: False for key in dependencies}
                
                # Parse matches
                for raw_k, v in matches:
                    k_clean = raw_k.strip()
                    v_bool = v.strip().lower() == "yes"
                    
                    if k_clean in element_dict:
                        element_dict[k_clean] = v_bool
                    elif k_clean in dependency_dict:
                        dependency_dict[k_clean] = v_bool
                    else:
                        # Fuzzy match
                        for ele in element_dict.keys():
                            if k_clean.lower() == ele.lower():
                                element_dict[ele] = v_bool
                                break
                        for dep in dependency_dict.keys():
                            if k_clean.lower() == dep.lower():
                                dependency_dict[dep] = v_bool
                                break
                
                return element_dict, dependency_dict
            
            def parse_visual_quality_score(text):
                """Extract overall quality score and detailed reasoning from JSON visual quality evaluation
                
                Returns:
                    score: float - Overall visual quality score (0-1)
                    details: dict - Detailed breakdown with scores and reasons for each dimension
                """
                import sys
                
                print(f"\n      📄 GPT Visual Quality Response (first 500 chars):\n      {text[:500]}\n", flush=True)
                sys.stdout.flush()
                
                try:
                    # Extract JSON from response (handle markdown code blocks)
                    json_text = text.strip()
                    
                    # Remove markdown code blocks if present
                    if "```json" in json_text:
                        json_text = json_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_text:
                        json_text = json_text.split("```")[1].split("```")[0].strip()
                    
                    # Parse JSON
                    data = json.loads(json_text)
                    
                    # Get overall score (normalize from 0-10 to 0-1)
                    if "overall_score" in data:
                        score = float(data["overall_score"]) / 10.0
                        print(f"      ✓ Parsed overall score from JSON: {score:.3f} (raw: {data['overall_score']}/10)", flush=True)
                        return score, data  # Return both score and full details
                    
                    # Fallback: calculate average from individual dimensions
                    dimension_keys = ["visual_clarity", "layout_composition", "color_contrast", 
                                     "text_readability", "visual_coherence"]
                    scores = []
                    for key in dimension_keys:
                        if key in data and isinstance(data[key], dict) and "score" in data[key]:
                            scores.append(float(data[key]["score"]) / 10.0)
                    
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        print(f"      ✓ Calculated average from {len(scores)} JSON dimensions: {avg_score:.3f}", flush=True)
                        return avg_score, data  # Return both score and full details
                    
                    print(f"      ⚠️  JSON parsed but no valid scores found", flush=True)
                    logger.warning(f"Visual quality JSON missing scores. Data: {data}")
                    return 0.5, {"error": "No valid scores found", "raw_data": data}
                    
                except json.JSONDecodeError as e:
                    print(f"      ⚠️  JSON parsing failed: {e}", flush=True)
                    logger.warning(f"Visual quality JSON parsing failed: {e}. Response: {text[:200]}")
                    return 0.5, {"error": f"JSON parsing failed: {e}", "raw_response": text[:500]}
                except Exception as e:
                    print(f"      ⚠️  Unexpected error parsing visual quality: {e}", flush=True)
                    logger.warning(f"Visual quality parsing error: {e}. Response: {text[:200]}")
                    return 0.5, {"error": f"Unexpected error: {e}", "raw_response": text[:500]}
            
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
                                    print(f"Error parsing dependency BRACKETS: {dep}")
                                
                                # Try splitting with ", " first
                                source, target = nodes.split(', ', 1)
                            except ValueError:
                                # Fallback: try splitting with "," only
                                if len(nodes.split(', ', 1)) == 1 and len(nodes.split(',', 1)) == 2:
                                    source, target = nodes.split(',', 1)
                                else:
                                    print(f"Error parsing dependency: {dep}")
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
                Follows MMMG step3_stat.py exactly: lines 123-130, 198-202
                
                Returns:
                    knowledge_fidelity: float in [0, 1], where 1 = perfect match, 0 = completely wrong
                    G_gt: Ground truth graph (for logging)
                    G_pred: Predicted graph (for logging)
                """
                import networkx as nx
                
                # Build GT graph (all elements and dependencies)
                all_elements_gt = {elem: True for elem in gt_elements}
                all_dependencies_gt = {dep: True for dep in gt_dependencies}
                G_gt = build_graph(all_elements_gt, all_dependencies_gt)
                
                # Build predicted graph (only elements/deps marked as True)
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
            
            def save_generated_images(images, prompts, completions, keys, rewards, knowledge_fidelities, visual_qualities, visual_quality_details, gt_graphs, pred_graphs, current_step):
                """Save generated images with metadata (including graphs and visual quality reasoning) for analysis and upload to wandb (respecting logging_steps)"""
                import json
                from datetime import datetime
                
                # Create output directory
                output_dir = Path(training_args.output_dir) / "generated_images"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get current training step (if available)
                try:
                    from torch.distributed import get_rank, is_initialized
                    if is_initialized():
                        rank = get_rank()
                    else:
                        rank = 0
                except:
                    rank = 0
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_dir = output_dir / f"step_{timestamp}_rank{rank}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                
                # Prepare wandb logging (if enabled)
                wandb_images = []
                use_wandb = training_args.report_to and "wandb" in training_args.report_to
                
                # Save each image with metadata
                for i, (img, prompt, completion, key, reward, kf, vq, vq_detail, gt_graph, pred_graph) in enumerate(
                    zip(images, prompts, completions, keys or [None]*len(images), 
                        rewards, knowledge_fidelities, visual_qualities, visual_quality_details, gt_graphs, pred_graphs)
                ):
                    # Save image to disk
                    img_filename = f"image_{i:03d}.png"
                    img_path = batch_dir / img_filename
                    img.save(img_path)
                    
                    # Save metadata (including graphs and visual quality details)
                    metadata = {
                        "image_filename": img_filename,
                        "key": key,
                        "prompt": prompt,
                        "completion": completion,
                        "completion_full_length": len(completion),
                        "rewards": {
                            "knowledge_fidelity": float(kf),
                            "visual_quality": float(vq),
                            "visual_quality_details": vq_detail,  # Full reasoning
                            "final_reward": float(reward),
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
                    
                    # Prepare data for gathering - ALL ranks must do this!
                    # Store PIL image directly (not wandb.Image yet)
                    # We'll convert to wandb.Image during gathering/logging
                    wandb_images.append({
                        "image_pil": img,  # Store PIL image directly
                        "system_prompt": SYSTEM_TEMPLATE_PROMPT,
                        "prompt": prompt,
                        "completion_preview": completion,
                        "key": key,
                        "knowledge_fidelity": kf,
                        "visual_quality": vq,
                        "visual_quality_details": vq_detail,
                        "reward": reward,
                        "gt_graph": gt_graph,
                        "pred_graph": pred_graph,
                    })
                
                # Save batch summary
                summary = {
                    "timestamp": timestamp,
                    "rank": rank,
                    "num_images": len(images),
                    "average_knowledge_fidelity": float(sum(knowledge_fidelities) / len(knowledge_fidelities)),
                    "average_visual_quality": float(sum(visual_qualities) / len(visual_qualities)),
                    "average_reward": float(sum(rewards) / len(rewards)),
                    "reward_std": float(torch.tensor(rewards).std().item()) if len(rewards) > 1 else 0.0,
                }
                
                summary_path = batch_dir / "batch_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                logger.info(f"Saved {len(images)} generated images to {batch_dir}")
                print(f"💾 Saved {len(images)} images to {batch_dir}", flush=True)
                
                # Gather data from all ranks to rank 0 (ALL RANKS must participate!)
                # This is a collective operation that must be called by all ranks
                should_log_to_wandb = (current_step % logging_steps == 0)
                
                # Initialize these for all code paths
                rewards = []
                knowledge_fidelities = []
                visual_qualities = []
                
                try:
                    import torch.distributed as dist
                    
                    # Check if we're in distributed mode AND it's a logging step
                    # ALL ranks must participate in gathering (collective operation)
                    if dist.is_initialized() and should_log_to_wandb:
                        # Gather data from all ranks
                        world_size = dist.get_world_size()
                        
                        logger.info(f"Rank {rank}: Preparing data for gathering ({len(wandb_images)} samples)")
                        print(f"🔄 Rank {rank}: Preparing {len(wandb_images)} samples for gathering", flush=True)
                        
                        # Each rank prepares its data as a serializable dict
                        rank_data = {
                            "rank": rank,
                            "images_base64": [],
                            "metadata": [],
                        }
                        
                        # Convert images to base64 for gathering (wandb.Image objects can't be pickled)
                        from io import BytesIO
                        import base64
                        
                        for item in wandb_images:
                            # Convert PIL image to base64
                            img_pil = item["image_pil"]  # Get PIL image directly
                            buffered = BytesIO()
                            img_pil.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode()
                            
                            rank_data["images_base64"].append(img_base64)
                            rank_data["metadata"].append({
                                "prompt": item["prompt"],
                                "completion_preview": item["completion_preview"],
                                "key": item["key"],
                                "knowledge_fidelity": float(item["knowledge_fidelity"]),
                                "visual_quality": float(item["visual_quality"]),
                                "visual_quality_details": item.get("visual_quality_details", ""),
                                "reward": float(item["reward"]),
                                "gt_graph": item.get("gt_graph", ""),
                                "pred_graph": item.get("pred_graph", ""),
                            })
                        
                        logger.info(f"Rank {rank}: Calling gather_object with {len(rank_data['images_base64'])} images")
                        print(f"📡 Rank {rank}: Calling gather_object...", flush=True)
                        
                        # Gather to rank 0 - ALL RANKS must call this!
                        gathered_data = [None] * world_size if rank == 0 else None
                        dist.gather_object(rank_data, gathered_data, dst=0)
                        
                        logger.info(f"Rank {rank}: gather_object completed")
                        print(f"✅ Rank {rank}: Gathering completed", flush=True)
                        
                        # Only rank 0 processes and logs
                        if rank == 0:
                            # Reconstruct wandb_images from gathered data
                            all_wandb_images = []
                            all_rewards = []
                            all_kf = []
                            all_vq = []
                            
                            import wandb
                            from PIL import Image
                            
                            logger.info(f"Rank 0: Processing gathered data from {len(gathered_data)} ranks")
                            
                            for rank_idx, data in enumerate(gathered_data):
                                logger.info(f"Rank 0: Processing data from rank {rank_idx} ({len(data['images_base64'])} images)")
                                for img_base64, meta in zip(data["images_base64"], data["metadata"]):
                                    # Decode base64 back to PIL Image
                                    img_bytes = base64.b64decode(img_base64)
                                    img_pil = Image.open(BytesIO(img_bytes))
                                    
                                    # Create wandb.Image
                                    wandb_img = wandb.Image(img_pil)
                                    
                                    all_wandb_images.append({
                                        "image": wandb_img,
                                        "rank": rank_idx,  # Track which GPU
                                        "prompt": meta["prompt"],
                                        "completion_preview": meta["completion_preview"],
                                        "key": meta["key"],
                                        "knowledge_fidelity": meta["knowledge_fidelity"],
                                        "visual_quality": meta["visual_quality"],
                                        "visual_quality_details": meta.get("visual_quality_details", ""),
                                        "reward": meta["reward"],
                                        "gt_graph": meta.get("gt_graph", ""),
                                        "pred_graph": meta.get("pred_graph", ""),
                                    })
                                    all_rewards.append(meta["reward"])
                                    all_kf.append(meta["knowledge_fidelity"])
                                    all_vq.append(meta["visual_quality"])
                            
                            logger.info(f"Rank 0: Gathered {len(all_wandb_images)} images from {world_size} ranks")
                            print(f"📦 Gathered {len(all_wandb_images)} images from {world_size} GPUs", flush=True)
                            
                            # Debug: Check what fields we have in the first item
                            if all_wandb_images:
                                sample_keys = list(all_wandb_images[0].keys())
                                logger.info(f"Rank 0: Sample wandb_images item has keys: {sample_keys}")
                                print(f"🔍 Sample item keys: {sample_keys}", flush=True)
                            
                            # Use gathered data for logging
                            wandb_images = all_wandb_images
                            rewards = all_rewards
                            knowledge_fidelities = all_kf
                            visual_qualities = all_vq
                    else:
                        # Single GPU or not a logging step - no gathering needed
                        logger.info(f"Rank {rank}: No gathering (dist.is_initialized()={dist.is_initialized()}, should_log={should_log_to_wandb})")
                        if use_wandb and wandb_images:
                            rewards = [item["reward"] for item in wandb_images]
                            knowledge_fidelities = [item["knowledge_fidelity"] for item in wandb_images]
                            visual_qualities = [item["visual_quality"] for item in wandb_images]
                        
                except Exception as e:
                    logger.error(f"Rank {rank}: Failed to gather data: {e}")
                    print(f"⚠️  Rank {rank}: Data gathering failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    # Fallback: use local data only
                    if use_wandb and wandb_images:
                        rewards = [item["reward"] for item in wandb_images]
                        knowledge_fidelities = [item["knowledge_fidelity"] for item in wandb_images]
                        visual_qualities = [item["visual_quality"] for item in wandb_images]
                
                # Only rank 0 logs to wandb
                if use_wandb and wandb_images and should_log_to_wandb and rank == 0:
                    try:
                        import wandb
                        
                        # Check if wandb is initialized (TRL should have done this on rank 0)
                        if wandb.run is None:
                            logger.warning(f"Rank 0: wandb.run is None, skipping upload. Wandb not initialized yet.")
                            print(f"⚠️  Rank 0: Skipping wandb upload - wandb not initialized", flush=True)
                        else:
                            logger.info(f"Rank 0: wandb.run exists, proceeding with upload")
                        
                            # Convert PIL images to wandb.Image if needed (for non-gathered case)
                            # After gathering, items already have "image" as wandb.Image
                            # But in single-GPU case, items have "image_pil" instead
                            for item in wandb_images:
                                if "image_pil" in item and "image" not in item:
                                    item["image"] = wandb.Image(item["image_pil"])
                        
                            # Prepare all metrics in a single dict to log together
                            log_dict = {}
                            
                            # Create enhanced completions table with all metadata (integrated with TRL format)
                            # This table mirrors TRL's completions table but adds our custom metrics
                            # Now includes data from ALL GPUs + knowledge graphs + visual quality reasoning!
                            enhanced_columns = [
                                "step", "image", "rank", "prompt", "completion", "key",
                                "knowledge_fidelity", "visual_quality", "reward",
                                "visual_quality_details", "ground_truth_graph", "predicted_graph"
                            ]
                            enhanced_data = [[
                                current_step,  # step (matches TRL format)
                                item["image"],
                                item.get("rank", 0),  # GPU rank
                                item["prompt"], 
                                item["completion_preview"],
                                item["key"], 
                                item["knowledge_fidelity"], 
                                item["visual_quality"], 
                                item["reward"],
                                item.get("visual_quality_details", ""),
                                item.get("gt_graph", ""),
                                item.get("pred_graph", "")
                            ] for item in wandb_images]
                            
                            enhanced_table = wandb.Table(columns=enhanced_columns, data=enhanced_data)
                            log_dict["completions_with_images"] = enhanced_table
                            
                            logger.info(f"Rank 0: Created completions_with_images table with {len(enhanced_data)} rows and columns: {enhanced_columns}")
                            print(f"📊 Created completions_with_images table:", flush=True)
                            print(f"   Rows: {len(enhanced_data)}", flush=True)
                            print(f"   Columns: {enhanced_columns}", flush=True)
                            
                            # Also keep the original image table for backward compatibility
                            columns = ["image", "rank", "prompt", "completion_preview", "key", 
                                      "knowledge_fidelity", "visual_quality", "reward",
                                      "visual_quality_details", "ground_truth_graph", "predicted_graph"]
                            data = [[item["image"], item.get("rank", 0), item["prompt"], item["completion_preview"], 
                                    item["key"], item["knowledge_fidelity"], item["visual_quality"], 
                                    item["reward"], item.get("visual_quality_details", ""), 
                                    item.get("gt_graph", ""), item.get("pred_graph", "")] for item in wandb_images]
                            
                            table = wandb.Table(columns=columns, data=data)
                            log_dict["generated_images"] = table
                            
                            logger.info(f"Rank 0: Created generated_images table with {len(data)} rows")
                            print(f"📊 Created generated_images table with {len(data)} rows", flush=True)
                            
                            # Add aggregate metrics (these show up in main metrics panel)
                            # Compute from gathered data across all ranks
                            log_dict["generation/avg_knowledge_fidelity"] = sum(knowledge_fidelities) / len(knowledge_fidelities) if knowledge_fidelities else 0.0
                            log_dict["generation/avg_visual_quality"] = sum(visual_qualities) / len(visual_qualities) if visual_qualities else 0.0
                            log_dict["generation/avg_reward"] = sum(rewards) / len(rewards) if rewards else 0.0
                            log_dict["generation/reward_std"] = float(torch.tensor(rewards).std().item()) if len(rewards) > 1 else 0.0
                            log_dict["generation/num_images"] = len(wandb_images)
                            log_dict["generation/num_gpus"] = len(set(item.get("rank", 0) for item in wandb_images))
                            
                            # Add per-GPU metrics
                            from collections import defaultdict
                            gpu_metrics = defaultdict(lambda: {"kf": [], "vq": [], "rewards": []})
                            for item in wandb_images:
                                gpu_rank = item.get("rank", 0)
                                gpu_metrics[gpu_rank]["kf"].append(item["knowledge_fidelity"])
                                gpu_metrics[gpu_rank]["vq"].append(item["visual_quality"])
                                gpu_metrics[gpu_rank]["rewards"].append(item["reward"])
                            
                            for gpu_rank, metrics in gpu_metrics.items():
                                log_dict[f"per_gpu/rank{gpu_rank}/avg_knowledge_fidelity"] = sum(metrics["kf"]) / len(metrics["kf"])
                                log_dict[f"per_gpu/rank{gpu_rank}/avg_visual_quality"] = sum(metrics["vq"]) / len(metrics["vq"])
                                log_dict[f"per_gpu/rank{gpu_rank}/avg_reward"] = sum(metrics["rewards"]) / len(metrics["rewards"])
                                log_dict[f"per_gpu/rank{gpu_rank}/num_samples"] = len(metrics["rewards"])
                            
                            # Add per-sample metrics for detailed tracking
                            for i, item in enumerate(wandb_images):
                                sample_key = item.get("key", f"sample_{i}")
                                log_dict[f"samples/{sample_key}/knowledge_fidelity"] = item["knowledge_fidelity"]
                                log_dict[f"samples/{sample_key}/visual_quality"] = item["visual_quality"]
                                log_dict[f"samples/{sample_key}/reward"] = item["reward"]
                            
                            # Add individual sample images (limit to first 4)
                            for i, item in enumerate(wandb_images[:4]):
                                log_dict[f"sample_{i}/image"] = item["image"]
                                log_dict[f"sample_{i}/knowledge_fidelity"] = item["knowledge_fidelity"]
                                log_dict[f"sample_{i}/visual_quality"] = item["visual_quality"]
                                log_dict[f"sample_{i}/reward"] = item["reward"]
                            
                            # Add reward distributions
                            log_dict["generation/reward_distribution"] = wandb.Histogram(rewards)
                            log_dict["generation/knowledge_fidelity_distribution"] = wandb.Histogram(knowledge_fidelities)
                            log_dict["generation/visual_quality_distribution"] = wandb.Histogram(visual_qualities)
                            
                            # Debug: Print what we're about to log
                            logger.info(f"Rank 0: About to log {len(log_dict)} items to wandb")
                            logger.info(f"Rank 0: Wandb log keys: {list(log_dict.keys())}")
                            print(f"📤 About to log to wandb:", flush=True)
                            print(f"   Keys: {list(log_dict.keys())}", flush=True)
                            
                            # Verify table has correct columns
                            if "completions_with_images" in log_dict:
                                table_obj = log_dict["completions_with_images"]
                                print(f"   completions_with_images: {len(table_obj.data)} rows, {len(table_obj.columns)} columns", flush=True)
                                print(f"   Column names: {table_obj.columns}", flush=True)
                            
                            # Log everything at once (without specifying step - let TRL manage it)
                            # Note: We don't specify step parameter to avoid conflicts with TRL's step tracking
                            wandb.log(log_dict, commit=False)  # commit=False to batch with TRL's logs
                            
                            num_gpus = len(set(item.get("rank", 0) for item in wandb_images))
                            logger.info(f"Rank 0: Uploaded {len(wandb_images)} images from {num_gpus} GPUs to wandb")
                            print(f"☁️  Rank 0: Uploaded {len(wandb_images)} images from {num_gpus} GPUs to wandb", flush=True)
                            print(f"   📊 Tables: completions_with_images ({len(wandb_images)} samples), generated_images", flush=True)
                            print(f"   📈 Metrics: generation/*, per_gpu/rank*/*", flush=True)
                        
                    except Exception as e:
                        logger.warning(f"Rank 0: Failed to upload to wandb: {e}")
                        print(f"⚠️  Rank 0: wandb upload failed: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                elif use_wandb and wandb_images and not should_log_to_wandb and rank == 0:
                    logger.info(f"Rank 0: Skipped wandb upload (step {current_step}, logging every {logging_steps} steps)")
                    print(f"⏭️  Rank 0: Skipped wandb upload (logging at step {(current_step // logging_steps + 1) * logging_steps})", flush=True)
            
            def evaluate_batch(generated_images, knowledge_graphs, annotations, prompts=None, completions=None, keys=None):
                """Evaluate with GPT using MMMG protocol + visual quality"""
                import base64
                from io import BytesIO
                import sys
                
                print(f"\n🔥 MMMG GPT EVALUATOR CALLED with {len(generated_images)} images", flush=True)
                sys.stdout.flush()
                
                rewards = []
                knowledge_fidelities = []
                visual_qualities = []
                visual_quality_details = []  # Store detailed visual quality reasoning
                gt_graphs = []  # Store ground truth graphs for logging
                pred_graphs = []  # Store predicted graphs for logging
                
                for idx, (gen_img, kg_str, annotation) in enumerate(zip(generated_images, knowledge_graphs, annotations)):
                    print(f"   📊 Evaluating sample {idx+1}/{len(generated_images)}...", flush=True)
                    
                    # Convert image to base64
                    def img_to_b64(img):
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        return base64.b64encode(buffered.getvalue()).decode()
                    
                    gen_b64 = img_to_b64(gen_img)
                    
                    try:
                        # Parse ground truth knowledge graph
                        kg = json.loads(kg_str)
                        gt_elements = kg.get("elements", [])
                        gt_dependencies = kg.get("dependencies", [])
                        
                        # ========== STEP 1: Knowledge Fidelity Evaluation ==========
                        # Format MMMG prompt
                        elem_depend_text = str(kg) + "\n" + str(annotation)
                        full_prompt = MMMG_INSTRUCTION_PROMPT.replace("[ELEM_DEPEND]", elem_depend_text)
                        
                        # Call GPT for knowledge graph extraction
                        print("      🔍 Step 1: Knowledge fidelity evaluation...", flush=True)
                        kg_response = client.chat.completions.create(
                            model="gpt-5-mini",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": full_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gen_b64}"}},
                                ]
                            }],
                            max_completion_tokens=2048,
                        )
                        
                        kg_response_text = kg_response.choices[0].message.content
                        
                        # Parse predicted knowledge graph
                        pred_elements, pred_dependencies = parse_mmmg_response(kg_response_text, gt_elements, gt_dependencies)
                        
                        # Compute Graph Edit Distance → Knowledge Fidelity (also returns graphs)
                        knowledge_fidelity, G_gt, G_pred = compute_graph_edit_distance(
                            pred_elements, pred_dependencies,
                            gt_elements, gt_dependencies
                        )
                        
                        # Store graphs for logging
                        gt_graphs.append(serialize_graph(G_gt))
                        pred_graphs.append(serialize_graph(G_pred))
                        
                        print(f"         ✓ Knowledge Fidelity: {knowledge_fidelity:.3f}", flush=True)
                        
                        # ========== STEP 2: Visual Quality Evaluation ==========
                        print("      🎨 Step 2: Visual quality evaluation...", flush=True)
                        quality_response = client.chat.completions.create(
                            model="gpt-5-mini",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": VISUAL_QUALITY_PROMPT},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gen_b64}"}},
                                ]
                            }],
                            max_completion_tokens=1024,
                        )
                        
                        quality_response_text = quality_response.choices[0].message.content
                        visual_quality, vq_details = parse_visual_quality_score(quality_response_text)
                        
                        print(f"         ✓ Visual Quality: {visual_quality:.3f}", flush=True)
                        
                        # ========== Combined Reward ==========
                        # Weighted combination: 70% knowledge fidelity + 30% visual quality
                        reward = alignment_weight * knowledge_fidelity + quality_weight * visual_quality
                        rewards.append(reward)
                        knowledge_fidelities.append(knowledge_fidelity)
                        visual_qualities.append(visual_quality)
                        visual_quality_details.append(json.dumps(vq_details, indent=2))
                        
                        # Detailed logging
                        pred_elem_count = sum(pred_elements.values())
                        pred_dep_count = sum(pred_dependencies.values())
                        
                        print(f"      ✅ Sample {idx+1}: 2KF={knowledge_fidelity:.3f} (E={pred_elem_count}/{len(gt_elements)}, D={pred_dep_count}/{len(gt_dependencies)}), VQ={visual_quality:.3f}, Reward={reward:.3f}", flush=True)
                        logger.info(f"MMMG Eval - Knowledge Fidelity: {knowledge_fidelity:.3f} (Elements: {pred_elem_count}/{len(gt_elements)}, Deps: {pred_dep_count}/{len(gt_dependencies)}), Visual Quality: {visual_quality:.3f}, Final Reward: {reward:.3f}")
                        
                    except Exception as e:
                        print(f"      ❌ Sample {idx+1} failed: {e}", flush=True)
                        print(f"      📍 Error type: {type(e).__name__}", flush=True)
                        print(f"      📍 Error details: {str(e)[:500]}", flush=True)
                        logger.error(f"MMMG GPT eval error: {e}")
                        logger.error(f"Error type: {type(e).__name__}")
                        import traceback
                        error_trace = traceback.format_exc()
                        logger.error(f"Full traceback: {error_trace}")
                        print(f"      📍 Full traceback:\n{error_trace}", flush=True)
                        rewards.append(0.5)
                        knowledge_fidelities.append(0.5)
                        visual_qualities.append(0.5)
                        visual_quality_details.append(json.dumps({"error": str(e)}, indent=2))
                        gt_graphs.append("Error: evaluation failed")
                        pred_graphs.append("Error: evaluation failed")
                
                print(f"🎁 Final rewards: {rewards}", flush=True)
                
                # Save generated images with metadata
                if prompts is not None and completions is not None:
                    save_generated_images(
                        generated_images, prompts, completions, keys,
                        rewards, knowledge_fidelities, visual_qualities,
                        visual_quality_details, gt_graphs, pred_graphs,
                        current_step=reward_step_counter[0]
                    )
                
                return rewards
            
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
            # Force flush logs immediately
            import sys
            print("=" * 100, flush=True)
            print("🎯 MMMG REWARD FUNCTION CALLED!", flush=True)
            print(f"   Prompt batch size: {len(prompt_text)}", flush=True)
            print(f"   Completions: {len(completions)}", flush=True)
            print("=" * 100, flush=True)
            sys.stdout.flush()
            
            batch_size = len(prompt_text)
            logger.info(f"=== MMMG Reward computation for batch of {batch_size} ===")
            
            # Extract completion texts from GRPO (handle different formats)
            print(f"🔍 DEBUG: completions type: {type(completions)}", flush=True)
            print(f"🔍 DEBUG: completions[0] type: {type(completions[0])}", flush=True)
            print(f"🔍 DEBUG: completions[0] content: {str(completions[0])[:200]}", flush=True)
            
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
                        logger.warning(f"Unexpected completion format at index {i}: {type(comp[0])}")
                        completion_texts.append(str(comp))
                elif isinstance(comp, dict) and "content" in comp:
                    # Single message dict
                    completion_texts.append(comp["content"])
                else:
                    logger.warning(f"Unexpected completion format at index {i}: {type(comp)}")
                    completion_texts.append(str(comp))
            
            logger.info(f"Completion text sample: {completion_texts[0][:200]}...")
            print(f"📝 First completion: {completion_texts[0][:100]}...", flush=True)
            
            # Get pipeline
            pipe = get_pipeline()
            
            # STEP 1: Extract embeddings from completion text
            logger.info("Extracting response embeddings from completion text...")
            response_embeds, response_masks = extract_response_embeddings(
                vlm_model=vlm_model[0],
                vlm_processor=vlm_processor,
                prompts=prompt_text,
                completion_texts=completion_texts,
            )
            
            # STEP 2: Generate images using these embeddings
            logger.info("Generating images...")
            
            generated_images = generate_with_embeddings(
                pipe=pipe,
                prompt_embeds=response_embeds,
                prompt_embeds_mask=response_masks,
            )
            
            # STEP 3: Evaluate with MMMG GPT protocol
            print("🔍 STEP 3: Calling MMMG GPT evaluator...", flush=True)
            logger.info("Evaluating with MMMG protocol...")
            evaluator = get_gpt_evaluator()
            print(f"   Evaluator initialized, calling with {len(prompt_text)} samples...", flush=True)
            
            # Extract keys if available in kwargs
            keys = kwargs.get('key', None)
            if keys is not None and not isinstance(keys, list):
                keys = [keys] * len(prompt_text)
            
            # Call evaluator with all metadata for logging
            rewards = evaluator(
                generated_images, 
                knowledge_graph, 
                annotation,
                prompts=prompt_text,
                completions=completion_texts,
                keys=keys
            )
            
            print(f"✅ MMMG GPT Evaluation complete! Rewards: {rewards}", flush=True)
            logger.info(f"Rewards: {rewards}")
            
            # Increment step counter
            reward_step_counter[0] += 1
            return rewards
            
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Increment step counter even on error
            reward_step_counter[0] += 1
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
    
    return prompt_embeds, prompt_masks


def extract_masked_hidden(hidden_states, mask):
    """Extract hidden states using attention mask"""
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def generate_with_embeddings(pipe, prompt_embeds, prompt_embeds_mask):
    """Generate images using pre-computed embeddings"""
    
    # Use pipeline with pre-computed embeddings instead of text prompts
    output = pipe(
        prompt=None,  # Must be None when using prompt_embeds
        negative_prompt=" ",
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        num_inference_steps=50,
    )
    
    return output.images


################
# Main
################
if __name__ == "__main__":
    # Parse args
    parser = TrlParser((ImageGenArgs, GRPOConfig, ModelConfig))
    img_args, training_args, model_args = parser.parse_args_and_config()
    
    # Setup
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.add(
        Path(training_args.output_dir) / "training.log",
        level="INFO",
    )
    
    logger.info("="*80)
    logger.info("GRPO Training: Qwen-Image-Response")
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
        gen_model_path=img_args.gen_model_path,
        vlm_model=vlm_model_ref,  # Pass mutable reference
        vlm_processor=processor,
        api_key=api_key,
        alignment_weight=img_args.alignment_weight,
        quality_weight=img_args.quality_weight,
        n_evals=img_args.n_evals,
        logging_steps=training_args.logging_steps,
        training_args=training_args,
    )
    
    # Initialize GRPO trainer with reward function
    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[reward_fn] + ([think_format_reward] if img_args.use_think_format_reward else []),
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

