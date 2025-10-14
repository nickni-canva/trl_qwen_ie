"""
Test wandb table creation to verify the fix
"""

import os
os.environ["WANDB_MODE"] = "offline"  # Test without actual upload

try:
    import wandb
    print("✅ wandb imported successfully")
except ImportError:
    print("❌ wandb not installed")
    exit(1)

from PIL import Image
import numpy as np

print("\n" + "="*80)
print("TEST: Wandb Table Creation")
print("="*80)

# Create fake data similar to what we use in training
print("\n1. Creating sample images...")
sample_images = []
for i in range(3):
    # Create a small random image
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    sample_images.append(img)
print(f"   Created {len(sample_images)} sample images")

# Create wandb Images
print("\n2. Creating wandb.Image objects...")
wandb_images = []
for i, img in enumerate(sample_images):
    wandb_img = wandb.Image(img)
    wandb_images.append({
        "image": wandb_img,
        "rank": 0,
        "prompt": f"Test prompt {i}",
        "completion_preview": f"Test completion {i}",
        "key": f"test_key_{i}",
        "knowledge_fidelity": 0.8 + i * 0.05,
        "visual_quality": 0.75 + i * 0.05,
        "reward": 0.78 + i * 0.05,
    })
print(f"   Created {len(wandb_images)} wandb image objects")

# Create table
print("\n3. Creating wandb.Table...")
try:
    columns = ["image", "rank", "prompt", "completion_preview", "key", 
              "knowledge_fidelity", "visual_quality", "reward"]
    data = [[item["image"], item["rank"], item["prompt"], item["completion_preview"], 
            item["key"], item["knowledge_fidelity"], item["visual_quality"], 
            item["reward"]] for item in wandb_images]
    
    table = wandb.Table(columns=columns, data=data)
    print(f"   ✅ Table created successfully")
    print(f"   Columns: {table.columns}")
    print(f"   Rows: {len(table.data)}")
except Exception as e:
    print(f"   ❌ Failed to create table: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test logging everything together
print("\n4. Testing combined logging...")
try:
    # Initialize wandb in offline mode
    wandb.init(project="test", mode="offline")
    
    log_dict = {}
    
    # Add table
    log_dict["rank_0/generated_images"] = table
    
    # Add metrics
    log_dict["rank_0/avg_knowledge_fidelity"] = 0.825
    log_dict["rank_0/avg_visual_quality"] = 0.775
    log_dict["rank_0/avg_reward"] = 0.805
    log_dict["rank_0/reward_std"] = 0.05
    log_dict["rank_0/num_images"] = len(wandb_images)
    
    # Add individual sample images
    for i, item in enumerate(wandb_images[:2]):
        log_dict[f"rank_0/sample_{i}/image"] = item["image"]
        log_dict[f"rank_0/sample_{i}/knowledge_fidelity"] = item["knowledge_fidelity"]
        log_dict[f"rank_0/sample_{i}/visual_quality"] = item["visual_quality"]
        log_dict[f"rank_0/sample_{i}/reward"] = item["reward"]
    
    # Add histograms
    rewards = [0.78, 0.83, 0.88]
    log_dict["rank_0/reward_distribution"] = wandb.Histogram(rewards)
    
    print(f"   Log dict keys: {list(log_dict.keys())}")
    print(f"   Total items to log: {len(log_dict)}")
    
    # Log everything at once
    wandb.log(log_dict)
    print(f"   ✅ Logged successfully")
    
    wandb.finish()
    
except Exception as e:
    print(f"   ❌ Failed to log: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("   - wandb.Table created successfully")
print("   - Combined logging works")
print("   - Table upload should work in training")
print("="*80)

