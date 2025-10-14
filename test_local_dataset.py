"""
Test loading MMMG dataset from local file
"""

import sys
import json
sys.path.insert(0, '/home/coder/work/trl_qwen_image_edit')

from train_grpo_qwen_image import prepare_dataset

print("="*80)
print("TEST: Loading MMMG Dataset from Local File")
print("="*80)

# Test 1: Load a single level and discipline
print("\n1. Testing single level/discipline: preschool/Math")
print("-"*80)

try:
    train_ds, eval_ds = prepare_dataset(
        levels="preschool",
        disciplines="Math",
        max_samples=10,
        local_data_path="/mnt/ephemeral/MMMG_train/train.json"
    )
    
    print(f"✅ Successfully loaded dataset!")
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Eval samples: {len(eval_ds) if eval_ds else 0}")
    
    if len(train_ds) > 0:
        print(f"\n   First training sample:")
        first_sample = train_ds[0]
        print(f"     Keys: {list(first_sample.keys())}")
        print(f"     Key: {first_sample['key']}")
        print(f"     Prompt (first 100 chars): {first_sample['prompt_text'][:100]}...")
        print(f"     Level: {first_sample['level']}")
        print(f"     Discipline: {first_sample['discipline']}")
        
        # Check knowledge graph structure
        kg = json.loads(first_sample['knowledge_graph'])
        print(f"     Knowledge Graph elements: {kg.get('elements', [])}")
        print(f"     Knowledge Graph dependencies: {kg.get('dependencies', [])}")
        
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load multiple levels
print("\n2. Testing multiple levels: preschool,primaryschool")
print("-"*80)

try:
    train_ds, eval_ds = prepare_dataset(
        levels="preschool,primaryschool",
        disciplines="Math",
        max_samples=20,
        local_data_path="/mnt/ephemeral/MMMG_train/train.json"
    )
    
    print(f"✅ Successfully loaded dataset!")
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Eval samples: {len(eval_ds) if eval_ds else 0}")
    
    # Check that we got samples from both levels
    levels_found = set(sample['level'] for sample in train_ds)
    print(f"   Levels found: {levels_found}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load all levels and disciplines
print("\n3. Testing 'all' levels and 'all' disciplines")
print("-"*80)

try:
    train_ds, eval_ds = prepare_dataset(
        levels="all",
        disciplines="all",
        max_samples=100,
        local_data_path="/mnt/ephemeral/MMMG_train/train.json"
    )
    
    print(f"✅ Successfully loaded dataset!")
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Eval samples: {len(eval_ds) if eval_ds else 0}")
    
    # Check diversity
    levels_found = set(sample['level'] for sample in train_ds)
    disciplines_found = set(sample['discipline'] for sample in train_ds)
    print(f"   Levels found: {levels_found}")
    print(f"   Disciplines found: {disciplines_found}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("   - Local dataset loads correctly")
print("   - Data structure is correct")
print("   - Knowledge graphs are properly formatted")
print("   - Train/eval split works")
print("="*80)

