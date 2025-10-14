#!/usr/bin/env python3
"""
Test GPT evaluation to see why Visual Quality defaults to 0.5
"""

import os
import json
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not set in environment")
    exit(1)

client = OpenAI(api_key=api_key)

# Create a simple test image
def create_test_image():
    """Create a simple colored image for testing"""
    img = Image.new('RGB', (512, 512), color=(73, 109, 137))
    return img

# Test the model names
def test_model_names():
    """Test if gpt-5-mini exists"""
    print("Testing model names...")
    
    test_img = create_test_image()
    buffered = BytesIO()
    test_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    models_to_test = [
        "gpt-5-mini",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
    ]
    
    for model_name in models_to_test:
        print(f"\n🔍 Testing model: {model_name}")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Just say the color."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ]
                }],
                max_completion_tokens=50,
            )
            result = response.choices[0].message.content
            print(f"   ✅ {model_name} works! Response: {result}")
            return model_name  # Return first working model
            
        except Exception as e:
            print(f"   ❌ {model_name} failed: {e}")
    
    return None

# Test Visual Quality evaluation
VISUAL_QUALITY_PROMPT = """You are an expert in evaluating educational image quality. Analyze this image across these dimensions:

1. **Clarity**: Are objects, text, and details clearly visible and well-defined?
2. **Composition**: Is the layout well-organized and visually balanced?
3. **Color Usage**: Are colors appropriate, harmonious, and enhance understanding?
4. **Relevance**: Does the image appear suitable for educational content?
5. **Detail Level**: Is there appropriate detail without clutter?
6. **Overall Coherence**: Do all elements work together effectively?

Provide your assessment in this format:
Clarity: [score 0-10]
Composition: [score 0-10]
Color: [score 0-10]
Relevance: [score 0-10]
Detail: [score 0-10]
Coherence: [score 0-10]

Overall Quality Score: [average score 0-10]"""

def parse_visual_quality_score(text):
    """Parse the visual quality score from GPT response"""
    import re
    
    print(f"\n📝 GPT Response:\n{text}\n")
    
    # Look for "Overall Quality Score: X" or "Overall: X"
    patterns = [
        r"Overall Quality Score:\s*([0-9.]+)",
        r"Overall:\s*([0-9.]+)",
        r"overall.*?([0-9.]+)\s*(?:/\s*10)?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # Normalize to 0-1 if it's 0-10
            if score > 1.0:
                score = score / 10.0
            print(f"✅ Extracted score: {score}")
            return score
    
    print(f"⚠️  Could not parse score, returning 0.5")
    return 0.5

def test_visual_quality_evaluation(model_name):
    """Test the visual quality evaluation"""
    print(f"\n🎨 Testing Visual Quality Evaluation with {model_name}...")
    
    test_img = create_test_image()
    buffered = BytesIO()
    test_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISUAL_QUALITY_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ]
            }],
            max_completion_tokens=1024,
        )
        
        response_text = response.choices[0].message.content
        visual_quality = parse_visual_quality_score(response_text)
        
        print(f"\n✅ Visual Quality Score: {visual_quality}")
        return visual_quality
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=" * 80)
    print("GPT EVALUATION TEST")
    print("=" * 80)
    
    # Step 1: Find a working model
    working_model = test_model_names()
    
    if working_model:
        print(f"\n✅ Found working model: {working_model}")
        
        # Step 2: Test visual quality evaluation
        score = test_visual_quality_evaluation(working_model)
        
        if score is not None:
            print(f"\n✅ Test passed! Visual quality score: {score}")
        else:
            print(f"\n❌ Visual quality evaluation failed")
    else:
        print("\n❌ No working model found!")
    
    print("\n" + "=" * 80)

