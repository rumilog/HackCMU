#!/usr/bin/env python3
"""
Simple test script to verify the classification is working.
"""

import requests
import json
import os
from pathlib import Path

def test_classification():
    """Test the classification endpoint."""
    print("🧪 Testing classification endpoint...")
    
    # Look for a sample image
    sample_dirs = [
        "ml-model/data/processed/train/lantern_fly",
        "ml-model/data/processed/test/lantern_fly",
        "ml-model/data/processed/val/lantern_fly"
    ]
    
    sample_image = None
    for sample_dir in sample_dirs:
        if os.path.exists(sample_dir):
            images = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
            if images:
                sample_image = os.path.join(sample_dir, images[0])
                break
    
    if not sample_image:
        print("❌ No sample images found")
        return False
    
    print(f"📸 Using sample image: {sample_image}")
    
    try:
        # Test the backend classification endpoint
        with open(sample_image, 'rb') as f:
            files = {'image': (os.path.basename(sample_image), f, 'image/jpeg')}
            response = requests.post("http://localhost:5000/api/photos/classify", files=files, timeout=30)
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response content: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Classification successful!")
            print(f"   Result: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Classification failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_classification()
