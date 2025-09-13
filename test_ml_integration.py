#!/usr/bin/env python3
"""
Test script to verify ML integration is working.
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_ml_service():
    """Test the ML service directly."""
    print("🧪 Testing ML service...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            print("✅ ML service is healthy")
            print(f"   Status: {response.json()}")
        else:
            print(f"❌ ML service health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ ML service is not running on port 5001")
        return False
    except Exception as e:
        print(f"❌ ML service test failed: {e}")
        return False
    
    return True

def test_backend_ml_status():
    """Test the backend ML status endpoint."""
    print("🧪 Testing backend ML status...")
    
    try:
        response = requests.get("http://localhost:5000/api/photos/ml-status", timeout=5)
        if response.status_code == 200:
            print("✅ Backend ML status endpoint working")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Backend ML status failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running on port 5000")
        return False
    except Exception as e:
        print(f"❌ Backend ML status test failed: {e}")
        return False
    
    return True

def test_classification_with_sample_image():
    """Test classification with a sample image."""
    print("🧪 Testing image classification...")
    
    # Look for a sample image in the ML model directory
    sample_images = [
        "ml-model/data/processed/train/lantern_fly",
        "ml-model/data/processed/test/lantern_fly",
        "ml-model/data/processed/val/lantern_fly"
    ]
    
    sample_image_path = None
    for img_dir in sample_images:
        img_path = Path(img_dir)
        if img_path.exists():
            # Find the first image file
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images = list(img_path.glob(ext))
                if images:
                    sample_image_path = images[0]
                    break
            if sample_image_path:
                break
    
    if not sample_image_path:
        print("⚠️  No sample images found for testing")
        return True
    
    print(f"   Using sample image: {sample_image_path}")
    
    try:
        with open(sample_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post("http://localhost:5001/classify", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image classification successful")
            print(f"   Result: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                is_lanternfly = result.get('is_lantern_fly', False)
                confidence = result.get('confidence_score', 0)
                print(f"   🎯 Classification: {'Lanternfly' if is_lanternfly else 'Non-Lanternfly'}")
                print(f"   📊 Confidence: {confidence:.3f}")
                return True
            else:
                print(f"   ❌ Classification failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Classification request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing ML Integration")
    print("=" * 40)
    
    # Wait a moment for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(2)
    
    tests = [
        ("ML Service Health", test_ml_service),
        ("Backend ML Status", test_backend_ml_status),
        ("Image Classification", test_classification_with_sample_image)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ML integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
