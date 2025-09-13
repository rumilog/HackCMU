#!/usr/bin/env python3
"""
Setup Binary Classification for Lantern Fly Detection
Organizes data for: Lantern Fly vs Everything Else
"""

import os
import shutil
from pathlib import Path
from typing import List
import json
from datetime import datetime

def setup_binary_classification():
    """Set up the binary classification data structure"""
    
    print("🎯 SETTING UP BINARY CLASSIFICATION")
    print("=" * 50)
    print("Model: Lantern Fly vs Everything Else")
    print("Training: 500 lantern fly photos + 2000+ general images")
    print()
    
    # Create directory structure
    base_dir = Path("data/processed/train")
    
    # Lantern fly directory (for your 500 photos)
    lf_dir = base_dir / "lantern_fly"
    lf_dir.mkdir(parents=True, exist_ok=True)
    
    # Non-lantern fly directory (for everything else)
    nlf_dir = base_dir / "non_lantern_fly"
    nlf_dir.mkdir(parents=True, exist_ok=True)
    
    print("📁 DIRECTORY STRUCTURE CREATED:")
    print(f"✅ {lf_dir}")
    print(f"✅ {nlf_dir}")
    print()
    
    # Create subdirectories for organization
    nlf_subdirs = ["general_photos", "other_insects", "objects", "nature", "people"]
    for subdir in nlf_subdirs:
        subdir_path = nlf_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"✅ Created: {subdir_path}")
    
    print()
    
    # Create README files
    create_readme_files(lf_dir, nlf_dir)
    
    # Create data collection checklist
    create_collection_checklist()
    
    print("🎉 BINARY CLASSIFICATION SETUP COMPLETE!")
    print("=" * 50)

def create_readme_files(lf_dir: Path, nlf_dir: Path):
    """Create README files for each directory"""
    
    # Lantern fly README
    lf_readme = """# Lantern Fly Photos

This directory should contain 500+ photos of dead/squashed spotted lantern flies.

## Requirements:
- Subject: Dead or squashed spotted lantern flies (Lycorma delicatula)
- Quality: Sharp focus, good lighting, clear visibility
- Variety: Different angles, conditions, backgrounds
- Format: JPG, PNG, or other common formats
- Resolution: Minimum 224x224 pixels

## Naming Convention:
lantern_fly_YYYYMMDD_HHMMSS_location.jpg

## Examples:
- lantern_fly_20241209_143022_sidewalk_001.jpg
- lantern_fly_20241209_143045_parking_lot_002.jpg

## Collection Guide:
See: photo_collection_guide.md for detailed instructions.
"""
    
    with open(lf_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(lf_readme)
    
    # Non-lantern fly README
    nlf_readme = """# Non-Lantern Fly Images

This directory should contain 2000+ images of everything that is NOT a squashed lantern fly.

## What to Include:
- General photos (landscapes, objects, people, etc.)
- Other insects (alive or dead, but not lantern flies)
- Buildings, cars, food, nature scenes
- Any clear, well-lit images

## What to Avoid:
- Squashed lantern flies (those go in lantern_fly/)
- Blurry or low-quality images
- Images that could be confused with insects

## Organization:
- general_photos/ - Personal photos, stock photos
- other_insects/ - Any insects that are not lantern flies
- objects/ - Cars, buildings, furniture, etc.
- nature/ - Trees, flowers, landscapes
- people/ - Portraits, activities (with permission)

## Naming Convention:
negative_YYYYMMDD_HHMMSS_category.jpg

## Examples:
- negative_20241209_143022_landscape_001.jpg
- negative_20241209_143045_object_002.jpg
- negative_20241209_143108_insect_003.jpg

## Collection Guide:
See: negative_image_collection_guide.json for detailed instructions.
"""
    
    with open(nlf_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(nlf_readme)
    
    print("📝 README files created for both directories")

def create_collection_checklist():
    """Create a data collection checklist"""
    
    checklist = {
        "title": "Binary Classification Data Collection Checklist",
        "model_type": "Lantern Fly vs Everything Else",
        "created": datetime.now().isoformat(),
        
        "lantern_fly_photos": {
            "target_count": 500,
            "current_count": 0,
            "directory": "data/processed/train/lantern_fly/",
            "requirements": [
                "Dead or squashed spotted lantern flies only",
                "Sharp focus and good lighting",
                "Various angles and backgrounds",
                "Minimum 224x224 pixels",
                "JPG, PNG, or other common formats"
            ],
            "collection_guide": "photo_collection_guide.md",
            "status": "Not started"
        },
        
        "non_lantern_fly_images": {
            "target_count": 2000,
            "current_count": 0,
            "directory": "data/processed/train/non_lantern_fly/",
            "requirements": [
                "Everything that is NOT a squashed lantern fly",
                "General photos, other insects, objects, nature",
                "Clear, well-lit images",
                "Minimum 224x224 pixels",
                "Diverse content and styles"
            ],
            "collection_guide": "negative_image_collection_guide.json",
            "status": "Not started",
            "subdirectories": {
                "general_photos": "Personal photos, stock photos",
                "other_insects": "Any insects that are not lantern flies",
                "objects": "Cars, buildings, furniture, etc.",
                "nature": "Trees, flowers, landscapes",
                "people": "Portraits, activities (with permission)"
            }
        },
        
        "data_balance": {
            "target_ratio": "1:4 (lantern_fly : non_lantern_fly)",
            "augmentation": "4x augmentation for lantern fly images",
            "final_ratio": "1:2 (after augmentation)"
        },
        
        "quality_control": {
            "lantern_fly": [
                "Verify all images show dead/squashed lantern flies",
                "Check focus and lighting quality",
                "Ensure variety in conditions and locations",
                "Remove blurry or poor quality images"
            ],
            "non_lantern_fly": [
                "Verify no lantern fly images included",
                "Check for diverse content",
                "Ensure good image quality",
                "Remove any confusing or low-quality images"
            ]
        },
        
        "next_steps": [
            "1. Collect 500 lantern fly photos",
            "2. Collect 2000+ non-lantern fly images",
            "3. Run data processing script",
            "4. Train the binary classification model",
            "5. Evaluate model performance"
        ]
    }
    
    # Save checklist
    with open("binary_classification_checklist.json", 'w', encoding='utf-8') as f:
        json.dump(checklist, f, indent=2, ensure_ascii=False)
    
    print("📋 Data collection checklist created: binary_classification_checklist.json")

def print_setup_summary():
    """Print a summary of the setup"""
    
    print("\n📊 SETUP SUMMARY:")
    print("-" * 30)
    print("✅ Binary classification structure created")
    print("✅ Lantern fly directory: data/processed/train/lantern_fly/")
    print("✅ Non-lantern fly directory: data/processed/train/non_lantern_fly/")
    print("✅ README files created for both directories")
    print("✅ Collection checklist created")
    print()
    
    print("🎯 YOUR TASK:")
    print("-" * 30)
    print("1. 📸 Collect 500 squashed lantern fly photos")
    print("   → Place in: data/processed/train/lantern_fly/")
    print("   → Follow: photo_collection_guide.md")
    print()
    print("2. 🖼️  Collect 2000+ general images")
    print("   → Place in: data/processed/train/non_lantern_fly/")
    print("   → Use personal photos + free stock photos")
    print("   → Follow: negative_image_collection_guide.json")
    print()
    print("3. 🚀 Train your model")
    print("   → Run: python scripts/train_model.py")
    print()
    
    print("💡 TIP: Start with lantern fly photos first!")
    print("   They're the most important for your model.")

def main():
    """Main function"""
    setup_binary_classification()
    print_setup_summary()

if __name__ == "__main__":
    main()
