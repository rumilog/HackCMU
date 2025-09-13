#!/usr/bin/env python3
"""
Collect Negative Images for Training
Helps collect general images to use as negative examples
"""

import os
import shutil
import requests
from pathlib import Path
from typing import List
import json
from datetime import datetime

def create_negative_image_collection_guide():
    """Create a guide for collecting negative images"""
    
    guide = {
        "title": "Negative Image Collection Guide",
        "purpose": "Collect general images to use as negative examples for lantern fly classification",
        "target_count": "2000-3000 images",
        "strategy": "Use diverse, general images that are clearly not lantern flies",
        
        "image_sources": {
            "recommended": [
                "Personal photos (landscapes, objects, people, etc.)",
                "Free stock photos (Unsplash, Pexels, Pixabay)",
                "Public domain images",
                "Your own general photo collection"
            ],
            "avoid": [
                "Any insect images",
                "Copyrighted material",
                "Low-quality or blurry images",
                "Images that could be confused with insects"
            ]
        },
        
        "image_categories": {
            "nature": [
                "Trees, leaves, flowers",
                "Landscapes, mountains, rivers",
                "Sky, clouds, sunsets",
                "Rocks, stones, soil"
            ],
            "objects": [
                "Cars, buildings, furniture",
                "Food, drinks, utensils",
                "Books, papers, electronics",
                "Clothing, shoes, bags"
            ],
            "people": [
                "Portraits (with permission)",
                "Activities, sports",
                "Hands, feet, body parts",
                "Groups, crowds"
            ],
            "abstract": [
                "Textures, patterns",
                "Colors, gradients",
                "Shapes, geometric forms",
                "Art, paintings"
            ]
        },
        
        "quality_requirements": {
            "resolution": "Minimum 224x224 pixels",
            "format": "JPG, PNG, or other common formats",
            "quality": "Clear, well-lit, in focus",
            "diversity": "Variety in content, colors, and styles"
        },
        
        "collection_methods": {
            "method_1": {
                "name": "Personal Photos",
                "description": "Use your own photos",
                "pros": ["No copyright issues", "Relevant to your context", "Easy to access"],
                "cons": ["Limited quantity", "May not be diverse enough"],
                "steps": [
                    "Go through your phone/computer photos",
                    "Select diverse, clear images",
                    "Avoid any insect-related photos",
                    "Copy to data/processed/train/non_lantern_fly/"
                ]
            },
            "method_2": {
                "name": "Free Stock Photos",
                "description": "Download from free stock photo sites",
                "pros": ["Large quantity", "High quality", "Diverse content"],
                "cons": ["Requires internet", "May have usage restrictions"],
                "sites": [
                    "https://unsplash.com",
                    "https://www.pexels.com",
                    "https://pixabay.com",
                    "https://www.freepik.com"
                ],
                "steps": [
                    "Search for general categories (nature, objects, people)",
                    "Download high-quality images",
                    "Ensure they're free to use",
                    "Save to data/processed/train/non_lantern_fly/"
                ]
            },
            "method_3": {
                "name": "Mixed Approach",
                "description": "Combine personal photos with stock photos",
                "pros": ["Best of both worlds", "Diverse and relevant"],
                "cons": ["More time-consuming"],
                "steps": [
                    "Start with personal photos (500-1000 images)",
                    "Add stock photos (1000-2000 images)",
                    "Ensure good mix of categories",
                    "Verify no insect images included"
                ]
            }
        },
        
        "organization": {
            "naming": "negative_YYYYMMDD_HHMMSS_category.jpg",
            "examples": [
                "negative_20241209_143022_landscape_001.jpg",
                "negative_20241209_143045_object_002.jpg",
                "negative_20241209_143108_person_003.jpg"
            ],
            "directory": "data/processed/train/non_lantern_fly/",
            "metadata": {
                "source": "Personal photos, stock photos, etc.",
                "category": "Nature, objects, people, abstract",
                "date_collected": "YYYY-MM-DD",
                "quality_check": "Passed"
            }
        },
        
        "quality_control": {
            "immediate_checks": [
                "Image is clear and in focus",
                "No insects visible in image",
                "Good lighting and contrast",
                "Appropriate resolution (224x224+)"
            ],
            "batch_review": [
                "Remove any images with insects",
                "Ensure diversity in content",
                "Check for consistent quality",
                "Verify no copyright issues"
            ],
            "final_validation": [
                "Count total images (target: 2000-3000)",
                "Check for good distribution across categories",
                "Verify quality standards met",
                "Ensure no lantern fly images included"
            ]
        },
        
        "quick_start": {
            "step_1": "Create directory: data/processed/train/non_lantern_fly/",
            "step_2": "Collect 500-1000 personal photos",
            "step_3": "Download 1000-2000 stock photos",
            "step_4": "Organize and rename files",
            "step_5": "Run quality control checks"
        }
    }
    
    return guide

def save_guide_to_file(guide, filename="negative_image_collection_guide.json"):
    """Save guide to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(guide, f, indent=2, ensure_ascii=False)
    print(f"Negative image collection guide saved to: {filename}")

def print_guide_summary(guide):
    """Print a summary of the guide"""
    print("=" * 70)
    print(f"📸 {guide['title']}")
    print("=" * 70)
    print(f"🎯 Target: {guide['target_count']} images")
    print(f"📅 Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("📋 QUICK START:")
    print("-" * 40)
    for step, instruction in guide['quick_start'].items():
        print(f"□ {instruction}")
    print()
    
    print("📁 DIRECTORY TO USE:")
    print(f"   {guide['organization']['directory']}")
    print()
    
    print("📝 NAMING CONVENTION:")
    print("   negative_YYYYMMDD_HHMMSS_category.jpg")
    print("   Example: negative_20241209_143022_landscape_001.jpg")
    print()
    
    print("🖼️  RECOMMENDED SOURCES:")
    print("-" * 40)
    for source in guide['image_sources']['recommended']:
        print(f"□ {source}")
    print()
    
    print("🚫 AVOID:")
    print("-" * 40)
    for item in guide['image_sources']['avoid']:
        print(f"□ {item}")
    print()
    
    print("📊 IMAGE CATEGORIES:")
    print("-" * 40)
    for category, items in guide['image_categories'].items():
        print(f"□ {category.title()}: {', '.join(items[:3])}...")
    print()
    
    print("🔍 QUALITY CHECKS:")
    print("-" * 40)
    for check in guide['quality_control']['immediate_checks']:
        print(f"□ {check}")
    print()
    
    print("🚀 READY TO START COLLECTING!")
    print("=" * 70)

def create_directory_structure():
    """Create the directory structure for negative images"""
    
    print("📁 Creating directory structure...")
    
    # Create main directory
    negative_dir = Path("data/processed/train/non_lantern_fly")
    negative_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory: {negative_dir}")
    
    # Create subdirectories for organization
    subdirs = ["nature", "objects", "people", "abstract", "mixed"]
    for subdir in subdirs:
        subdir_path = negative_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"✅ Created subdirectory: {subdir_path}")
    
    return negative_dir

def download_sample_images():
    """Download a few sample images to get started"""
    
    print("\n📥 Downloading sample images...")
    
    # Sample image URLs (free stock photos)
    sample_urls = [
        {
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
            "filename": "negative_20241209_143022_landscape_001.jpg",
            "category": "nature"
        },
        {
            "url": "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400", 
            "filename": "negative_20241209_143045_object_002.jpg",
            "category": "objects"
        },
        {
            "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
            "filename": "negative_20241209_143108_person_003.jpg", 
            "category": "people"
        }
    ]
    
    negative_dir = Path("data/processed/train/non_lantern_fly")
    
    for sample in sample_urls:
        try:
            response = requests.get(sample["url"], timeout=10)
            if response.status_code == 200:
                file_path = negative_dir / sample["filename"]
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded: {sample['filename']}")
            else:
                print(f"❌ Failed to download: {sample['filename']}")
        except Exception as e:
            print(f"❌ Error downloading {sample['filename']}: {e}")
    
    print(f"\n📁 Sample images saved to: {negative_dir}")

def main():
    """Main function"""
    print("📸 Negative Image Collection Setup")
    print("=" * 50)
    
    # Generate guide
    guide = create_negative_image_collection_guide()
    
    # Save guide
    save_guide_to_file(guide, "negative_image_collection_guide.json")
    
    # Print summary
    print_guide_summary(guide)
    
    # Create directory structure
    create_directory_structure()
    
    # Download sample images
    download_sample_images()
    
    print("\n📚 Files created:")
    print("  - negative_image_collection_guide.json")
    print("  - data/processed/train/non_lantern_fly/ (with sample images)")
    print("\n🎯 You're ready to start collecting negative images!")
    print("\n💡 TIP: Start with your personal photos, then add stock photos")

if __name__ == "__main__":
    main()
