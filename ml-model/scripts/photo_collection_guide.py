#!/usr/bin/env python3
"""
Photo Collection Guide for Lantern Fly Dataset
Generates a comprehensive guide for collecting 500 squashed lantern fly photos
"""

import os
from pathlib import Path
from datetime import datetime
import json

def generate_photo_collection_guide():
    """Generate a comprehensive photo collection guide"""
    
    guide = {
        "title": "Lantern Fly Photo Collection Guide",
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "target": "500 squashed lantern fly photos",
        "purpose": "Training data for binary classification model",
        
        "overview": {
            "goal": "Collect 500 high-quality photos of dead/squashed lantern flies",
            "use_case": "Training a machine learning model to identify lantern flies",
            "classification": "Binary: Lantern Fly vs Non-Lantern Fly",
            "timeline": "1-2 days of focused collection"
        },
        
        "photo_requirements": {
            "quality": {
                "resolution": "Minimum 224x224 pixels (preferably 512x512 or higher)",
                "format": "JPG, PNG, or other common image formats",
                "focus": "Sharp focus on the insect, avoid blurry images",
                "lighting": "Good lighting, avoid shadows or overexposure",
                "composition": "Insect should fill at least 30% of the frame"
            },
            "content": {
                "subject": "Dead or squashed spotted lantern flies (Lycorma delicatula)",
                "condition": "Various states of damage (freshly dead to heavily squashed)",
                "background": "Diverse backgrounds (sidewalks, roads, leaves, etc.)",
                "angle": "Multiple angles (top view, side view, close-up)",
                "size": "Include both adult and nymph lantern flies if possible"
            },
            "exclusions": {
                "avoid": [
                    "Live lantern flies",
                    "Other insect species",
                    "Blurry or out-of-focus images",
                    "Images where the insect is too small",
                    "Images with multiple insects (focus on single insects)"
                ]
            }
        },
        
        "collection_strategy": {
            "locations": {
                "high_priority": [
                    "Sidewalks and walkways",
                    "Parking lots",
                    "Roads and bike paths",
                    "Outdoor seating areas",
                    "Building entrances and exits"
                ],
                "medium_priority": [
                    "Tree bases and trunks",
                    "Outdoor furniture",
                    "Playground equipment",
                    "Sports fields and courts"
                ],
                "low_priority": [
                    "Natural areas (leaves, grass)",
                    "Water features",
                    "Indoor locations"
                ]
            },
            "timing": {
                "best_times": [
                    "Early morning (7-9 AM)",
                    "Late afternoon (4-6 PM)",
                    "After rain (when insects are more visible)"
                ],
                "weather_conditions": [
                    "Clear or partly cloudy days",
                    "Avoid heavy rain or strong winds",
                    "Good lighting conditions"
                ]
            },
            "systematic_approach": {
                "grid_search": "Divide campus into sections and search systematically",
                "high_traffic_areas": "Focus on areas with high foot traffic",
                "variety": "Ensure diversity in locations and conditions",
                "documentation": "Note location and conditions for each photo"
            }
        },
        
        "photo_techniques": {
            "camera_settings": {
                "mode": "Auto mode is fine, but manual focus if possible",
                "flash": "Use natural lighting when possible, flash if needed",
                "stability": "Hold camera steady, use two hands if needed"
            },
            "composition": {
                "framing": "Fill frame with insect, leave some background context",
                "angle": "Take multiple angles (top, side, close-up)",
                "distance": "Close enough to see details, far enough for context",
                "background": "Include some background for context"
            },
            "quality_checks": {
                "before_moving": [
                    "Check image is in focus",
                    "Verify insect is clearly visible",
                    "Ensure good lighting",
                    "Check composition"
                ],
                "review_later": [
                    "Delete blurry or poor quality images",
                    "Keep best 2-3 images per insect",
                    "Organize by location or date"
                ]
            }
        },
        
        "data_organization": {
            "naming_convention": "lantern_fly_YYYYMMDD_HHMMSS_location.jpg",
            "examples": [
                "lantern_fly_20241209_143022_sidewalk_001.jpg",
                "lantern_fly_20241209_143045_parking_lot_002.jpg",
                "lantern_fly_20241209_143108_tree_base_003.jpg"
            ],
            "metadata": {
                "capture_date": "YYYY-MM-DD",
                "capture_time": "HH:MM:SS",
                "location": "General area description",
                "condition": "Fresh, moderately damaged, heavily squashed",
                "size": "Adult, nymph, or unknown",
                "background": "Sidewalk, road, leaf, etc."
            }
        },
        
        "quality_control": {
            "immediate_checks": [
                "Image is in focus",
                "Insect is clearly visible",
                "Good lighting and contrast",
                "Appropriate composition"
            ],
            "batch_review": [
                "Remove duplicate or very similar images",
                "Ensure variety in conditions and locations",
                "Check for consistent quality",
                "Verify all images show lantern flies"
            ],
            "final_validation": [
                "Count total images (target: 500)",
                "Check for diversity in locations",
                "Verify quality standards met",
                "Organize into final dataset"
            ]
        },
        
        "safety_considerations": {
            "general": [
                "Be aware of traffic when photographing on roads",
                "Watch for uneven surfaces and obstacles",
                "Stay hydrated and take breaks",
                "Work in pairs if possible for safety"
            ],
            "campus_specific": [
                "Follow campus safety guidelines",
                "Be respectful of campus activities",
                "Avoid disrupting classes or events",
                "Stay in public areas"
            ]
        },
        
        "equipment_checklist": {
            "essential": [
                "Smartphone or camera",
                "Fully charged battery",
                "Extra storage space (at least 2GB)",
                "Comfortable walking shoes"
            ],
            "recommended": [
                "Portable charger",
                "Small notebook for notes",
                "Water bottle",
                "Sun protection"
            ],
            "optional": [
                "Camera tripod or stabilizer",
                "Extra memory cards",
                "Lens cleaning cloth"
            ]
        },
        
        "progress_tracking": {
            "daily_goals": {
                "day_1": "200-250 photos",
                "day_2": "250-300 photos (total: 500)"
            },
            "milestones": [
                "100 photos: 20% complete",
                "250 photos: 50% complete",
                "400 photos: 80% complete",
                "500 photos: 100% complete"
            ],
            "quality_metrics": [
                "Focus quality: >90% sharp images",
                "Lighting quality: >80% well-lit images",
                "Composition: >70% well-composed images",
                "Diversity: Photos from at least 10 different locations"
            ]
        },
        
        "troubleshooting": {
            "common_issues": {
                "blurry_images": "Hold camera steady, use two hands, check focus",
                "poor_lighting": "Move to better lighting, use flash if needed",
                "small_insects": "Get closer, use zoom if available",
                "similar_photos": "Vary angles and distances, include different backgrounds"
            },
            "quality_improvement": {
                "focus": "Tap to focus on the insect, wait for focus confirmation",
                "lighting": "Position yourself to avoid shadows, use natural light",
                "composition": "Fill frame with insect, leave some background context",
                "stability": "Brace against objects, use two hands, take multiple shots"
            }
        },
        
        "post_collection": {
            "immediate_steps": [
                "Backup photos to computer/cloud",
                "Review and delete poor quality images",
                "Organize photos by location or date",
                "Count total images collected"
            ],
            "preparation_for_training": [
                "Ensure 500+ high-quality images",
                "Verify diversity in locations and conditions",
                "Organize into final dataset structure",
                "Create backup copies"
            ],
            "next_steps": [
                "Run data processing script",
                "Begin model training",
                "Monitor training progress",
                "Evaluate model performance"
            ]
        }
    }
    
    return guide

def save_guide_to_file(guide, filename="photo_collection_guide.json"):
    """Save guide to JSON file"""
    with open(filename, 'w') as f:
        json.dump(guide, f, indent=2)
    print(f"Photo collection guide saved to: {filename}")

def print_guide_summary(guide):
    """Print a summary of the guide"""
    print("=" * 60)
    print(f"📸 {guide['title']}")
    print("=" * 60)
    print(f"🎯 Target: {guide['target']}")
    print(f"📅 Created: {guide['created']}")
    print()
    
    print("📋 QUICK CHECKLIST:")
    print("□ Smartphone/camera with charged battery")
    print("□ At least 2GB free storage space")
    print("□ Comfortable walking shoes")
    print("□ Water bottle and sun protection")
    print()
    
    print("🎯 DAILY GOALS:")
    for day, goal in guide['progress_tracking']['daily_goals'].items():
        print(f"□ {day.replace('_', ' ').title()}: {goal}")
    print()
    
    print("📍 HIGH-PRIORITY LOCATIONS:")
    for location in guide['collection_strategy']['locations']['high_priority']:
        print(f"□ {location}")
    print()
    
    print("⏰ BEST TIMES:")
    for time in guide['collection_strategy']['timing']['best_times']:
        print(f"□ {time}")
    print()
    
    print("🔍 QUALITY CHECKS:")
    for check in guide['quality_control']['immediate_checks']:
        print(f"□ {check}")
    print()
    
    print("📁 NAMING CONVENTION:")
    print("   lantern_fly_YYYYMMDD_HHMMSS_location.jpg")
    print("   Example: lantern_fly_20241209_143022_sidewalk_001.jpg")
    print()
    
    print("🚀 READY TO START COLLECTING!")
    print("=" * 60)

def create_markdown_guide(guide):
    """Create a markdown version of the guide"""
    md_content = f"""# {guide['title']}

**Version:** {guide['version']}  
**Created:** {guide['created']}  
**Target:** {guide['target']}

## 🎯 Overview

**Goal:** {guide['overview']['goal']}  
**Use Case:** {guide['overview']['use_case']}  
**Classification:** {guide['overview']['classification']}  
**Timeline:** {guide['overview']['timeline']}

## 📸 Photo Requirements

### Quality Standards
- **Resolution:** {guide['photo_requirements']['quality']['resolution']}
- **Format:** {guide['photo_requirements']['quality']['format']}
- **Focus:** {guide['photo_requirements']['quality']['focus']}
- **Lighting:** {guide['photo_requirements']['quality']['lighting']}
- **Composition:** {guide['photo_requirements']['quality']['composition']}

### Content Requirements
- **Subject:** {guide['photo_requirements']['content']['subject']}
- **Condition:** {guide['photo_requirements']['content']['condition']}
- **Background:** {guide['photo_requirements']['content']['background']}
- **Angle:** {guide['photo_requirements']['content']['angle']}
- **Size:** {guide['photo_requirements']['content']['size']}

### What to Avoid
{chr(10).join([f"- {item}" for item in guide['photo_requirements']['exclusions']['avoid']])}

## 📍 Collection Strategy

### High-Priority Locations
{chr(10).join([f"- {location}" for location in guide['collection_strategy']['locations']['high_priority']])}

### Best Times
{chr(10).join([f"- {time}" for time in guide['collection_strategy']['timing']['best_times']])}

## 📁 Data Organization

### Naming Convention
`{guide['data_organization']['naming_convention']}`

### Examples
{chr(10).join([f"- `{example}`" for example in guide['data_organization']['examples']])}

## 📊 Progress Tracking

### Daily Goals
{chr(10).join([f"- **{day.replace('_', ' ').title()}:** {goal}" for day, goal in guide['progress_tracking']['daily_goals'].items()])}

### Milestones
{chr(10).join([f"- {milestone}" for milestone in guide['progress_tracking']['milestones']])}

## 🔍 Quality Control

### Immediate Checks
{chr(10).join([f"- {check}" for check in guide['quality_control']['immediate_checks']])}

### Final Validation
{chr(10).join([f"- {check}" for check in guide['quality_control']['final_validation']])}

## 🛡️ Safety Considerations

### General Safety
{chr(10).join([f"- {item}" for item in guide['safety_considerations']['general']])}

### Campus-Specific
{chr(10).join([f"- {item}" for item in guide['safety_considerations']['campus_specific']])}

## 📦 Equipment Checklist

### Essential
{chr(10).join([f"- {item}" for item in guide['equipment_checklist']['essential']])}

### Recommended
{chr(10).join([f"- {item}" for item in guide['equipment_checklist']['recommended']])}

## 🚀 Ready to Start!

Follow this guide to collect 500 high-quality photos of squashed lantern flies for your machine learning model. Good luck with your data collection!
"""
    
    return md_content

def main():
    """Generate and save the photo collection guide"""
    print("Generating Photo Collection Guide...")
    
    # Generate guide
    guide = generate_photo_collection_guide()
    
    # Save to JSON
    save_guide_to_file(guide, "photo_collection_guide.json")
    
    # Save to Markdown
    md_content = create_markdown_guide(guide)
    with open("photo_collection_guide.md", 'w', encoding='utf-8') as f:
        f.write(md_content)
    print("Photo collection guide saved to: photo_collection_guide.md")
    
    # Print summary
    print_guide_summary(guide)
    
    print("\n📚 Files created:")
    print("  - photo_collection_guide.json (detailed guide)")
    print("  - photo_collection_guide.md (markdown version)")
    print("\n🎯 You're ready to start collecting photos!")

if __name__ == "__main__":
    main()
