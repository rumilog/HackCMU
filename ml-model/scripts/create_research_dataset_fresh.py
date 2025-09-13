#!/usr/bin/env python3
"""
Create a fresh research dataset for human-verified lanternfly sightings
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import json

def load_credentials():
    """Load Hugging Face token from credentials file"""
    credentials_path = Path(__file__).parent.parent / '.credentials'
    try:
        with open(credentials_path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('HUGGINGFACE_TOKEN='):
                    return line.split('=', 1)[1].strip()
    except FileNotFoundError:
        print("❌ Credentials file not found")
        return None
    return None

def create_research_dataset():
    """Create the research dataset on Hugging Face"""
    
    # Load credentials
    hf_token = load_credentials()
    if not hf_token:
        print("❌ No Hugging Face token found")
        return False
    
    print(f"✅ Loaded HF token: {hf_token[:10]}...")
    
    # Initialize API
    api = HfApi(token=hf_token)
    
    # Dataset configuration
    repo_name = "rlogh/lanternfly_research_dataset"
    
    try:
        print(f"🚀 Creating research dataset: {repo_name}")
        
        # Create the dataset repository
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            repo_type="dataset",
            private=False,  # Make it public for researchers
            exist_ok=True
        )
        
        print("✅ Dataset repository created successfully")
        
        # Create comprehensive README
        readme_content = """---
license: cc-by-4.0
task_categories:
- image-classification
- object-detection
tags:
- biology
- ecology
- invasive-species
- spotted-lanternfly
- geolocation
- citizen-science
- conservation
size_categories:
- n<1K
---

# Lantern Fly Research Dataset

This dataset contains **human-verified spotted lanternfly sightings** collected through the Lantern Fly Tracker app. Each entry includes high-quality photos, precise geolocation data, and comprehensive metadata for ecological research.

## 🎯 Purpose

This dataset supports:
- **Ecological research** on spotted lanternfly distribution and spread patterns
- **Machine learning model training** with verified, high-quality data
- **Temporal and spatial analysis** of invasive species behavior
- **Conservation efforts** and pest management strategies
- **Citizen science** validation and research

## 📊 Dataset Structure

```
lanternfly_research_dataset/
├── images/           # High-quality lanternfly photos
├── metadata/         # JSON files with location and metadata
└── README.md         # This file
```

## 🔍 Data Quality

All photos in this dataset have been:
1. **AI-classified** as potential spotted lanternflies
2. **Human-verified** by app users for accuracy
3. **Geolocated** with GPS coordinates
4. **Timestamped** for temporal analysis
5. **Quality-checked** for research standards

## 📋 Data Fields

Each sighting includes:

### Image Data
- **High-resolution photo** of verified spotted lanternfly
- **Unique filename** with timestamp and username

### Location Data
- **GPS coordinates** (latitude/longitude)
- **Location name** (human-readable description)
- **Geographic precision** for spatial analysis

### Metadata
- **Username** of person who verified the sighting
- **Sighting date/time** (when photo was taken)
- **Upload date** (when added to dataset)
- **AI confidence score** (0-1 scale)
- **Verification status** (always "human_verified")
- **Image file size** and technical details

## 🚀 Usage

### For Researchers
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("rlogh/lanternfly_research_dataset")

# Access images and metadata
images = dataset["images"]
metadata = dataset["metadata"]
```

### For Conservation
- Track spread patterns over time
- Identify high-risk areas
- Monitor population dynamics
- Support eradication efforts

## 📈 Current Statistics
- **Total sightings**: Growing daily
- **Geographic coverage**: Multiple states
- **Time range**: 2024-present
- **Data quality**: Human-verified

## 🤝 Contributing

This dataset is automatically updated when users verify lanternfly sightings through the Lantern Fly Tracker app. Each verified sighting contributes valuable data to ecological research.

## 📜 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{lanternfly_research_2024,
  title={Lantern Fly Research Dataset: Human-verified spotted lanternfly sightings with geolocation data},
  author={Lantern Fly Tracker Community},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/rlogh/lanternfly_research_dataset}
}
```

## 📄 License

This dataset is released under the **Creative Commons Attribution 4.0 International License** (CC BY 4.0).

## 🔗 Links

- **Dataset**: https://huggingface.co/datasets/rlogh/lanternfly_research_dataset
- **App**: Lantern Fly Tracker (citizen science app)
- **Research**: Contact for collaboration opportunities

---

*This dataset is maintained by the Lantern Fly Tracker community and updated automatically with verified sightings.*
"""
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Add comprehensive dataset README"
        )
        
        print("✅ README.md uploaded")
        
        # Create initial directory structure with placeholders
        placeholder_content = """# Images Directory

This directory contains high-quality photos of human-verified spotted lanternflies.

## File Naming Convention
- Format: `lanternfly_{username}_{timestamp}.jpg`
- Example: `lanternfly_john_2024-01-15T10-30-45-123Z.jpg`

## Image Quality
- High resolution photos
- Clear visibility of spotted lanternfly
- Human-verified for accuracy
- GPS coordinates included in metadata

## Automatic Updates
New images are automatically added when users verify lanternfly sightings through the Lantern Fly Tracker app.
"""
        
        api.upload_file(
            path_or_fileobj=placeholder_content.encode(),
            path_in_repo="images/README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Create images directory structure"
        )
        
        metadata_placeholder = """# Metadata Directory

This directory contains JSON metadata files for each lanternfly sighting.

## File Naming Convention
- Format: `lanternfly_{username}_{timestamp}.json`
- Example: `lanternfly_john_2024-01-15T10-30-45-123Z.json`

## Metadata Fields
```json
{
  "username": "john",
  "sighting_date": "2024-01-15T10:30:45.123Z",
  "upload_date": "2024-01-15T10:35:22.456Z",
  "latitude": "40.7128",
  "longitude": "-74.0060",
  "location_name": "Central Park, New York",
  "confidence_score": "0.95",
  "verification_status": "human_verified",
  "dataset_version": "1.0.0",
  "data_source": "lantern_fly_tracker_app",
  "image_filename": "lanternfly_john_2024-01-15T10-30-45-123Z.jpg",
  "image_size_bytes": 2048576
}
```

## Automatic Updates
Metadata files are automatically created and uploaded when images are added to the dataset.
"""
        
        api.upload_file(
            path_or_fileobj=metadata_placeholder.encode(),
            path_in_repo="metadata/README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Create metadata directory structure"
        )
        
        print("✅ Directory structure created")
        
        # Create dataset card with schema
        dataset_card = {
            "dataset_name": "lanternfly_research_dataset",
            "version": "1.0.0",
            "description": "Human-verified spotted lanternfly sightings with geolocation data for ecological research",
            "created_by": "Lantern Fly Tracker App",
            "total_sightings": 0,
            "last_updated": "2024-01-01T00:00:00Z",
            "data_schema": {
                "image": {
                    "type": "file",
                    "description": "High-quality photo of verified spotted lanternfly",
                    "format": "JPEG"
                },
                "metadata": {
                    "type": "json",
                    "fields": {
                        "username": "string - Username of person who verified the sighting",
                        "sighting_date": "ISO datetime - When the photo was taken",
                        "upload_date": "ISO datetime - When uploaded to dataset",
                        "latitude": "string - GPS latitude coordinate",
                        "longitude": "string - GPS longitude coordinate",
                        "location_name": "string - Human-readable location description",
                        "confidence_score": "string - AI classification confidence (0-1)",
                        "verification_status": "string - Always 'human_verified'",
                        "dataset_version": "string - Dataset version",
                        "data_source": "string - Source application",
                        "image_filename": "string - Associated image filename",
                        "image_size_bytes": "number - Image file size in bytes"
                    }
                }
            },
            "usage_notes": [
                "All images are human-verified for accuracy",
                "GPS coordinates are provided for spatial analysis",
                "Timestamps enable temporal analysis of spread patterns",
                "Dataset is automatically updated with new verified sightings"
            ],
            "research_applications": [
                "Ecological distribution modeling",
                "Invasive species spread analysis",
                "Machine learning model training",
                "Conservation planning",
                "Citizen science validation"
            ]
        }
        
        api.upload_file(
            path_or_fileobj=json.dumps(dataset_card, indent=2).encode(),
            path_in_repo="dataset_card.json",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Add comprehensive dataset card with schema"
        )
        
        print("✅ Dataset card created")
        
        print(f"\n🎉 Research dataset created successfully!")
        print(f"📊 Dataset URL: https://huggingface.co/datasets/{repo_name}")
        print(f"📁 Repository: {repo_name}")
        print(f"🔗 Direct link: https://huggingface.co/datasets/{repo_name}")
        print(f"\n📝 Dataset is ready to receive human-verified lanternfly sightings!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create research dataset: {e}")
        return False

if __name__ == "__main__":
    success = create_research_dataset()
    if success:
        print("\n✅ Research dataset setup complete!")
        print("🚀 Ready to upload verified lanternfly sightings!")
    else:
        print("\n❌ Failed to create research dataset")
        sys.exit(1)
