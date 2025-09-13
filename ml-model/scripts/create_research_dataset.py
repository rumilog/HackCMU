#!/usr/bin/env python3
"""
Create a new Hugging Face dataset for human-verified lanternfly sightings
This dataset will contain photos, geolocation data, and metadata for researchers
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import json

# Add the parent directory to the path to import credentials
sys.path.append(str(Path(__file__).parent.parent))

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
    
    # Initialize API
    api = HfApi(token=hf_token)
    
    # Dataset configuration
    repo_name = "rlogh/lanternfly_research_dataset"
    dataset_description = """
# Lantern Fly Research Dataset

This dataset contains human-verified spotted lanternfly sightings collected through the Lantern Fly Tracker app. Each entry includes:

- **High-quality photos** of verified spotted lanternflies
- **Precise geolocation data** (latitude/longitude coordinates)
- **Timestamp information** for temporal analysis
- **Confidence scores** from AI classification
- **Verification status** (human-verified)

## Dataset Structure

- `images/` - Contains the actual lanternfly photos
- `metadata/` - Contains JSON files with location, timestamp, and other metadata

## Use Cases

This dataset is valuable for:
- **Ecological research** on spotted lanternfly distribution
- **Machine learning model training** with verified data
- **Temporal and spatial analysis** of lanternfly spread
- **Conservation efforts** and pest management

## Data Quality

All photos in this dataset have been:
1. **AI-classified** as potential lanternflies
2. **Human-verified** by app users
3. **Geolocated** with GPS coordinates
4. **Timestamped** for temporal analysis

## Citation

If you use this dataset in your research, please cite:
```
Lantern Fly Tracker Research Dataset. (2024). Human-verified spotted lanternfly sightings with geolocation data. Hugging Face Datasets.
```

## License

This dataset is released under the Creative Commons Attribution 4.0 International License.
"""
    
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
        
        # Create README.md
        readme_content = f"""---
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
size_categories:
- n<1K
---

{dataset_description}
"""
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Add dataset README with description and metadata"
        )
        
        print("✅ README.md uploaded")
        
        # Create initial directory structure
        initial_structure = {
            "images": "Directory containing lanternfly photos",
            "metadata": "Directory containing JSON metadata files"
        }
        
        # Create a placeholder file to establish directory structure
        placeholder_content = "# This directory will contain lanternfly photos\n# Uploaded automatically when users verify lanternfly sightings"
        
        api.upload_file(
            path_or_fileobj=placeholder_content.encode(),
            path_in_repo="images/README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Create images directory structure"
        )
        
        api.upload_file(
            path_or_fileobj=placeholder_content.encode(),
            path_in_repo="metadata/README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Create metadata directory structure"
        )
        
        print("✅ Directory structure created")
        
        # Create dataset card
        dataset_card = {
            "dataset_name": "lanternfly_research_dataset",
            "version": "1.0.0",
            "description": "Human-verified spotted lanternfly sightings with geolocation data",
            "created_by": "Lantern Fly Tracker App",
            "total_sightings": 0,
            "last_updated": "2024-01-01T00:00:00Z",
            "data_fields": {
                "image": "High-quality photo of verified lanternfly",
                "latitude": "GPS latitude coordinate",
                "longitude": "GPS longitude coordinate", 
                "location_name": "Human-readable location description",
                "username": "Username of person who verified the sighting",
                "confidence_score": "AI classification confidence (0-1)",
                "sighting_date": "Date and time of the sighting",
                "verification_status": "Always 'human_verified' for this dataset",
                "upload_date": "Date when uploaded to dataset"
            }
        }
        
        api.upload_file(
            path_or_fileobj=json.dumps(dataset_card, indent=2).encode(),
            path_in_repo="dataset_card.json",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Add dataset card with metadata schema"
        )
        
        print("✅ Dataset card created")
        
        print(f"\n🎉 Research dataset created successfully!")
        print(f"📊 Dataset URL: https://huggingface.co/datasets/{repo_name}")
        print(f"📁 Repository: {repo_name}")
        print(f"🔗 Direct link: https://huggingface.co/datasets/{repo_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create research dataset: {e}")
        return False

if __name__ == "__main__":
    success = create_research_dataset()
    if success:
        print("\n✅ Research dataset setup complete!")
        print("📝 The dataset is now ready to receive human-verified lanternfly sightings")
    else:
        print("\n❌ Failed to create research dataset")
        sys.exit(1)
