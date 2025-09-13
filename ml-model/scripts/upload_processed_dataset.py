#!/usr/bin/env python3
"""
Script to upload processed lanternfly images to Hugging Face dataset.
Run this after authenticating with Hugging Face CLI.
"""

import os
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm

from datasets import Dataset
from huggingface_hub import whoami

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_images(images_dir: str = "processed_lanternflies"):
    """
    Load all processed images from the local directory.
    
    Args:
        images_dir: Directory containing processed images
        
    Returns:
        List of image data dictionaries
    """
    images_path = Path(images_dir)
    
    if not images_path.exists():
        raise FileNotFoundError(f"Directory {images_dir} not found")
    
    image_files = sorted(list(images_path.glob("*.jpg")))
    logger.info(f"Found {len(image_files)} processed images")
    
    images_data = []
    for img_path in tqdm(image_files, desc="Loading images"):
        try:
            image = Image.open(img_path)
            images_data.append({
                "image": image,
                "filename": img_path.name
            })
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
    
    return images_data

def create_dataset(images_data):
    """Create Hugging Face dataset from processed images."""
    logger.info("Creating dataset...")
    
    dataset_dict = {
        "image": [item["image"] for item in images_data],
        "filename": [item["filename"] for item in images_data]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Created dataset with {len(dataset)} images")
    
    return dataset

def upload_dataset(dataset, repo_name: str = "rlogh/lanternfly_swatter_training"):
    """Upload dataset to Hugging Face."""
    logger.info(f"Uploading dataset to {repo_name}...")
    
    try:
        # Check authentication
        user_info = whoami()
        logger.info(f"Logged in as: {user_info['name']}")
        
        # Upload with "original" split
        dataset.push_to_hub(
            repo_name,
            private=False,
            split="original"
        )
        
        logger.info("Dataset uploaded successfully!")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

def main():
    """Main function."""
    try:
        # Load processed images
        images_data = load_processed_images()
        
        if not images_data:
            logger.error("No images found to upload")
            return
        
        # Create dataset
        dataset = create_dataset(images_data)
        
        # Upload to Hugging Face
        upload_dataset(dataset)
        
        logger.info("Upload completed successfully!")
        
    except Exception as e:
        logger.error(f"Upload process failed: {e}")
        raise

if __name__ == "__main__":
    main()
