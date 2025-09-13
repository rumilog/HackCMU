#!/usr/bin/env python3
"""
Script to create negative dataset using only real images from BIOSCAN and nature datasets.
Removes synthetic images and maintains 1:1 ratio with positive data.
"""

import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

from datasets import load_dataset, Dataset
from huggingface_hub import whoami
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealNegativeDataCollector:
    """Class to collect only real negative data from BIOSCAN and nature datasets."""
    
    def __init__(self, seed: int = 42):
        """Initialize the collector with a random seed."""
        random.seed(seed)
        np.random.seed(seed)
        
    def download_bioscan_images(self, target_size: int) -> List[Dict]:
        """Download images from BIOSCAN-30k dataset."""
        logger.info(f"Downloading {target_size} images from BIOSCAN-30k...")
        negative_images = []
        
        try:
            bioscan_dataset = load_dataset("Voxel51/BIOSCAN-30k", split="train", streaming=True)
            
            for i, item in enumerate(tqdm(bioscan_dataset, desc="BIOSCAN-30k", total=target_size)):
                if i >= target_size:
                    break
                    
                try:
                    # Try different possible image field names
                    image = None
                    for field in ["image", "img", "photo", "picture"]:
                        if field in item and item[field] is not None:
                            image = item[field]
                            break
                    
                    if image is None:
                        continue
                        
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Resize to 512x512 to match our lanternfly images
                    image = image.resize((512, 512), Image.LANCZOS)
                    
                    negative_images.append({
                        "image": image,
                        "filename": f"bioscan_{i:04d}.jpg",
                        "source": "BIOSCAN-30k",
                        "category": "insect"
                    })
                except Exception as e:
                    logger.warning(f"Error processing BIOSCAN-30k image {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load BIOSCAN-30k: {e}")
        
        logger.info(f"Downloaded {len(negative_images)} BIOSCAN images")
        return negative_images
    
    def download_nature_images(self, target_size: int) -> List[Dict]:
        """Download images from nature-dataset."""
        logger.info(f"Downloading {target_size} images from nature-dataset...")
        negative_images = []
        
        try:
            nature_dataset = load_dataset("mertcobanov/nature-dataset", split="train", streaming=True)
            
            for i, item in enumerate(tqdm(nature_dataset, desc="nature-dataset", total=target_size)):
                if i >= target_size:
                    break
                    
                try:
                    # Try different possible image field names
                    image = None
                    for field in ["image", "img", "photo", "picture"]:
                        if field in item and item[field] is not None:
                            image = item[field]
                            break
                    
                    if image is None:
                        continue
                        
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Resize to 512x512 to match our lanternfly images
                    image = image.resize((512, 512), Image.LANCZOS)
                    
                    negative_images.append({
                        "image": image,
                        "filename": f"nature_{i:04d}.jpg",
                        "source": "nature-dataset",
                        "category": "nature"
                    })
                except Exception as e:
                    logger.warning(f"Error processing nature-dataset image {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load nature-dataset: {e}")
        
        logger.info(f"Downloaded {len(negative_images)} nature images")
        return negative_images
    
    def collect_real_negatives(self, total_target_size: int = 1826) -> List[Dict]:
        """Collect real negative images from BIOSCAN and nature datasets."""
        logger.info(f"Collecting {total_target_size} real negative images...")
        
        # Split evenly between BIOSCAN and nature datasets
        bioscan_size = total_target_size // 2
        nature_size = total_target_size - bioscan_size
        
        logger.info(f"Target distribution: {bioscan_size} BIOSCAN, {nature_size} nature")
        
        all_negatives = []
        
        # Collect from BIOSCAN
        bioscan_images = self.download_bioscan_images(bioscan_size)
        all_negatives.extend(bioscan_images)
        
        # Collect from nature dataset
        nature_images = self.download_nature_images(nature_size)
        all_negatives.extend(nature_images)
        
        # Shuffle the combined dataset
        random.shuffle(all_negatives)
        
        logger.info(f"Collected {len(all_negatives)} total real negative images")
        return all_negatives

def save_real_negative_images_locally(negative_images: List[Dict], output_dir: str = "real_negative_lanternflies"):
    """Save real negative images locally for verification."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Saving {len(negative_images)} real negative images to {output_path}")
    
    for i, item in enumerate(tqdm(negative_images, desc="Saving real negative images")):
        try:
            image = item["image"]
            filename = item["filename"]
            image.save(output_path / filename, "JPEG", quality=95)
        except Exception as e:
            logger.error(f"Error saving real negative image {i}: {e}")
    
    logger.info(f"Real negative images saved to {output_path}")

def create_real_negatives_dataset(negative_images: List[Dict]) -> Dataset:
    """Create a Hugging Face dataset from real negative images."""
    logger.info("Creating real negatives dataset...")
    
    # Match the features of the original dataset
    dataset_dict = {
        "image": [item["image"] for item in negative_images],
        "filename": [item["filename"] for item in negative_images]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Created real negatives dataset with {len(dataset)} images")
    
    return dataset

def upload_real_negatives_dataset(dataset: Dataset, repo_name: str = "rlogh/lanternfly_swatter_training"):
    """Upload the real negatives dataset to Hugging Face."""
    logger.info(f"Uploading real negatives dataset to {repo_name}...")
    
    try:
        # Check authentication
        user_info = whoami()
        logger.info(f"Logged in as: {user_info['name']}")
        
        # Upload with "negatives" split (this will replace the existing one)
        dataset.push_to_hub(
            repo_name,
            private=False,
            split="negatives"
        )
        
        logger.info("Real negatives dataset uploaded successfully!")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

def get_positive_dataset_size(repo_name: str = "rlogh/lanternfly_swatter_training") -> int:
    """Get the total size of positive data (original + augmented)."""
    try:
        original_dataset = load_dataset(repo_name, split="original")
        augmented_dataset = load_dataset(repo_name, split="augmented")
        total_positive = len(original_dataset) + len(augmented_dataset)
        logger.info(f"Total positive dataset size: {total_positive} images")
        return total_positive
    except Exception as e:
        logger.error(f"Failed to get positive dataset size: {e}")
        return 1826

def main():
    """Main function to orchestrate the real negative data collection pipeline."""
    parser = argparse.ArgumentParser(description="Create real negative dataset for lanternfly classification")
    parser.add_argument("--repo-name", type=str, default="rlogh/lanternfly_swatter_training",
                       help="Target Hugging Face repository name")
    parser.add_argument("--target-size", type=int, default=None,
                       help="Target size for negative dataset (default: match positive dataset)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling")
    parser.add_argument("--test-only", action="store_true",
                       help="Only process a small number of images for testing")
    
    args = parser.parse_args()
    
    if args.test_only:
        args.target_size = 100
        logger.info("Running in test mode - using 100 images")
    elif args.target_size is None:
        args.target_size = get_positive_dataset_size(args.repo_name)
    
    try:
        # Initialize collector
        collector = RealNegativeDataCollector(seed=args.seed)
        
        # Collect real negative images
        negative_images = collector.collect_real_negatives(args.target_size)
        
        if not negative_images:
            logger.error("No real negative images were collected")
            return
        
        # Save images locally first for verification
        save_real_negative_images_locally(negative_images, "real_negative_lanternflies")
        
        # Create real negatives dataset
        negatives_dataset = create_real_negatives_dataset(negative_images)
        
        # Upload to Hugging Face
        upload_real_negatives_dataset(negatives_dataset, args.repo_name)
        
        logger.info("Real negative data collection pipeline completed successfully!")
        logger.info(f"Collected {len(negative_images)} real negative images")
        
        # Show distribution by category
        categories = {}
        for item in negative_images:
            category = item["category"]
            categories[category] = categories.get(category, 0) + 1
        
        logger.info("Distribution by category:")
        for category, count in categories.items():
            logger.info(f"  {category}: {count} images")
        
    except Exception as e:
        logger.error(f"Real negative data collection pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
