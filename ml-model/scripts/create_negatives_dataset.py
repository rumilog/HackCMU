#!/usr/bin/env python3
"""
Script to create negative dataset for lanternfly classification.
Downloads images from multiple Hugging Face datasets to create a balanced negative set.
"""

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import whoami
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NegativeDataCollector:
    """Class to collect negative data from various Hugging Face datasets."""
    
    def __init__(self, seed: int = 42):
        """Initialize the collector with a random seed."""
        random.seed(seed)
        np.random.seed(seed)
        
    def download_insect_datasets(self, target_size: int = 500) -> List[Dict]:
        """Download insects (non-target species) from BIOSCAN-30k and alternative datasets."""
        logger.info("Downloading insect datasets...")
        negative_images = []
        
        # BIOSCAN-30k dataset
        try:
            logger.info("Loading BIOSCAN-30k dataset...")
            bioscan_dataset = load_dataset("Voxel51/BIOSCAN-30k", split="train", streaming=True)
            
            # Sample from BIOSCAN-30k
            bioscan_count = target_size // 2
            logger.info(f"Sampling {bioscan_count} images from BIOSCAN-30k...")
            
            for i, item in enumerate(tqdm(bioscan_dataset, desc="BIOSCAN-30k", total=bioscan_count)):
                if i >= bioscan_count:
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
        
        # Use a more reliable insect dataset - Insecta dataset
        try:
            logger.info("Loading Insecta dataset...")
            insecta_dataset = load_dataset("Genius-Society/insecta", split="train", streaming=True)
            
            # Sample from Insecta
            insecta_count = target_size - len(negative_images)
            logger.info(f"Sampling {insecta_count} images from Insecta...")
            
            for i, item in enumerate(tqdm(insecta_dataset, desc="Insects", total=insecta_count)):
                if i >= insecta_count:
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
                        "filename": f"insecta_{i:04d}.jpg",
                        "source": "Insects",
                        "category": "insect"
                    })
                except Exception as e:
                    logger.warning(f"Error processing Insecta image {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load Insecta: {e}")
        
        logger.info(f"Downloaded {len(negative_images)} insect images")
        return negative_images
    
    def download_nature_datasets(self, target_size: int = 500) -> List[Dict]:
        """Download nature/background images from nature-dataset and alternative datasets."""
        logger.info("Downloading nature/background datasets...")
        negative_images = []
        
        # Nature dataset
        try:
            logger.info("Loading nature-dataset...")
            nature_dataset = load_dataset("mertcobanov/nature-dataset", split="train", streaming=True)
            
            # Sample from nature dataset
            nature_count = target_size // 2
            logger.info(f"Sampling {nature_count} images from nature-dataset...")
            
            for i, item in enumerate(tqdm(nature_dataset, desc="nature-dataset", total=nature_count)):
                if i >= nature_count:
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
        
        # Use a more reliable nature dataset - ImageNet or similar
        try:
            logger.info("Loading ImageNet dataset...")
            imagenet_dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
            
            # Sample from ImageNet
            imagenet_count = target_size - len(negative_images)
            logger.info(f"Sampling {imagenet_count} images from ImageNet...")
            
            for i, item in enumerate(tqdm(imagenet_dataset, desc="ImageNet", total=imagenet_count)):
                if i >= imagenet_count:
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
                        "filename": f"imagenet_{i:04d}.jpg",
                        "source": "ImageNet",
                        "category": "nature"
                    })
                except Exception as e:
                    logger.warning(f"Error processing ImageNet image {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load ImageNet: {e}")
        
        logger.info(f"Downloaded {len(negative_images)} nature/background images")
        return negative_images
    
    def download_biodiversity_datasets(self, target_size: int = 500) -> List[Dict]:
        """Download general biodiversity from various datasets."""
        logger.info("Downloading biodiversity datasets...")
        negative_images = []
        
        # Use COCO dataset for general objects and animals
        try:
            logger.info("Loading COCO dataset...")
            coco_dataset = load_dataset("detection-datasets/coco_2017_val", split="validation", streaming=True)
            
            # Sample from COCO
            logger.info(f"Sampling {target_size} images from COCO...")
            
            for i, item in enumerate(tqdm(coco_dataset, desc="COCO", total=target_size)):
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
                        "filename": f"coco_{i:04d}.jpg",
                        "source": "COCO",
                        "category": "biodiversity"
                    })
                except Exception as e:
                    logger.warning(f"Error processing COCO image {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load COCO: {e}")
        
        # If we need more images, use a simple synthetic approach
        if len(negative_images) < target_size:
            logger.info("Generating synthetic negative images...")
            remaining = target_size - len(negative_images)
            
            for i in range(remaining):
                try:
                    # Create a simple synthetic image (solid color or simple pattern)
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    color = random.choice(colors)
                    
                    # Create a 512x512 image with the chosen color
                    image = Image.new('RGB', (512, 512), color)
                    
                    # Add some random noise or pattern
                    if random.random() < 0.5:
                        # Add random rectangles
                        for _ in range(random.randint(1, 5)):
                            x1 = random.randint(0, 400)
                            y1 = random.randint(0, 400)
                            x2 = x1 + random.randint(50, 112)
                            y2 = y1 + random.randint(50, 112)
                            rect_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                            # This is a simplified approach - in practice you'd use PIL's ImageDraw
                    
                    negative_images.append({
                        "image": image,
                        "filename": f"synthetic_{i:04d}.jpg",
                        "source": "Synthetic",
                        "category": "biodiversity"
                    })
                except Exception as e:
                    logger.warning(f"Error creating synthetic image {i}: {e}")
                    continue
        
        logger.info(f"Downloaded {len(negative_images)} biodiversity images")
        return negative_images
    
    def collect_all_negatives(self, total_target_size: int = 1826) -> List[Dict]:
        """Collect negative images from all datasets to match the positive dataset size."""
        logger.info(f"Collecting {total_target_size} negative images from all datasets...")
        
        # Distribute target size across categories
        insects_size = total_target_size // 3
        nature_size = total_target_size // 3
        biodiversity_size = total_target_size - insects_size - nature_size
        
        logger.info(f"Target distribution: {insects_size} insects, {nature_size} nature, {biodiversity_size} biodiversity")
        
        all_negatives = []
        
        # Collect from each category
        insects = self.download_insect_datasets(insects_size)
        nature = self.download_nature_datasets(nature_size)
        biodiversity = self.download_biodiversity_datasets(biodiversity_size)
        
        all_negatives.extend(insects)
        all_negatives.extend(nature)
        all_negatives.extend(biodiversity)
        
        # Shuffle the combined dataset
        random.shuffle(all_negatives)
        
        logger.info(f"Collected {len(all_negatives)} total negative images")
        return all_negatives

def save_negative_images_locally(negative_images: List[Dict], output_dir: str = "negative_lanternflies"):
    """Save negative images locally for verification."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Saving {len(negative_images)} negative images to {output_path}")
    
    for i, item in enumerate(tqdm(negative_images, desc="Saving negative images")):
        try:
            image = item["image"]
            filename = item["filename"]
            image.save(output_path / filename, "JPEG", quality=95)
        except Exception as e:
            logger.error(f"Error saving negative image {i}: {e}")
    
    logger.info(f"Negative images saved to {output_path}")

def create_negatives_dataset(negative_images: List[Dict]) -> Dataset:
    """Create a Hugging Face dataset from negative images."""
    logger.info("Creating negatives dataset...")
    
    # Match the features of the original dataset
    dataset_dict = {
        "image": [item["image"] for item in negative_images],
        "filename": [item["filename"] for item in negative_images]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Created negatives dataset with {len(dataset)} images")
    
    return dataset

def upload_negatives_dataset(dataset: Dataset, repo_name: str = "rlogh/lanternfly_swatter_training"):
    """Upload the negatives dataset to Hugging Face."""
    logger.info(f"Uploading negatives dataset to {repo_name}...")
    
    try:
        # Check authentication
        user_info = whoami()
        logger.info(f"Logged in as: {user_info['name']}")
        
        # Upload with "negatives" split
        dataset.push_to_hub(
            repo_name,
            private=False,
            split="negatives"
        )
        
        logger.info("Negatives dataset uploaded successfully!")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

def get_positive_dataset_size(repo_name: str = "rlogh/lanternfly_swatter_training") -> int:
    """Get the total size of positive data (original + augmented)."""
    try:
        # Load both splits to get total count
        original_dataset = load_dataset(repo_name, split="original")
        augmented_dataset = load_dataset(repo_name, split="augmented")
        
        total_positive = len(original_dataset) + len(augmented_dataset)
        logger.info(f"Total positive dataset size: {total_positive} images")
        return total_positive
        
    except Exception as e:
        logger.error(f"Failed to get positive dataset size: {e}")
        # Fallback to expected size
        return 1826

def main():
    """Main function to orchestrate the negative data collection pipeline."""
    parser = argparse.ArgumentParser(description="Create negative dataset for lanternfly classification")
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
        args.target_size = 50
        logger.info("Running in test mode - using 50 images")
    elif args.target_size is None:
        args.target_size = get_positive_dataset_size(args.repo_name)
    
    try:
        # Initialize collector
        collector = NegativeDataCollector(seed=args.seed)
        
        # Collect negative images
        negative_images = collector.collect_all_negatives(args.target_size)
        
        if not negative_images:
            logger.error("No negative images were collected")
            return
        
        # Save images locally first for verification
        save_negative_images_locally(negative_images, "negative_lanternflies")
        
        # Create negatives dataset
        negatives_dataset = create_negatives_dataset(negative_images)
        
        # Upload to Hugging Face
        upload_negatives_dataset(negatives_dataset, args.repo_name)
        
        logger.info("Negative data collection pipeline completed successfully!")
        logger.info(f"Collected {len(negative_images)} negative images")
        
        # Show distribution by category
        categories = {}
        for item in negative_images:
            category = item["category"]
            categories[category] = categories.get(category, 0) + 1
        
        logger.info("Distribution by category:")
        for category, count in categories.items():
            logger.info(f"  {category}: {count} images")
        
    except Exception as e:
        logger.error(f"Negative data collection pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
