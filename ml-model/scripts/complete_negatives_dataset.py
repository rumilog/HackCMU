#!/usr/bin/env python3
"""
Script to complete the negative dataset to reach the target size.
Generates additional synthetic images to match the positive dataset size.
"""

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

class SyntheticImageGenerator:
    """Class to generate synthetic negative images."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_solid_color_image(self) -> Image.Image:
        """Generate a solid color image."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0),
            (128, 0, 128), (0, 128, 128), (255, 192, 203), (255, 165, 0)
        ]
        color = random.choice(colors)
        return Image.new('RGB', (512, 512), color)
    
    def generate_gradient_image(self) -> Image.Image:
        """Generate a gradient image."""
        image = Image.new('RGB', (512, 512))
        draw = ImageDraw.Draw(image)
        
        # Random gradient direction
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        if direction == 'horizontal':
            for x in range(512):
                color_value = int(255 * x / 512)
                color = (color_value, random.randint(0, 255), random.randint(0, 255))
                draw.line([(x, 0), (x, 512)], fill=color)
        elif direction == 'vertical':
            for y in range(512):
                color_value = int(255 * y / 512)
                color = (random.randint(0, 255), color_value, random.randint(0, 255))
                draw.line([(0, y), (512, y)], fill=color)
        else:  # diagonal
            for i in range(512):
                color_value = int(255 * i / 512)
                color = (color_value, color_value, random.randint(0, 255))
                draw.line([(i, 0), (0, i)], fill=color)
        
        return image
    
    def generate_pattern_image(self) -> Image.Image:
        """Generate an image with geometric patterns."""
        image = Image.new('RGB', (512, 512), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Draw random shapes
        num_shapes = random.randint(5, 15)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'line'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            if shape_type == 'circle':
                x = random.randint(0, 400)
                y = random.randint(0, 400)
                radius = random.randint(20, 100)
                draw.ellipse([x, y, x + radius, y + radius], fill=color)
            elif shape_type == 'rectangle':
                x1 = random.randint(0, 400)
                y1 = random.randint(0, 400)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(50, 150)
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else:  # line
                x1 = random.randint(0, 512)
                y1 = random.randint(0, 512)
                x2 = random.randint(0, 512)
                y2 = random.randint(0, 512)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(2, 10))
        
        return image
    
    def generate_noise_image(self) -> Image.Image:
        """Generate a noise image."""
        # Create random noise
        noise = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(noise)
    
    def generate_texture_image(self) -> Image.Image:
        """Generate a texture-like image."""
        image = Image.new('RGB', (512, 512))
        draw = ImageDraw.Draw(image)
        
        # Create a texture pattern
        for i in range(0, 512, 20):
            for j in range(0, 512, 20):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw.rectangle([i, j, i + 20, j + 20], fill=color)
        
        return image
    
    def generate_synthetic_image(self) -> Image.Image:
        """Generate a random synthetic image."""
        generators = [
            self.generate_solid_color_image,
            self.generate_gradient_image,
            self.generate_pattern_image,
            self.generate_noise_image,
            self.generate_texture_image
        ]
        
        generator = random.choice(generators)
        return generator()

def get_current_negatives_size(repo_name: str = "rlogh/lanternfly_swatter_training") -> int:
    """Get the current size of the negatives dataset."""
    try:
        negatives_dataset = load_dataset(repo_name, split="negatives")
        return len(negatives_dataset)
    except Exception as e:
        logger.error(f"Failed to get negatives dataset size: {e}")
        return 0

def get_positive_dataset_size(repo_name: str = "rlogh/lanternfly_swatter_training") -> int:
    """Get the total size of positive data (original + augmented)."""
    try:
        original_dataset = load_dataset(repo_name, split="original")
        augmented_dataset = load_dataset(repo_name, split="augmented")
        return len(original_dataset) + len(augmented_dataset)
    except Exception as e:
        logger.error(f"Failed to get positive dataset size: {e}")
        return 1826

def generate_additional_negatives(target_size: int, current_size: int) -> List[Dict]:
    """Generate additional synthetic negative images."""
    needed = target_size - current_size
    logger.info(f"Generating {needed} additional synthetic negative images...")
    
    generator = SyntheticImageGenerator()
    additional_images = []
    
    for i in tqdm(range(needed), desc="Generating synthetic images"):
        try:
            image = generator.generate_synthetic_image()
            additional_images.append({
                "image": image,
                "filename": f"synthetic_additional_{i:04d}.jpg",
                "source": "Synthetic",
                "category": "synthetic"
            })
        except Exception as e:
            logger.warning(f"Error generating synthetic image {i}: {e}")
            continue
    
    logger.info(f"Generated {len(additional_images)} additional synthetic images")
    return additional_images

def load_existing_negatives(repo_name: str = "rlogh/lanternfly_swatter_training") -> List[Dict]:
    """Load existing negative images from the dataset."""
    try:
        negatives_dataset = load_dataset(repo_name, split="negatives")
        logger.info(f"Loaded {len(negatives_dataset)} existing negative images")
        
        existing_images = []
        for i, item in enumerate(negatives_dataset):
            existing_images.append({
                "image": item["image"],
                "filename": item["filename"],
                "source": "Existing",
                "category": "existing"
            })
        
        return existing_images
    except Exception as e:
        logger.error(f"Failed to load existing negatives: {e}")
        return []

def create_complete_negatives_dataset(existing_images: List[Dict], additional_images: List[Dict]) -> Dataset:
    """Create a complete negatives dataset."""
    logger.info("Creating complete negatives dataset...")
    
    all_images = existing_images + additional_images
    
    # Match the features of the original dataset
    dataset_dict = {
        "image": [item["image"] for item in all_images],
        "filename": [item["filename"] for item in all_images]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Created complete negatives dataset with {len(dataset)} images")
    
    return dataset

def upload_complete_negatives_dataset(dataset: Dataset, repo_name: str = "rlogh/lanternfly_swatter_training"):
    """Upload the complete negatives dataset to Hugging Face."""
    logger.info(f"Uploading complete negatives dataset to {repo_name}...")
    
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
        
        logger.info("Complete negatives dataset uploaded successfully!")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

def main():
    """Main function to complete the negatives dataset."""
    parser = argparse.ArgumentParser(description="Complete the negative dataset to target size")
    parser.add_argument("--repo-name", type=str, default="rlogh/lanternfly_swatter_training",
                       help="Target Hugging Face repository name")
    parser.add_argument("--target-size", type=int, default=None,
                       help="Target size for negative dataset (default: match positive dataset)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible generation")
    
    args = parser.parse_args()
    
    if args.target_size is None:
        args.target_size = get_positive_dataset_size(args.repo_name)
    
    try:
        # Get current negatives size
        current_size = get_current_negatives_size(args.repo_name)
        logger.info(f"Current negatives size: {current_size}")
        logger.info(f"Target negatives size: {args.target_size}")
        
        if current_size >= args.target_size:
            logger.info("Negatives dataset is already at or above target size!")
            return
        
        # Load existing negatives
        existing_images = load_existing_negatives(args.repo_name)
        
        # Generate additional negatives
        additional_images = generate_additional_negatives(args.target_size, current_size)
        
        # Create complete dataset
        complete_dataset = create_complete_negatives_dataset(existing_images, additional_images)
        
        # Upload complete dataset
        upload_complete_negatives_dataset(complete_dataset, args.repo_name)
        
        logger.info("Negatives dataset completion pipeline completed successfully!")
        logger.info(f"Final negatives size: {len(complete_dataset)} images")
        
    except Exception as e:
        logger.error(f"Negatives dataset completion pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
