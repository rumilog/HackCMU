#!/usr/bin/env python3
"""
Script to create augmented lanternfly images for training.
Generates 10x the original dataset size using various augmentation techniques.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
from typing import List, Tuple
import logging
from tqdm import tqdm

from datasets import load_dataset, Dataset
from huggingface_hub import whoami
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageAugmenter:
    """Class to handle various image augmentation techniques."""
    
    def __init__(self, seed: int = 42):
        """Initialize the augmenter with a random seed."""
        random.seed(seed)
        np.random.seed(seed)
        
    def rotate_image(self, image: Image.Image, angle_range: Tuple[int, int] = (-15, 15)) -> Image.Image:
        """Rotate image by a random angle within the specified range."""
        angle = random.uniform(angle_range[0], angle_range[1])
        return image.rotate(angle, fillcolor=(255, 255, 255), expand=False)
    
    def flip_horizontal(self, image: Image.Image) -> Image.Image:
        """Flip image horizontally."""
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def flip_vertical(self, image: Image.Image) -> Image.Image:
        """Flip image vertically."""
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    
    def adjust_brightness(self, image: Image.Image, factor_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
        """Adjust image brightness."""
        factor = random.uniform(factor_range[0], factor_range[1])
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image: Image.Image, factor_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Adjust image contrast."""
        factor = random.uniform(factor_range[0], factor_range[1])
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def adjust_saturation(self, image: Image.Image, factor_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Adjust image saturation."""
        factor = random.uniform(factor_range[0], factor_range[1])
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def add_noise(self, image: Image.Image, noise_factor: float = 0.1) -> Image.Image:
        """Add random noise to the image."""
        img_array = np.array(image)
        noise = np.random.normal(0, noise_factor * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    def blur_image(self, image: Image.Image, blur_range: Tuple[int, int] = (1, 3)) -> Image.Image:
        """Apply Gaussian blur to the image."""
        blur_radius = random.uniform(blur_range[0], blur_range[1])
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    def crop_and_resize(self, image: Image.Image, crop_range: Tuple[float, float] = (0.85, 0.95)) -> Image.Image:
        """Randomly crop and resize back to original size."""
        width, height = image.size
        crop_factor = random.uniform(crop_range[0], crop_range[1])
        
        new_width = int(width * crop_factor)
        new_height = int(height * crop_factor)
        
        # Random crop position
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        
        cropped = image.crop((left, top, left + new_width, top + new_height))
        return cropped.resize((width, height), Image.LANCZOS)
    
    def color_jitter(self, image: Image.Image) -> Image.Image:
        """Apply random color jittering."""
        # Random hue shift
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hue_shift = random.randint(-10, 10)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 180
        hsv[:, :, 0] = hsv[:, :, 0].astype(np.uint8)
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(img_array)
    
    def elastic_transform(self, image: Image.Image, alpha: int = 1000, sigma: int = 50) -> Image.Image:
        """Apply elastic transformation."""
        img_array = np.array(image)
        shape = img_array.shape[:2]
        
        # Generate random displacement fields
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        # Apply displacement
        x += dx
        y += dy
        
        # Remap the image
        transformed = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(transformed)
    
    def apply_random_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply a random combination of augmentations."""
        augmented = image.copy()
        
        # List of augmentation functions and their probabilities
        augmentations = [
            (self.rotate_image, 0.3),
            (self.flip_horizontal, 0.2),
            (self.flip_vertical, 0.1),
            (self.adjust_brightness, 0.4),
            (self.adjust_contrast, 0.4),
            (self.adjust_saturation, 0.3),
            (self.add_noise, 0.2),
            (self.blur_image, 0.15),
            (self.crop_and_resize, 0.3),
            (self.color_jitter, 0.2),
            (self.elastic_transform, 0.1)
        ]
        
        # Apply 2-4 random augmentations
        num_augmentations = random.randint(2, 4)
        selected_augmentations = random.sample(augmentations, num_augmentations)
        
        for aug_func, prob in selected_augmentations:
            if random.random() < prob:
                try:
                    augmented = aug_func(augmented)
                except Exception as e:
                    logger.warning(f"Augmentation failed: {e}")
                    continue
        
        return augmented

def load_original_dataset(repo_name: str = "rlogh/lanternfly_swatter_training") -> Dataset:
    """Load the original dataset from Hugging Face."""
    logger.info(f"Loading original dataset from {repo_name}...")
    
    try:
        dataset = load_dataset(repo_name, split="original")
        logger.info(f"Loaded {len(dataset)} original images")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def generate_augmented_images(dataset: Dataset, target_multiplier: int = 10) -> List[dict]:
    """
    Generate augmented images to reach target_multiplier times the original size.
    
    Args:
        dataset: Original dataset
        target_multiplier: How many times to multiply the original dataset size
        
    Returns:
        List of augmented image data
    """
    original_size = len(dataset)
    target_size = original_size * target_multiplier
    augmenter = ImageAugmenter()
    
    logger.info(f"Generating {target_size} augmented images from {original_size} originals...")
    
    augmented_images = []
    
    # Calculate how many augmentations per original image
    augmentations_per_image = target_multiplier - 1  # -1 because we keep one original
    
    for i, item in enumerate(tqdm(dataset, desc="Generating augmentations")):
        try:
            # Keep the original image
            original_image = item["image"]
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            augmented_images.append({
                "image": original_image,
                "filename": f"augmented_{i:04d}_original.jpg",
                "source_index": i
            })
            
            # Generate augmentations for this image
            for aug_idx in range(augmentations_per_image):
                try:
                    augmented_image = augmenter.apply_random_augmentation(original_image)
                    
                    augmented_images.append({
                        "image": augmented_image,
                        "filename": f"augmented_{i:04d}_{aug_idx:02d}.jpg",
                        "source_index": i
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to augment image {i}, augmentation {aug_idx}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing original image {i}: {e}")
            continue
    
    logger.info(f"Generated {len(augmented_images)} total images")
    return augmented_images

def save_augmented_images_locally(augmented_images: List[dict], output_dir: str = "augmented_lanternflies"):
    """Save augmented images locally for verification."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Saving {len(augmented_images)} augmented images to {output_path}")
    
    for i, item in enumerate(tqdm(augmented_images, desc="Saving augmented images")):
        try:
            image = item["image"]
            filename = item["filename"]
            image.save(output_path / filename, "JPEG", quality=95)
        except Exception as e:
            logger.error(f"Error saving augmented image {i}: {e}")
    
    logger.info(f"Augmented images saved to {output_path}")

def create_augmented_dataset(augmented_images: List[dict]) -> Dataset:
    """Create a Hugging Face dataset from augmented images."""
    logger.info("Creating augmented dataset...")
    
    # Match the features of the original dataset (no source_index)
    dataset_dict = {
        "image": [item["image"] for item in augmented_images],
        "filename": [item["filename"] for item in augmented_images]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Created augmented dataset with {len(dataset)} images")
    
    return dataset

def upload_augmented_dataset(dataset: Dataset, repo_name: str = "rlogh/lanternfly_swatter_training"):
    """Upload the augmented dataset to Hugging Face."""
    logger.info(f"Uploading augmented dataset to {repo_name}...")
    
    try:
        # Check authentication
        user_info = whoami()
        logger.info(f"Logged in as: {user_info['name']}")
        
        # Upload with "augmented" split
        dataset.push_to_hub(
            repo_name,
            private=False,
            split="augmented"
        )
        
        logger.info("Augmented dataset uploaded successfully!")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

def main():
    """Main function to orchestrate the augmentation pipeline."""
    parser = argparse.ArgumentParser(description="Create augmented lanternfly dataset")
    parser.add_argument("--repo-name", type=str, default="rlogh/lanternfly_swatter_training",
                       help="Source Hugging Face repository name")
    parser.add_argument("--multiplier", type=int, default=10,
                       help="How many times to multiply the original dataset size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible augmentations")
    parser.add_argument("--test-only", action="store_true",
                       help="Only process a few images for testing")
    
    args = parser.parse_args()
    
    if args.test_only:
        args.multiplier = 3  # Smaller multiplier for testing
        logger.info("Running in test mode - using 3x multiplier")
    
    try:
        # Load original dataset
        original_dataset = load_original_dataset(args.repo_name)
        
        # Generate augmented images
        augmented_images = generate_augmented_images(original_dataset, args.multiplier)
        
        if not augmented_images:
            logger.error("No augmented images were generated")
            return
        
        # Save images locally first for verification
        save_augmented_images_locally(augmented_images, "augmented_lanternflies")
        
        # Create augmented dataset
        augmented_dataset = create_augmented_dataset(augmented_images)
        
        # Upload to Hugging Face
        upload_augmented_dataset(augmented_dataset, args.repo_name)
        
        logger.info("Augmentation pipeline completed successfully!")
        logger.info(f"Original: {len(original_dataset)} images")
        logger.info(f"Augmented: {len(augmented_images)} images")
        logger.info(f"Multiplier achieved: {len(augmented_images) / len(original_dataset):.1f}x")
        
    except Exception as e:
        logger.error(f"Augmentation pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
