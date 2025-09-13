#!/usr/bin/env python3
"""
Script to create dramatically augmented lanternfly images for training.
Generates 10x the original dataset size using highly visible augmentation techniques.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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

class DramaticImageAugmenter:
    """Class to handle dramatic and highly visible image augmentation techniques."""
    
    def __init__(self, seed: int = 42):
        """Initialize the augmenter with a random seed."""
        random.seed(seed)
        np.random.seed(seed)
        
    def dramatic_rotate(self, image: Image.Image) -> Image.Image:
        """Rotate image by dramatic angles."""
        angles = [90, 180, 270, -45, -30, -15, 15, 30, 45]
        angle = random.choice(angles)
        return image.rotate(angle, fillcolor=(255, 255, 255), expand=False)
    
    def dramatic_flip(self, image: Image.Image) -> Image.Image:
        """Apply dramatic flips."""
        flip_type = random.choice(['horizontal', 'vertical', 'both'])
        if flip_type == 'horizontal':
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip_type == 'vertical':
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        else:  # both
            return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    
    def dramatic_brightness(self, image: Image.Image) -> Image.Image:
        """Apply dramatic brightness changes."""
        factors = [0.3, 0.5, 0.7, 1.5, 2.0, 2.5]
        factor = random.choice(factors)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def dramatic_contrast(self, image: Image.Image) -> Image.Image:
        """Apply dramatic contrast changes."""
        factors = [0.3, 0.5, 0.7, 1.5, 2.0, 2.5]
        factor = random.choice(factors)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def dramatic_saturation(self, image: Image.Image) -> Image.Image:
        """Apply dramatic saturation changes."""
        factors = [0.0, 0.3, 0.5, 1.5, 2.0, 3.0]
        factor = random.choice(factors)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def dramatic_noise(self, image: Image.Image) -> Image.Image:
        """Add dramatic noise to the image."""
        img_array = np.array(image)
        noise_factors = [0.2, 0.3, 0.4, 0.5]
        noise_factor = random.choice(noise_factors)
        noise = np.random.normal(0, noise_factor * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    def dramatic_blur(self, image: Image.Image) -> Image.Image:
        """Apply dramatic blur effects."""
        blur_radii = [3, 5, 7, 10, 15]
        blur_radius = random.choice(blur_radii)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    def dramatic_crop(self, image: Image.Image) -> Image.Image:
        """Apply dramatic cropping and resizing."""
        width, height = image.size
        crop_factors = [0.6, 0.7, 0.8, 0.9]
        crop_factor = random.choice(crop_factors)
        
        new_width = int(width * crop_factor)
        new_height = int(height * crop_factor)
        
        # Random crop position
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        
        cropped = image.crop((left, top, left + new_width, top + new_height))
        return cropped.resize((width, height), Image.LANCZOS)
    
    def dramatic_color_shift(self, image: Image.Image) -> Image.Image:
        """Apply dramatic color shifts."""
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Dramatic hue shifts
        hue_shifts = [-30, -20, -10, 10, 20, 30, 60, 90, 120]
        hue_shift = random.choice(hue_shifts)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 180
        hsv[:, :, 0] = hsv[:, :, 0].astype(np.uint8)
        
        # Dramatic saturation changes
        sat_factors = [0.3, 0.5, 1.5, 2.0]
        sat_factor = random.choice(sat_factors)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255).astype(np.uint8)
        
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(img_array)
    
    def dramatic_elastic_transform(self, image: Image.Image) -> Image.Image:
        """Apply dramatic elastic transformation."""
        img_array = np.array(image)
        shape = img_array.shape[:2]
        
        # More dramatic displacement
        alpha = random.choice([1500, 2000, 2500])
        sigma = random.choice([30, 40, 50])
        
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        x += dx
        y += dy
        
        transformed = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(transformed)
    
    def dramatic_perspective_transform(self, image: Image.Image) -> Image.Image:
        """Apply dramatic perspective transformation."""
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Define dramatic perspective points
        max_distortion = 0.3
        distortion = random.uniform(0.1, max_distortion)
        
        # Random perspective transformation
        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        # Add dramatic distortion
        dst_points = np.float32([
            [width * distortion, height * distortion],
            [width * (1 - distortion), height * distortion],
            [width * (1 - distortion), height * (1 - distortion)],
            [width * distortion, height * (1 - distortion)]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(img_array, matrix, (width, height))
        return Image.fromarray(transformed)
    
    def dramatic_histogram_equalization(self, image: Image.Image) -> Image.Image:
        """Apply histogram equalization for dramatic contrast."""
        img_array = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)
    
    def dramatic_edge_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply dramatic edge enhancement."""
        # Apply edge enhancement filter
        enhanced = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Sometimes combine with emboss for more dramatic effect
        if random.random() < 0.3:
            embossed = image.filter(ImageFilter.EMBOSS)
            # Blend the images
            enhanced = Image.blend(enhanced, embossed, 0.3)
        
        return enhanced
    
    def dramatic_sharpness(self, image: Image.Image) -> Image.Image:
        """Apply dramatic sharpness changes."""
        factors = [0.0, 0.3, 0.5, 2.0, 3.0, 4.0]
        factor = random.choice(factors)
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def dramatic_posterize(self, image: Image.Image) -> Image.Image:
        """Apply dramatic posterization."""
        bits = random.choice([2, 3, 4, 5])
        return ImageOps.posterize(image, bits)
    
    def dramatic_solarize(self, image: Image.Image) -> Image.Image:
        """Apply dramatic solarization."""
        thresholds = [128, 150, 180, 200]
        threshold = random.choice(thresholds)
        return ImageOps.solarize(image, threshold)
    
    def apply_dramatic_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply a dramatic combination of augmentations."""
        augmented = image.copy()
        
        # List of dramatic augmentation functions and their probabilities
        dramatic_augmentations = [
            (self.dramatic_rotate, 0.4),
            (self.dramatic_flip, 0.3),
            (self.dramatic_brightness, 0.5),
            (self.dramatic_contrast, 0.5),
            (self.dramatic_saturation, 0.4),
            (self.dramatic_noise, 0.3),
            (self.dramatic_blur, 0.2),
            (self.dramatic_crop, 0.4),
            (self.dramatic_color_shift, 0.4),
            (self.dramatic_elastic_transform, 0.2),
            (self.dramatic_perspective_transform, 0.2),
            (self.dramatic_histogram_equalization, 0.3),
            (self.dramatic_edge_enhancement, 0.3),
            (self.dramatic_sharpness, 0.4),
            (self.dramatic_posterize, 0.2),
            (self.dramatic_solarize, 0.1)
        ]
        
        # Apply 3-6 dramatic augmentations
        num_augmentations = random.randint(3, 6)
        selected_augmentations = random.sample(dramatic_augmentations, num_augmentations)
        
        for aug_func, prob in selected_augmentations:
            if random.random() < prob:
                try:
                    augmented = aug_func(augmented)
                except Exception as e:
                    logger.warning(f"Dramatic augmentation failed: {e}")
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

def generate_dramatic_augmented_images(dataset: Dataset, target_multiplier: int = 10) -> List[dict]:
    """
    Generate dramatically augmented images to reach target_multiplier times the original size.
    
    Args:
        dataset: Original dataset
        target_multiplier: How many times to multiply the original dataset size
        
    Returns:
        List of dramatically augmented image data
    """
    original_size = len(dataset)
    target_size = original_size * target_multiplier
    augmenter = DramaticImageAugmenter()
    
    logger.info(f"Generating {target_size} dramatically augmented images from {original_size} originals...")
    
    augmented_images = []
    
    # Calculate how many augmentations per original image
    augmentations_per_image = target_multiplier - 1  # -1 because we keep one original
    
    for i, item in enumerate(tqdm(dataset, desc="Generating dramatic augmentations")):
        try:
            # Keep the original image
            original_image = item["image"]
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            augmented_images.append({
                "image": original_image,
                "filename": f"dramatic_{i:04d}_original.jpg",
                "source_index": i
            })
            
            # Generate dramatic augmentations for this image
            for aug_idx in range(augmentations_per_image):
                try:
                    augmented_image = augmenter.apply_dramatic_augmentation(original_image)
                    
                    augmented_images.append({
                        "image": augmented_image,
                        "filename": f"dramatic_{i:04d}_{aug_idx:02d}.jpg",
                        "source_index": i
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to dramatically augment image {i}, augmentation {aug_idx}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing original image {i}: {e}")
            continue
    
    logger.info(f"Generated {len(augmented_images)} total dramatically augmented images")
    return augmented_images

def save_dramatic_augmented_images_locally(augmented_images: List[dict], output_dir: str = "dramatic_augmented_lanternflies"):
    """Save dramatically augmented images locally for verification."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Saving {len(augmented_images)} dramatically augmented images to {output_path}")
    
    for i, item in enumerate(tqdm(augmented_images, desc="Saving dramatic augmented images")):
        try:
            image = item["image"]
            filename = item["filename"]
            image.save(output_path / filename, "JPEG", quality=95)
        except Exception as e:
            logger.error(f"Error saving dramatic augmented image {i}: {e}")
    
    logger.info(f"Dramatically augmented images saved to {output_path}")

def create_dramatic_augmented_dataset(augmented_images: List[dict]) -> Dataset:
    """Create a Hugging Face dataset from dramatically augmented images."""
    logger.info("Creating dramatically augmented dataset...")
    
    # Match the features of the original dataset
    dataset_dict = {
        "image": [item["image"] for item in augmented_images],
        "filename": [item["filename"] for item in augmented_images]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Created dramatically augmented dataset with {len(dataset)} images")
    
    return dataset

def upload_dramatic_augmented_dataset(dataset: Dataset, repo_name: str = "rlogh/lanternfly_swatter_training"):
    """Upload the dramatically augmented dataset to Hugging Face."""
    logger.info(f"Uploading dramatically augmented dataset to {repo_name}...")
    
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
        
        logger.info("Dramatically augmented dataset uploaded successfully!")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

def main():
    """Main function to orchestrate the dramatic augmentation pipeline."""
    parser = argparse.ArgumentParser(description="Create dramatically augmented lanternfly dataset")
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
        
        # Generate dramatically augmented images
        augmented_images = generate_dramatic_augmented_images(original_dataset, args.multiplier)
        
        if not augmented_images:
            logger.error("No dramatically augmented images were generated")
            return
        
        # Save images locally first for verification
        save_dramatic_augmented_images_locally(augmented_images, "dramatic_augmented_lanternflies")
        
        # Create dramatically augmented dataset
        augmented_dataset = create_dramatic_augmented_dataset(augmented_images)
        
        # Upload to Hugging Face
        upload_dramatic_augmented_dataset(augmented_dataset, args.repo_name)
        
        logger.info("Dramatic augmentation pipeline completed successfully!")
        logger.info(f"Original: {len(original_dataset)} images")
        logger.info(f"Dramatically Augmented: {len(augmented_images)} images")
        logger.info(f"Multiplier achieved: {len(augmented_images) / len(original_dataset):.1f}x")
        
    except Exception as e:
        logger.error(f"Dramatic augmentation pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
