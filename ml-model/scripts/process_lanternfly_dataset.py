#!/usr/bin/env python3
"""
Script to process raw lanternfly images from Hugging Face dataset.
Downloads images, crops them around the subject, and uploads to training dataset.
"""

import os
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, login
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanternflyCropper:
    """Class to handle cropping of lanternfly images around the subject."""
    
    def __init__(self, min_contour_area: int = 1000):
        """
        Initialize the cropper.
        
        Args:
            min_contour_area: Minimum area for contour detection
        """
        self.min_contour_area = min_contour_area
        
    def detect_subject_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the main subject region in the image using contour detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (x, y, width, height) or None if no subject detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest contour that meets minimum area requirement
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        
        if not valid_contours:
            # If no large contours, use the largest one anyway
            valid_contours = [max(contours, key=cv2.contourArea)]
        
        # Get bounding box of the largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding around the detected region
        padding = 20
        img_height, img_width = image.shape[:2]
        
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_width - x, w + 2 * padding)
        h = min(img_height - y, h + 2 * padding)
        
        return (x, y, w, h)
    
    def crop_image(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop image to the specified bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Cropped image
        """
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    def make_square(self, image: np.ndarray, target_size: int = 512) -> np.ndarray:
        """
        Make image square by cropping to the smaller dimension and resizing.
        
        Args:
            image: Input image as numpy array
            target_size: Target size for the square image
            
        Returns:
            Square image
        """
        height, width = image.shape[:2]
        
        # Determine the smaller dimension
        min_dim = min(height, width)
        
        # Calculate crop coordinates to center the square
        if height > width:
            # Image is taller, crop height
            start_y = (height - min_dim) // 2
            end_y = start_y + min_dim
            cropped = image[start_y:end_y, :]
        else:
            # Image is wider, crop width
            start_x = (width - min_dim) // 2
            end_x = start_x + min_dim
            cropped = image[:, start_x:end_x]
        
        # Resize to target size
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
        return resized

    def process_image(self, image: np.ndarray, target_size: int = 512) -> np.ndarray:
        """
        Process a single image: detect subject, crop, and make square.
        
        Args:
            image: Input image as numpy array
            target_size: Target size for the square image
            
        Returns:
            Processed square image
        """
        bbox = self.detect_subject_region(image)
        
        if bbox is None:
            logger.warning("No subject detected, using original image")
            processed = image
        else:
            cropped = self.crop_image(image, bbox)
            
            # Ensure minimum size
            if cropped.shape[0] < 100 or cropped.shape[1] < 100:
                logger.warning("Cropped image too small, using original")
                processed = image
            else:
                processed = cropped
        
        # Make the image square
        square_image = self.make_square(processed, target_size)
        
        return square_image

def download_raw_dataset() -> Dataset:
    """Download the raw lanternfly dataset from Hugging Face."""
    logger.info("Downloading raw lanternfly dataset...")
    
    try:
        dataset = load_dataset("ddecosmo/raw_lanternflies_1", split="train")
        logger.info(f"Downloaded {len(dataset)} images")
        return dataset
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def process_images(dataset: Dataset, cropper: LanternflyCropper, max_images: Optional[int] = None, target_size: int = 512) -> List[dict]:
    """
    Process images from the dataset.
    
    Args:
        dataset: Hugging Face dataset
        cropper: LanternflyCropper instance
        max_images: Maximum number of images to process (None for all)
        target_size: Target size for square images
        
    Returns:
        List of processed image data
    """
    processed_images = []
    
    # Limit number of images if specified
    if max_images:
        dataset = dataset.select(range(min(max_images, len(dataset))))
    
    logger.info(f"Processing {len(dataset)} images to {target_size}x{target_size} squares...")
    
    for i, item in enumerate(tqdm(dataset, desc="Processing images")):
        try:
            # Get image
            image = item["image"]
            
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Process the image (detect subject, crop, and make square)
            processed_array = cropper.process_image(image_array, target_size)
            
            # Convert back to PIL Image and ensure RGB format
            processed_image = Image.fromarray(processed_array)
            if processed_image.mode != 'RGB':
                processed_image = processed_image.convert('RGB')
            
            # Store processed data
            processed_data = {
                "image": processed_image,
                "original_filename": f"processed_{i:04d}.jpg"
            }
            
            processed_images.append(processed_data)
            
        except Exception as e:
            logger.error(f"Error processing image {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(processed_images)} images")
    return processed_images

def create_training_dataset(processed_images: List[dict]) -> Dataset:
    """Create a Hugging Face dataset from processed images."""
    logger.info("Creating training dataset...")
    
    # Create dataset from processed images
    dataset_dict = {
        "image": [item["image"] for item in processed_images],
        "filename": [item["original_filename"] for item in processed_images]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Created dataset with {len(dataset)} images")
    
    return dataset

def save_images_locally(processed_images: List[dict], output_dir: str = "processed_lanternflies"):
    """
    Save processed images locally for verification.
    
    Args:
        processed_images: List of processed image data
        output_dir: Directory to save images
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Saving {len(processed_images)} images to {output_path}")
    
    for i, item in enumerate(tqdm(processed_images, desc="Saving images")):
        try:
            image = item["image"]
            filename = item["original_filename"]
            image.save(output_path / filename, "JPEG", quality=95)
        except Exception as e:
            logger.error(f"Error saving image {i}: {e}")
    
    logger.info(f"Images saved to {output_path}")

def upload_to_huggingface(dataset: Dataset, repo_name: str = "ddecosmo/lanternfly_swatter_training"):
    """
    Upload the processed dataset to Hugging Face.
    
    Args:
        dataset: Processed dataset
        repo_name: Name of the target repository
    """
    logger.info(f"Uploading dataset to {repo_name}...")
    
    try:
        # Check if user is logged in
        from huggingface_hub import whoami
        try:
            user_info = whoami()
            logger.info(f"Logged in as: {user_info['name']}")
        except Exception:
            logger.warning("Not logged in to Hugging Face. Please run 'huggingface-cli login' first.")
            logger.info("Skipping upload. Images have been saved locally.")
            return
        
        # Push to hub with the specified structure
        dataset.push_to_hub(
            repo_name,
            private=False,
            split="original"  # This will create the "original" split
        )
        logger.info("Dataset uploaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        logger.info("Images have been saved locally for manual upload.")

def main():
    """Main function to orchestrate the processing pipeline."""
    parser = argparse.ArgumentParser(description="Process lanternfly images for training")
    parser.add_argument("--max-images", type=int, default=None, 
                       help="Maximum number of images to process (for testing)")
    parser.add_argument("--min-contour-area", type=int, default=1000,
                       help="Minimum contour area for subject detection")
    parser.add_argument("--target-size", type=int, default=512,
                       help="Target size for square images (default: 512)")
    parser.add_argument("--repo-name", type=str, default="rlogh/lanternfly_swatter_training",
                       help="Target Hugging Face repository name")
    parser.add_argument("--test-only", action="store_true",
                       help="Only process a few images for testing")
    
    args = parser.parse_args()
    
    if args.test_only:
        args.max_images = 5
        logger.info("Running in test mode - processing only 5 images")
    
    try:
        # Initialize cropper
        cropper = LanternflyCropper(min_contour_area=args.min_contour_area)
        
        # Download raw dataset
        raw_dataset = download_raw_dataset()
        
        # Process images
        processed_images = process_images(raw_dataset, cropper, args.max_images, args.target_size)
        
        if not processed_images:
            logger.error("No images were successfully processed")
            return
        
        # Save images locally first for verification
        save_images_locally(processed_images, "processed_lanternflies")
        
        # Create training dataset
        training_dataset = create_training_dataset(processed_images)
        
        # Upload to Hugging Face
        upload_to_huggingface(training_dataset, args.repo_name)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
