#!/usr/bin/env python3
"""
Data Processing Script for Lantern Fly Classification
Handles data loading, preprocessing, and augmentation
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing, augmentation, and splitting"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RandomCrop(height=224, width=224, p=0.8),
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation pipeline (no augmentation)
        self.validation_pipeline = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.raw_dir,
            self.processed_dir / "train" / "lantern_fly",
            self.processed_dir / "train" / "non_lantern_fly",
            self.processed_dir / "val" / "lantern_fly",
            self.processed_dir / "val" / "non_lantern_fly",
            self.processed_dir / "test" / "lantern_fly",
            self.processed_dir / "test" / "non_lantern_fly"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def load_insecta_dataset(self, insecta_path: str) -> List[str]:
        """Load Insecta dataset and return list of image paths"""
        logger.info(f"Loading Insecta dataset from: {insecta_path}")
        
        if not os.path.exists(insecta_path):
            logger.error(f"Insecta dataset not found at: {insecta_path}")
            return []
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for root, dirs, files in os.walk(insecta_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_paths)} images in Insecta dataset")
        return image_paths
    
    def sample_insecta_images(self, image_paths: List[str], target_count: int = 2000) -> List[str]:
        """Sample images from Insecta dataset to balance with lantern fly data"""
        if len(image_paths) <= target_count:
            logger.info(f"Using all {len(image_paths)} Insecta images")
            return image_paths
        
        sampled_paths = random.sample(image_paths, target_count)
        logger.info(f"Sampled {len(sampled_paths)} images from Insecta dataset")
        return sampled_paths
    
    def copy_lantern_fly_images(self, source_dir: str, target_dir: str):
        """Copy lantern fly images to target directory"""
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        if not source_path.exists():
            logger.warning(f"Lantern fly source directory not found: {source_path}")
            return
        
        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        copied_count = 0
        
        for file_path in source_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                target_file = target_path / file_path.name
                shutil.copy2(file_path, target_file)
                copied_count += 1
        
        logger.info(f"Copied {copied_count} lantern fly images to {target_path}")
    
    def copy_non_lantern_fly_images(self, image_paths: List[str], target_dir: str):
        """Copy non-lantern fly images to target directory"""
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        for i, image_path in enumerate(image_paths):
            target_file = target_path / f"insecta_{i:06d}{Path(image_path).suffix}"
            shutil.copy2(image_path, target_file)
            copied_count += 1
        
        logger.info(f"Copied {copied_count} non-lantern fly images to {target_path}")
    
    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Split data into train/validation/test sets"""
        logger.info("Splitting data into train/validation/test sets")
        
        # Get all image files
        lantern_fly_dir = self.processed_dir / "train" / "lantern_fly"
        non_lantern_fly_dir = self.processed_dir / "train" / "non_lantern_fly"
        
        # Get lantern fly images
        lantern_fly_images = list(lantern_fly_dir.glob("*"))
        non_lantern_fly_images = list(non_lantern_fly_dir.glob("*"))
        
        logger.info(f"Found {len(lantern_fly_images)} lantern fly images")
        logger.info(f"Found {len(non_lantern_fly_images)} non-lantern fly images")
        
        # Split lantern fly images
        if len(lantern_fly_images) > 0:
            train_lf, temp_lf = train_test_split(
                lantern_fly_images, 
                test_size=(val_ratio + test_ratio), 
                random_state=42
            )
            val_lf, test_lf = train_test_split(
                temp_lf, 
                test_size=(test_ratio / (val_ratio + test_ratio)), 
                random_state=42
            )
            
            # Move to appropriate directories
            self._move_images(train_lf, self.processed_dir / "train" / "lantern_fly")
            self._move_images(val_lf, self.processed_dir / "val" / "lantern_fly")
            self._move_images(test_lf, self.processed_dir / "test" / "lantern_fly")
        
        # Split non-lantern fly images
        if len(non_lantern_fly_images) > 0:
            train_nlf, temp_nlf = train_test_split(
                non_lantern_fly_images, 
                test_size=(val_ratio + test_ratio), 
                random_state=42
            )
            val_nlf, test_nlf = train_test_split(
                temp_nlf, 
                test_size=(test_ratio / (val_ratio + test_ratio)), 
                random_state=42
            )
            
            # Move to appropriate directories
            self._move_images(train_nlf, self.processed_dir / "train" / "non_lantern_fly")
            self._move_images(val_nlf, self.processed_dir / "val" / "non_lantern_fly")
            self._move_images(test_nlf, self.processed_dir / "test" / "non_lantern_fly")
        
        logger.info("Data splitting completed")
    
    def _move_images(self, image_paths: List[Path], target_dir: Path):
        """Move images to target directory"""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for image_path in image_paths:
            target_file = target_dir / image_path.name
            shutil.move(str(image_path), str(target_file))
    
    def augment_lantern_fly_data(self, augmentation_factor: int = 4):
        """Augment lantern fly data to balance classes"""
        logger.info(f"Augmenting lantern fly data with factor {augmentation_factor}")
        
        train_lf_dir = self.processed_dir / "train" / "lantern_fly"
        val_lf_dir = self.processed_dir / "val" / "lantern_fly"
        
        # Augment training data
        self._augment_images_in_directory(train_lf_dir, augmentation_factor)
        
        # Augment validation data (smaller factor)
        self._augment_images_in_directory(val_lf_dir, 2)
    
    def _augment_images_in_directory(self, directory: Path, factor: int):
        """Augment images in a directory"""
        if not directory.exists():
            return
        
        original_images = list(directory.glob("*"))
        logger.info(f"Augmenting {len(original_images)} images in {directory}")
        
        for i, image_path in enumerate(original_images):
            try:
                # Load image
                image = np.array(Image.open(image_path))
                
                # Generate augmented versions
                for j in range(factor - 1):
                    augmented = self.augmentation_pipeline(image=image)["image"]
                    
                    # Save augmented image
                    augmented_path = directory / f"{image_path.stem}_aug_{j+1}{image_path.suffix}"
                    # Convert tensor back to PIL Image for saving
                    if hasattr(augmented, 'numpy'):
                        augmented_np = augmented.numpy().transpose(1, 2, 0)
                        augmented_np = (augmented_np * 255).astype(np.uint8)
                        Image.fromarray(augmented_np).save(augmented_path)
                
            except Exception as e:
                logger.error(f"Error augmenting {image_path}: {e}")
    
    def create_data_summary(self) -> pd.DataFrame:
        """Create a summary of the processed data"""
        summary_data = []
        
        for split in ["train", "val", "test"]:
            for class_name in ["lantern_fly", "non_lantern_fly"]:
                class_dir = self.processed_dir / split / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob("*")))
                    summary_data.append({
                        "split": split,
                        "class": class_name,
                        "count": count
                    })
        
        summary_df = pd.DataFrame(summary_data)
        logger.info("Data Summary:")
        logger.info(summary_df.to_string(index=False))
        
        return summary_df
    
    def process_all_data(self, 
                        lantern_fly_source: str = None,
                        insecta_path: str = None,
                        target_insecta_count: int = 2000,
                        augmentation_factor: int = 4):
        """Process all data: load, copy, split, and augment"""
        logger.info("Starting complete data processing pipeline")
        
        # Step 1: Copy lantern fly images (if source provided)
        if lantern_fly_source:
            self.copy_lantern_fly_images(
                lantern_fly_source, 
                self.processed_dir / "train" / "lantern_fly"
            )
        
        # Step 2: Load and copy Insecta images
        if insecta_path:
            insecta_images = self.load_insecta_dataset(insecta_path)
            sampled_images = self.sample_insecta_images(insecta_images, target_insecta_count)
            self.copy_non_lantern_fly_images(
                sampled_images, 
                self.processed_dir / "train" / "non_lantern_fly"
            )
        
        # Step 3: Split data
        self.split_data()
        
        # Step 4: Augment lantern fly data
        self.augment_lantern_fly_data(augmentation_factor)
        
        # Step 5: Create summary
        summary = self.create_data_summary()
        
        logger.info("Data processing pipeline completed")
        return summary

def main():
    """Main function for testing"""
    processor = DataProcessor()
    
    # Example usage (uncomment when you have data)
    # summary = processor.process_all_data(
    #     lantern_fly_source="path/to/lantern/fly/photos",
    #     insecta_path="path/to/insecta/dataset",
    #     target_insecta_count=2000,
    #     augmentation_factor=4
    # )
    
    print("Data processor initialized successfully!")
    print("Ready to process data when you have:")
    print("1. Lantern fly photos (500 images)")
    print("2. Insecta dataset")

if __name__ == "__main__":
    main()
