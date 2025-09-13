#!/usr/bin/env python3
"""
Data loader for Hugging Face lanternfly dataset.
Handles loading, splitting, and preprocessing of the dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class LanternflyDataset(Dataset):
    """Custom dataset class for lanternfly classification."""
    
    def __init__(self, images: List[Image.Image], labels: List[int], transform=None):
        """
        Initialize the dataset.
        
        Args:
            images: List of PIL Images
            labels: List of labels (0 for negative, 1 for positive)
            transform: Optional transform to be applied on images
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class LanternflyDataLoader:
    """Data loader for the lanternfly classification task."""
    
    def __init__(self, repo_name: str = "rlogh/lanternfly_swatter_training", 
                 batch_size: int = 32, image_size: int = 512, 
                 train_split: float = 0.7, val_split: float = 0.15, test_split: float = 0.15):
        """
        Initialize the data loader.
        
        Args:
            repo_name: Hugging Face repository name
            batch_size: Batch size for training
            image_size: Size to resize images to
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
        """
        self.repo_name = repo_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and prepare data
        self._load_data()
        self._create_splits()
        self._create_dataloaders()
    
    def _load_data(self):
        """Load data from Hugging Face dataset with streaming to save memory."""
        logger.info("Loading data from Hugging Face dataset...")
        
        # Load datasets with streaming to reduce memory usage
        original_dataset = load_dataset(self.repo_name, split="original", streaming=True)
        augmented_dataset = load_dataset(self.repo_name, split="augmented", streaming=True)
        negatives_dataset = load_dataset(self.repo_name, split="negatives", streaming=True)
        
        # Prepare positive data (real + artificial for training)
        self.positive_images = []
        self.positive_labels = []
        
        # Add original images (real data) - limit to prevent memory issues
        logger.info("Loading original images...")
        original_count = 0
        for item in original_dataset:
            self.positive_images.append(item["image"])
            self.positive_labels.append(1)  # Positive label
            original_count += 1
            if original_count % 50 == 0:
                logger.info(f"Loaded {original_count} original images...")
        
        logger.info(f"Loaded {original_count} original images")
        
        # Add augmented images (artificial data - for training only) - limit to prevent memory issues
        logger.info("Loading augmented images...")
        augmented_count = 0
        max_augmented = 1000  # Limit augmented images to prevent memory issues
        for item in augmented_dataset:
            if augmented_count >= max_augmented:
                break
            self.positive_images.append(item["image"])
            self.positive_labels.append(1)  # Positive label
            augmented_count += 1
            if augmented_count % 100 == 0:
                logger.info(f"Loaded {augmented_count} augmented images...")
        
        logger.info(f"Loaded {augmented_count} augmented images (limited to {max_augmented})")
        
        # Prepare negative data (real data only) - limit to prevent memory issues
        self.negative_images = []
        self.negative_labels = []
        
        logger.info("Loading negative images...")
        negative_count = 0
        max_negatives = 1000  # Limit negative images to prevent memory issues
        for item in negatives_dataset:
            if negative_count >= max_negatives:
                break
            self.negative_images.append(item["image"])
            self.negative_labels.append(0)  # Negative label
            negative_count += 1
            if negative_count % 100 == 0:
                logger.info(f"Loaded {negative_count} negative images...")
        
        logger.info(f"Loaded {negative_count} negative images (limited to {max_negatives})")
        
        logger.info(f"Total positive images: {len(self.positive_images)}")
        logger.info(f"Total negative images: {len(self.negative_images)}")
        
        # Store original data for validation/test splits
        self.original_positive_images = []
        self.original_positive_labels = []
        
        # Reload original dataset for validation/test splits
        original_dataset_reload = load_dataset(self.repo_name, split="original", streaming=True)
        for item in original_dataset_reload:
            self.original_positive_images.append(item["image"])
            self.original_positive_labels.append(1)
    
    def _create_splits(self):
        """Create train/validation/test splits."""
        logger.info("Creating data splits...")
        
        # Split original positive data for validation and test
        original_positive_size = len(self.original_positive_images)
        val_size = int(original_positive_size * self.val_split)
        test_size = int(original_positive_size * self.test_split)
        train_original_size = original_positive_size - val_size - test_size
        
        # Split original positive data
        original_splits = random_split(
            list(range(original_positive_size)), 
            [train_original_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Split negative data
        negative_size = len(self.negative_images)
        val_neg_size = int(negative_size * self.val_split)
        test_neg_size = int(negative_size * self.test_split)
        train_neg_size = negative_size - val_neg_size - test_neg_size
        
        negative_splits = random_split(
            list(range(negative_size)),
            [train_neg_size, val_neg_size, test_neg_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create training data (original + augmented positive + negative)
        self.train_images = []
        self.train_labels = []
        
        # Add original positive data for training
        for idx in original_splits[0]:
            self.train_images.append(self.original_positive_images[idx])
            self.train_labels.append(1)
        
        # Add all augmented positive data for training
        for i in range(len(self.positive_images) - len(self.original_positive_images)):
            self.train_images.append(self.positive_images[len(self.original_positive_images) + i])
            self.train_labels.append(1)
        
        # Add negative data for training
        for idx in negative_splits[0]:
            self.train_images.append(self.negative_images[idx])
            self.train_labels.append(0)
        
        # Create validation data (original positive + negative only)
        self.val_images = []
        self.val_labels = []
        
        for idx in original_splits[1]:
            self.val_images.append(self.original_positive_images[idx])
            self.val_labels.append(1)
        
        for idx in negative_splits[1]:
            self.val_images.append(self.negative_images[idx])
            self.val_labels.append(0)
        
        # Create test data (original positive + negative only)
        self.test_images = []
        self.test_labels = []
        
        for idx in original_splits[2]:
            self.test_images.append(self.original_positive_images[idx])
            self.test_labels.append(1)
        
        for idx in negative_splits[2]:
            self.test_images.append(self.negative_images[idx])
            self.test_labels.append(0)
        
        logger.info(f"Training set: {len(self.train_images)} images")
        logger.info(f"Validation set: {len(self.val_images)} images")
        logger.info(f"Test set: {len(self.test_images)} images")
        
        # Log class distribution
        train_pos = sum(self.train_labels)
        train_neg = len(self.train_labels) - train_pos
        val_pos = sum(self.val_labels)
        val_neg = len(self.val_labels) - val_pos
        test_pos = sum(self.test_labels)
        test_neg = len(self.test_labels) - test_pos
        
        logger.info(f"Training - Positive: {train_pos}, Negative: {train_neg}")
        logger.info(f"Validation - Positive: {val_pos}, Negative: {val_neg}")
        logger.info(f"Test - Positive: {test_pos}, Negative: {test_neg}")
    
    def _create_dataloaders(self):
        """Create PyTorch DataLoaders."""
        logger.info("Creating DataLoaders...")
        
        # Create datasets
        train_dataset = LanternflyDataset(
            self.train_images, self.train_labels, self.train_transform
        )
        val_dataset = LanternflyDataset(
            self.val_images, self.val_labels, self.val_test_transform
        )
        test_dataset = LanternflyDataset(
            self.test_images, self.test_labels, self.val_test_transform
        )
        
        # Create DataLoaders with reduced num_workers to prevent memory issues
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        logger.info("DataLoaders created successfully!")
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get the train, validation, and test DataLoaders."""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset."""
        return {
            "train_size": len(self.train_images),
            "val_size": len(self.val_images),
            "test_size": len(self.test_images),
            "total_positive": len(self.positive_images),
            "total_negative": len(self.negative_images),
            "image_size": self.image_size,
            "batch_size": self.batch_size
        }

