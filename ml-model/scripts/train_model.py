#!/usr/bin/env python3
"""
Training Script for Lantern Fly Classification Model
Uses EfficientNet-B0 for binary classification: Lantern Fly vs Non-Lantern Fly
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanternFlyDataset(Dataset):
    """Custom dataset for lantern fly classification"""
    
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.image_paths, self.labels = self._load_data()
        
        logger.info(f"Loaded {len(self.image_paths)} images for {split} split")
        logger.info(f"Class distribution: {np.bincount(self.labels)}")
    
    def _load_data(self) -> Tuple[List[str], List[int]]:
        """Load image paths and labels"""
        image_paths = []
        labels = []
        
        # Lantern fly images (label = 1)
        lantern_fly_dir = self.data_dir / "processed" / self.split / "lantern_fly"
        if lantern_fly_dir.exists():
            for img_path in lantern_fly_dir.glob("*"):
                if img_path.is_file():
                    image_paths.append(str(img_path))
                    labels.append(1)
        
        # Non-lantern fly images (label = 0)
        non_lantern_fly_dir = self.data_dir / "processed" / self.split / "non_lantern_fly"
        if non_lantern_fly_dir.exists():
            for img_path in non_lantern_fly_dir.glob("*"):
                if img_path.is_file():
                    image_paths.append(str(img_path))
                    labels.append(0)
        
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label

class LanternFlyClassifier(nn.Module):
    """EfficientNet-B0 based classifier for lantern fly detection"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(LanternFlyClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Handles model training, validation, and evaluation"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 model_dir: str = "models",
                 device: str = None):
        
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Define transforms
        self.train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RandomCrop(height=224, width=224, p=0.8),
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Initialize model
        self.model = LanternFlyClassifier(num_classes=2, pretrained=True)
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def create_data_loaders(self, batch_size: int = 32, num_workers: int = 4):
        """Create data loaders for training and validation"""
        
        # Create datasets
        train_dataset = LanternFlyDataset(
            self.data_dir, 
            split="train", 
            transform=self.train_transform
        )
        
        val_dataset = LanternFlyDataset(
            self.data_dir, 
            split="val", 
            transform=self.val_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    def train_epoch(self, optimizer, criterion) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, criterion) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, 
              epochs: int = 50,
              learning_rate: float = 0.001,
              batch_size: int = 32,
              save_best: bool = True):
        """Train the model"""
        
        logger.info("Starting training...")
        
        # Create data loaders
        self.create_data_loaders(batch_size=batch_size)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            logger.info(f'Epoch {epoch+1}/{epochs}')
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"best_model_epoch_{epoch+1}_acc_{val_acc:.2f}.pth")
                logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        logger.info("Training completed!")
        return self.history
    
    def evaluate(self, test_split: str = "test") -> Dict:
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")
        
        # Create test dataset
        test_dataset = LanternFlyDataset(
            self.data_dir, 
            split=test_split, 
            transform=self.val_transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        
        # Classification report
        class_names = ['Non-Lantern Fly', 'Lantern Fly']
        report = classification_report(all_targets, all_predictions, 
                                    target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        logger.info(f"Test Accuracy: {accuracy:.2f}%")
        logger.info("Classification Report:")
        logger.info(classification_report(all_targets, all_predictions, target_names=class_names))
        
        return results
    
    def save_model(self, filename: str = None):
        """Save model and training history"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lantern_fly_classifier_{timestamp}.pth"
        
        model_path = self.model_dir / filename
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': 'EfficientNet-B0',
            'num_classes': 2,
            'class_names': ['Non-Lantern Fly', 'Lantern Fly'],
            'training_history': self.history,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"Model saved to: {model_path}")
        
        # Save training history as JSON
        history_path = self.model_dir / f"training_history_{filename.replace('.pth', '.json')}"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return model_path
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('training_history', {})
        
        logger.info(f"Model loaded from: {model_path}")
        return checkpoint
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if not self.history['train_loss']:
            logger.warning("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()

def main():
    """Main training function"""
    logger.info("Starting Lantern Fly Classification Training")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Check if data exists
    train_lf_dir = trainer.data_dir / "processed" / "train" / "lantern_fly"
    train_nlf_dir = trainer.data_dir / "processed" / "train" / "non_lantern_fly"
    
    if not train_lf_dir.exists() or not train_nlf_dir.exists():
        logger.error("Training data not found!")
        logger.error("Please run data processing first:")
        logger.error("1. Add your lantern fly photos to data/processed/train/lantern_fly/")
        logger.error("2. Add Insecta dataset to data/processed/train/non_lantern_fly/")
        return
    
    # Count images
    lf_count = len(list(train_lf_dir.glob("*")))
    nlf_count = len(list(train_nlf_dir.glob("*")))
    
    logger.info(f"Found {lf_count} lantern fly images")
    logger.info(f"Found {nlf_count} non-lantern fly images")
    
    if lf_count == 0 or nlf_count == 0:
        logger.error("Insufficient data for training!")
        return
    
    # Train model
    history = trainer.train(
        epochs=50,
        learning_rate=0.001,
        batch_size=32,
        save_best=True
    )
    
    # Evaluate model
    results = trainer.evaluate()
    
    # Plot training history
    trainer.plot_training_history("models/training_history.png")
    
    # Save final model
    final_model_path = trainer.save_model("final_lantern_fly_classifier.pth")
    
    logger.info("Training completed successfully!")
    logger.info(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
