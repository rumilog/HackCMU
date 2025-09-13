#!/usr/bin/env python3
"""
Main training script for lanternfly binary classification.
Trains a model to distinguish between lanternflies and non-lanternflies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import gc
import psutil
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

from data_loader import LanternflyDataLoader
from model_architecture import LanternflyClassifier, ModelManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingManager:
    """Manager class for training operations."""
    
    def __init__(self, model_manager: ModelManager, train_loader, val_loader, 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """
        Initialize the training manager.
        
        Args:
            model_manager: Model manager instance
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model_manager = model_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = model_manager.device
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            model_manager.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
        logger.info(f"Training manager initialized with learning rate: {learning_rate}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model_manager.set_training_mode()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model_manager.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
            
            # Memory management - clear cache every 10 batches
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model_manager.set_evaluation_mode()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(self.val_loader, desc="Validation")):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model_manager.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Memory management - clear cache every 10 batches
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self, num_epochs: int = 50, save_dir: str = "models", 
              early_stopping_patience: int = 10) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save models
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_accuracy = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_accuracy)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                best_model_path = save_path / "best_model.pth"
                self.model_manager.save_model(
                    str(best_model_path),
                    epoch,
                    self.optimizer.state_dict(),
                    train_loss,
                    val_loss,
                    val_accuracy
                )
                
                logger.info(f"New best model saved! Val Accuracy: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            logger.info("-" * 50)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}% at epoch {self.best_epoch+1}")
        
        # Save final model
        final_model_path = save_path / "final_model.pth"
        self.model_manager.save_model(
            str(final_model_path),
            epoch,
            self.optimizer.state_dict(),
            train_loss,
            val_loss,
            val_accuracy
        )
        
        # Save training history
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "training_time": training_time,
            "total_epochs": epoch + 1
        }
        
        history_path = save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves(save_path)
        
        return history
    
    def plot_training_curves(self, save_path: Path):
        """Plot and save training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='green')
        ax2.axhline(y=self.best_val_accuracy, color='red', linestyle='--', 
                   label=f'Best: {self.best_val_accuracy:.2f}%')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate curve
        ax3.plot(self.learning_rates, label='Learning Rate', color='orange')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
        
        # Combined plot
        ax4_twin = ax4.twinx()
        ax4.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        ax4.plot(self.val_losses, label='Val Loss', color='red', alpha=0.7)
        ax4_twin.plot(self.val_accuracies, label='Val Accuracy', color='green', alpha=0.7)
        ax4.set_title('Training Progress')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='black')
        ax4_twin.set_ylabel('Accuracy (%)', color='green')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training curves saved")

def main():
    """Main training function."""
    # Configuration - reduced batch size to prevent memory issues
    config = {
        "repo_name": "rlogh/lanternfly_swatter_training",
        "batch_size": 8,  # Reduced from 32 to prevent memory issues
        "image_size": 224,  # Reduced from 512 to prevent memory issues
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 5,  # Reduced to 5 epochs for faster training
        "early_stopping_patience": 5,  # Reduced from 10
        "save_dir": "models"
    }
    
    logger.info("Starting lanternfly binary classification training...")
    logger.info(f"Configuration: {config}")
    
    # Memory monitoring
    memory_info = psutil.virtual_memory()
    logger.info(f"Available memory: {memory_info.available / (1024**3):.2f} GB")
    logger.info(f"Memory usage: {memory_info.percent}%")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Clear any existing cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    try:
        # Load data
        logger.info("Loading data...")
        data_loader = LanternflyDataLoader(
            repo_name=config["repo_name"],
            batch_size=config["batch_size"],
            image_size=config["image_size"]
        )
        
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()
        dataset_info = data_loader.get_dataset_info()
        
        logger.info(f"Dataset info: {dataset_info}")
        
        # Create model
        logger.info("Creating model...")
        model = LanternflyClassifier(num_classes=2, pretrained=True)
        model_manager = ModelManager(model, device)
        
        # Create training manager
        training_manager = TrainingManager(
            model_manager=model_manager,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Train model
        logger.info("Starting training...")
        history = training_manager.train(
            num_epochs=config["num_epochs"],
            save_dir=config["save_dir"],
            early_stopping_patience=config["early_stopping_patience"]
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {history['best_val_accuracy']:.2f}%")
        logger.info(f"Training time: {history['training_time']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
