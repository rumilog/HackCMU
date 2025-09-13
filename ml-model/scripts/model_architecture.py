#!/usr/bin/env python3
"""
Model architecture for lanternfly binary classification.
Uses EfficientNet-B0 as the base model with custom classifier head.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class LanternflyClassifier(nn.Module):
    """Binary classifier for lanternfly detection using EfficientNet-B0."""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.2):
        """
        Initialize the lanternfly classifier.
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(LanternflyClassifier, self).__init__()
        
        # Load EfficientNet-B0
        try:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            logger.info("Loaded EfficientNet-B0 with pretrained weights")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet-B0: {e}")
            # Fallback to ResNet-18 if EfficientNet is not available
            self.backbone = models.resnet18(pretrained=pretrained)
            logger.info("Fallback: Loaded ResNet-18 with pretrained weights")
        
        # Get the number of features from the backbone
        if hasattr(self.backbone, 'classifier'):
            # EfficientNet
            num_features = self.backbone.classifier[1].in_features
            # Replace the classifier
            self.backbone.classifier = nn.Identity()
        else:
            # ResNet
            num_features = self.backbone.fc.in_features
            # Replace the final layer
            self.backbone.fc = nn.Identity()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier_weights()
        
        logger.info(f"Model initialized with {num_features} input features")
        logger.info(f"Classifier output: {num_classes} classes")
    
    def _initialize_classifier_weights(self):
        """Initialize the classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the model."""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify using custom head
        output = self.classifier(features)
        
        return output
    
    def freeze_backbone(self):
        """Freeze the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
    
    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }

class ModelManager:
    """Manager class for model operations."""
    
    def __init__(self, model: LanternflyClassifier, device: torch.device):
        """
        Initialize the model manager.
        
        Args:
            model: The lanternfly classifier model
            device: Device to run the model on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        logger.info(f"Model moved to device: {device}")
        logger.info(f"Model info: {self.model.get_model_info()}")
    
    def save_model(self, filepath: str, epoch: int, optimizer_state: dict, 
                   train_loss: float, val_loss: float, val_accuracy: float):
        """
        Save the model checkpoint.
        
        Args:
            filepath: Path to save the model
            epoch: Current epoch
            optimizer_state: Optimizer state dict
            train_loss: Training loss
            val_loss: Validation loss
            val_accuracy: Validation accuracy
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'model_info': self.model.get_model_info()
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> dict:
        """
        Load a model checkpoint.
        
        Args:
            filepath: Path to the model checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        logger.info(f"Checkpoint validation accuracy: {checkpoint['val_accuracy']:.4f}")
        
        return checkpoint
    
    def set_training_mode(self):
        """Set the model to training mode."""
        self.model.train()
    
    def set_evaluation_mode(self):
        """Set the model to evaluation mode."""
        self.model.eval()
    
    def get_predictions(self, dataloader, return_probabilities: bool = False):
        """
        Get predictions from the model.
        
        Args:
            dataloader: DataLoader to get predictions for
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions and optionally probabilities
        """
        self.set_evaluation_mode()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                
                if return_probabilities:
                    probabilities = torch.softmax(outputs, dim=1)
                    all_probabilities.append(probabilities.cpu())
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.append(predicted.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        
        if return_probabilities:
            probabilities = torch.cat(all_probabilities, dim=0)
            return predictions, probabilities
        
        return predictions

