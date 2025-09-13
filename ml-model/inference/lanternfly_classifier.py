#!/usr/bin/env python3
"""
Lanternfly classification inference service.
Provides a simple API for classifying images as lanternflies or not.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import logging
from pathlib import Path
import sys
import os
import io

# Add the scripts directory to the path so we can import our model
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from model_architecture import LanternflyClassifier, ModelManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanternflyInferenceService:
    """Service for running lanternfly classification inference."""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the inference service.
        
        Args:
            model_path: Path to the trained model
            device: Device to run inference on ('cpu' or 'cuda')
        """
        if model_path is None:
            # Default to the best model
            model_path = Path(__file__).parent.parent / 'models' / 'best_model.pth'
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.model_path = model_path
        
        # Initialize model
        self._load_model()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Lanternfly inference service initialized on {device}")
        logger.info(f"Model loaded from: {model_path}")
    
    def _load_model(self):
        """Load the trained model."""
        try:
            # Create model architecture
            self.model = LanternflyClassifier(num_classes=2, pretrained=False)
            self.model_manager = ModelManager(self.model, self.device)
            
            # Load trained weights
            checkpoint = self.model_manager.load_model(str(self.model_path))
            
            # Set to evaluation mode
            self.model_manager.set_evaluation_mode()
            
            logger.info(f"Model loaded successfully from epoch {checkpoint['epoch']}")
            logger.info(f"Model validation accuracy: {checkpoint['val_accuracy']:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def classify_image(self, image_path: str) -> dict:
        """
        Classify an image as lanternfly or not.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            # Run inference
            self.model_manager.set_evaluation_mode()
            
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model_manager.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            # Get results
            confidence_scores = probabilities[0].cpu().numpy()
            is_lanternfly = bool(predicted.item() == 1)
            confidence = float(max(confidence_scores))
            
            # Determine points (10 for lanternfly, 0 for non-lanternfly)
            points_awarded = 10 if is_lanternfly else 0
            
            result = {
                'success': True,
                'is_lantern_fly': is_lanternfly,
                'confidence_score': confidence,
                'points_awarded': points_awarded,
                'model_version': '1.0.0-trained',
                'class_probabilities': {
                    'non_lanternfly': float(confidence_scores[0]),
                    'lanternfly': float(confidence_scores[1])
                }
            }
            
            logger.info(f"Classification result: {'Lanternfly' if is_lanternfly else 'Non-Lanternfly'} "
                       f"(confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_lantern_fly': False,
                'confidence_score': 0.0,
                'points_awarded': 0,
                'model_version': '1.0.0-error'
            }
    
    def classify_image_from_bytes(self, image_bytes: bytes) -> dict:
        """
        Classify an image from bytes data.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Run inference
            self.model_manager.set_evaluation_mode()
            
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model_manager.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            # Get results
            confidence_scores = probabilities[0].cpu().numpy()
            is_lanternfly = bool(predicted.item() == 1)
            confidence = float(max(confidence_scores))
            
            # Determine points
            points_awarded = 10 if is_lanternfly else 0
            
            result = {
                'success': True,
                'is_lantern_fly': is_lanternfly,
                'confidence_score': confidence,
                'points_awarded': points_awarded,
                'model_version': '1.0.0-trained',
                'class_probabilities': {
                    'non_lanternfly': float(confidence_scores[0]),
                    'lanternfly': float(confidence_scores[1])
                }
            }
            
            logger.info(f"Classification result: {'Lanternfly' if is_lanternfly else 'Non-Lanternfly'} "
                       f"(confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_lantern_fly': False,
                'confidence_score': 0.0,
                'points_awarded': 0,
                'model_version': '1.0.0-error'
            }

# Global service instance
_inference_service = None

def get_inference_service() -> LanternflyInferenceService:
    """Get the global inference service instance."""
    global _inference_service
    if _inference_service is None:
        _inference_service = LanternflyInferenceService()
    return _inference_service

def classify_image(image_path: str) -> dict:
    """Convenience function to classify an image."""
    service = get_inference_service()
    return service.classify_image(image_path)

def classify_image_from_bytes(image_bytes: bytes) -> dict:
    """Convenience function to classify an image from bytes."""
    service = get_inference_service()
    return service.classify_image_from_bytes(image_bytes)

if __name__ == "__main__":
    # Test the service
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python lanternfly_classifier.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize service
    service = LanternflyInferenceService()
    
    # Classify image
    result = service.classify_image(image_path)
    
    # Print results
    print(json.dumps(result, indent=2))
