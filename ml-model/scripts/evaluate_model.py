#!/usr/bin/env python3
"""
Evaluation script for the lanternfly binary classification model.
Provides comprehensive evaluation metrics and visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from pathlib import Path
import json
from typing import Dict, Tuple, List
import logging
from tqdm import tqdm

from data_loader import LanternflyDataLoader
from model_architecture import LanternflyClassifier, ModelManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for comprehensive model evaluation."""
    
    def __init__(self, model_manager: ModelManager, test_loader):
        """
        Initialize the evaluator.
        
        Args:
            model_manager: Model manager instance
            test_loader: Test data loader
        """
        self.model_manager = model_manager
        self.test_loader = test_loader
        self.device = model_manager.device
        
        # Class names
        self.class_names = ['Non-Lanternfly', 'Lanternfly']
    
    def get_predictions_and_labels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions, labels, and probabilities from the test set.
        
        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        self.model_manager.set_evaluation_mode()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model_manager.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                         probabilities: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions
            labels: True labels
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None)
        recall_per_class = recall_score(labels, predictions, average=None)
        f1_per_class = f1_score(labels, predictions, average=None)
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, save_path: Path):
        """Plot and save ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_distribution(self, labels: np.ndarray, save_path: Path):
        """Plot class distribution."""
        plt.figure(figsize=(8, 6))
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar([self.class_names[i] for i in unique], counts, color=['skyblue', 'lightcoral'])
        plt.title('Test Set Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(save_path / "class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_confidence(self, probabilities: np.ndarray, labels: np.ndarray, 
                                 predictions: np.ndarray, save_path: Path):
        """Plot prediction confidence distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confidence for correct predictions
        correct_mask = predictions == labels
        correct_confidences = probabilities[correct_mask, predictions[correct_mask]]
        
        # Confidence for incorrect predictions
        incorrect_mask = predictions != labels
        incorrect_confidences = probabilities[incorrect_mask, predictions[incorrect_mask]]
        
        ax1.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
        ax1.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.legend()
        ax1.grid(True)
        
        # Confidence by class
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = labels == class_idx
            class_confidences = probabilities[class_mask, class_idx]
            ax2.hist(class_confidences, bins=30, alpha=0.7, label=class_name)
        
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence by True Class')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / "prediction_confidence.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, save_dir: str = "evaluation_results") -> Dict:
        """
        Perform comprehensive evaluation.
        
        Args:
            save_dir: Directory to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Get predictions
        predictions, labels, probabilities = self.get_predictions_and_labels()
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, labels, probabilities)
        
        # Print results
        logger.info("Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
        logger.info(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        logger.info("\nPer-class metrics:")
        for i, class_name in enumerate(self.class_names):
            logger.info(f"{class_name}:")
            logger.info(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
            logger.info(f"  Recall: {metrics['recall_per_class'][i]:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        self.plot_confusion_matrix(np.array(metrics['confusion_matrix']), save_path)
        self.plot_roc_curve(np.array(metrics['fpr']), np.array(metrics['tpr']), 
                           metrics['roc_auc'], save_path)
        self.plot_class_distribution(labels, save_path)
        self.plot_prediction_confidence(probabilities, labels, predictions, save_path)
        
        # Save metrics
        metrics_path = save_path / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate classification report
        report = classification_report(labels, predictions, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        report_path = save_path / "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation results saved to {save_path}")
        
        return metrics

def main():
    """Main evaluation function."""
    # Configuration
    config = {
        "repo_name": "rlogh/lanternfly_swatter_training",
        "batch_size": 32,
        "image_size": 512,
        "model_path": "models/best_model.pth",
        "save_dir": "evaluation_results"
    }
    
    logger.info("Starting model evaluation...")
    logger.info(f"Configuration: {config}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load data
        logger.info("Loading test data...")
        data_loader = LanternflyDataLoader(
            repo_name=config["repo_name"],
            batch_size=config["batch_size"],
            image_size=config["image_size"]
        )
        
        _, _, test_loader = data_loader.get_dataloaders()
        
        # Create model
        logger.info("Loading model...")
        model = LanternflyClassifier(num_classes=2, pretrained=False)
        model_manager = ModelManager(model, device)
        
        # Load trained model
        checkpoint = model_manager.load_model(config["model_path"])
        
        # Create evaluator
        evaluator = ModelEvaluator(model_manager, test_loader)
        
        # Evaluate model
        metrics = evaluator.evaluate(save_dir=config["save_dir"])
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Final test accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()

