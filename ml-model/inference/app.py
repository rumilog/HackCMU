#!/usr/bin/env python3
"""
FastAPI Inference Server for Lantern Fly Classification
Provides REST API endpoints for image classification
"""

import os
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional
import base64

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class ClassificationResponse(BaseModel):
    """Response model for classification results"""
    is_lantern_fly: bool
    confidence: float
    class_name: str
    class_probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str

class InferenceServer:
    """Main inference server class"""
    
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        self.class_names = ['Non-Lantern Fly', 'Lantern Fly']
        
        # Define image preprocessing
        self.transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Initialize model
            self.model = LanternFlyClassifier(num_classes=2, pretrained=False)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference"""
        try:
            # Convert to numpy array
            image_np = np.array(image.convert('RGB'))
            
            # Apply transforms
            transformed = self.transform(image=image_np)
            image_tensor = transformed['image']
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")
    
    def classify_image(self, image: Image.Image) -> ClassificationResponse:
        """Classify a single image"""
        if not self.model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get results
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            is_lantern_fly = predicted_class == 1
            
            # Get class probabilities
            class_probs = probabilities[0].cpu().numpy()
            class_probabilities = {
                self.class_names[i]: float(class_probs[i]) 
                for i in range(len(self.class_names))
            }
            
            return ClassificationResponse(
                is_lantern_fly=is_lantern_fly,
                confidence=confidence_score,
                class_name=self.class_names[predicted_class],
                class_probabilities=class_probabilities
            )
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Lantern Fly Classification API",
    description="API for classifying images as lantern flies or not",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference server
inference_server = InferenceServer()

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    # Try to load the best available model
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        if model_files:
            # Load the most recent model
            latest_model = max(model_files, key=os.path.getctime)
            inference_server.load_model(str(latest_model))
            logger.info(f"Loaded model: {latest_model}")
        else:
            logger.warning("No trained models found in models/ directory")
    else:
        logger.warning("Models directory not found")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=inference_server.model_loaded,
        device=inference_server.device
    )

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    """Classify an uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Classify image
        result = inference_server.classify_image(image)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

@app.post("/classify_base64", response_model=ClassificationResponse)
async def classify_base64_image(data: Dict[str, str]):
    """Classify an image from base64 string"""
    try:
        # Extract base64 data
        if 'image' not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Classify image
        result = inference_server.classify_image(image)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Base64 classification error: {e}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if not inference_server.model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_architecture": "EfficientNet-B0",
        "num_classes": 2,
        "class_names": inference_server.class_names,
        "device": inference_server.device,
        "input_size": [224, 224],
        "preprocessing": {
            "resize": [224, 224],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }

@app.post("/load_model")
async def load_model_endpoint(model_path: str):
    """Load a specific model"""
    try:
        inference_server.load_model(model_path)
        return {"message": f"Model loaded successfully from {model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Lantern Fly Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "classify": "/classify",
            "classify_base64": "/classify_base64",
            "model_info": "/model_info",
            "load_model": "/load_model",
            "docs": "/docs"
        }
    }

def main():
    """Run the inference server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lantern Fly Classification API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", help="Path to model file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Load model if specified
    if args.model:
        inference_server.load_model(args.model)
    
    # Run server
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()