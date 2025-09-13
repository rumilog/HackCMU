"""
FastAPI application for serving the lantern fly classification model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lantern Fly Classification API",
    description="API for classifying images as dead lantern flies",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable (will be loaded on startup)
model = None

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    try:
        # TODO: Load your trained model here
        # model = load_model_from_file("models/lantern_fly_classifier.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # For now, we'll use a dummy model
        model = "dummy_model"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Classify an uploaded image as containing a dead lantern fly or not
    
    Args:
        file: Uploaded image file
        
    Returns:
        Classification result with confidence score
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to model input size (adjust as needed)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # TODO: Replace with actual model prediction
        # For now, return a dummy prediction
        prediction = predict_lantern_fly(image_array)
        
        return {
            "is_lantern_fly": prediction["is_lantern_fly"],
            "confidence_score": prediction["confidence_score"],
            "points_awarded": prediction["points_awarded"],
            "model_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

def predict_lantern_fly(image_array: np.ndarray) -> Dict[str, Any]:
    """
    Dummy prediction function - replace with actual model inference
    
    Args:
        image_array: Preprocessed image array
        
    Returns:
        Prediction result
    """
    # TODO: Replace with actual model prediction
    # This is a placeholder that returns random results
    import random
    
    confidence = random.uniform(0.3, 0.95)
    is_lantern_fly = confidence > 0.7
    
    points_awarded = 10 if is_lantern_fly else 1
    
    return {
        "is_lantern_fly": is_lantern_fly,
        "confidence_score": round(confidence, 4),
        "points_awarded": points_awarded
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "CNN Classifier",
        "version": "1.0.0",
        "input_size": [224, 224, 3],
        "classes": ["not_lantern_fly", "lantern_fly"],
        "status": "loaded" if model else "not_loaded"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
