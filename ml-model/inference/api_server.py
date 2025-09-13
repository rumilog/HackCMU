#!/usr/bin/env python3
"""
Flask API server for lanternfly classification.
Provides HTTP endpoints for the Node.js backend to call.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from pathlib import Path
import tempfile
import io
from PIL import Image

from lanternfly_classifier import get_inference_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the inference service
try:
    inference_service = get_inference_service()
    logger.info("Inference service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize inference service: {e}")
    inference_service = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'lanternfly-classifier',
        'model_loaded': inference_service is not None
    })

@app.route('/classify', methods=['POST'])
def classify_image():
    """
    Classify an uploaded image.
    Expects a multipart/form-data request with an 'image' field.
    """
    try:
        if not inference_service:
            return jsonify({
                'success': False,
                'error': 'Inference service not available'
            }), 500
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file selected'
            }), 400
        
        # Read image data
        image_data = image_file.read()
        
        # Validate that it's an image
        try:
            Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            }), 400
        
        # Classify the image
        result = inference_service.classify_image_from_bytes(image_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({
            'success': False,
            'error': f'Classification failed: {str(e)}'
        }), 500

@app.route('/classify-file', methods=['POST'])
def classify_image_file():
    """
    Classify an image from a file path.
    Expects JSON with 'image_path' field.
    """
    try:
        if not inference_service:
            return jsonify({
                'success': False,
                'error': 'Inference service not available'
            }), 500
        
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            return jsonify({
                'success': False,
                'error': 'No image_path provided'
            }), 400
        
        image_path = data['image_path']
        
        if not os.path.exists(image_path):
            return jsonify({
                'success': False,
                'error': f'Image file not found: {image_path}'
            }), 404
        
        # Classify the image
        result = inference_service.classify_image(image_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({
            'success': False,
            'error': f'Classification failed: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 10MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5001))
    
    logger.info(f"Starting lanternfly classification API server on port {port}")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Set to False for production
        threaded=True
    )
