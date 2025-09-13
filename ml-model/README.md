# 🦟 Lantern Fly Classification Model

A machine learning model for binary classification of spotted lantern flies (Lycorma delicatula) using EfficientNet-B0.

## 🎯 Overview

This project implements a binary classifier to identify dead/squashed lantern flies in images. The model uses transfer learning with EfficientNet-B0 and is trained on a dataset of 500+ lantern fly photos and 2000+ non-lantern fly images from the Insecta dataset.

### Classification Task
- **Input**: Image of a dead/squashed insect
- **Output**: Binary classification (Lantern Fly vs Non-Lantern Fly)
- **Model**: EfficientNet-B0 with custom classifier head
- **Framework**: PyTorch

## 📁 Project Structure

```
ml-model/
├── data/                          # Data directory
│   ├── raw/                       # Raw datasets
│   │   ├── insecta/              # Insecta dataset
│   │   └── insecta_organized/    # Organized Insecta images
│   └── processed/                # Processed training data
│       ├── train/                # Training data
│       │   ├── lantern_fly/      # Lantern fly images
│       │   └── non_lantern_fly/  # Non-lantern fly images
│       ├── val/                  # Validation data
│       └── test/                 # Test data
├── models/                        # Trained models
├── scripts/                       # Utility scripts
│   ├── data_processing.py        # Data processing and augmentation
│   ├── train_model.py            # Model training script
│   ├── download_insecta.py       # Insecta dataset downloader
│   └── photo_collection_guide.py # Photo collection guide
├── inference/                     # Inference API
│   └── app.py                    # FastAPI inference server
├── notebooks/                     # Jupyter notebooks
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create data directories
python scripts/data_processing.py
```

### 2. Download Insecta Dataset

```bash
# Download and organize Insecta dataset
python scripts/download_insecta.py
```

### 3. Collect Lantern Fly Photos

```bash
# Generate photo collection guide
python scripts/photo_collection_guide.py

# Follow the guide to collect 500 lantern fly photos
# Place them in: data/processed/train/lantern_fly/
```

### 4. Process Data

```bash
# Process and augment data
python scripts/data_processing.py
```

### 5. Train Model

```bash
# Train the classification model
python scripts/train_model.py
```

### 6. Run Inference API

```bash
# Start the inference server
python inference/app.py

# API will be available at: http://localhost:8000
# Documentation at: http://localhost:8000/docs
```

## 📊 Data Requirements

### Lantern Fly Photos (500 images)
- **Subject**: Dead/squashed spotted lantern flies
- **Quality**: Sharp focus, good lighting, clear visibility
- **Variety**: Different angles, conditions, backgrounds
- **Format**: JPG, PNG, or other common formats
- **Resolution**: Minimum 224x224 pixels

### Non-Lantern Fly Images (2000+ images)
- **Source**: Insecta dataset (Genius-Society/insecta)
- **Content**: Various insect species (excluding lantern flies)
- **Purpose**: Negative examples for binary classification

## 🔧 Model Architecture

### EfficientNet-B0 Base
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Size**: 224x224x3
- **Features**: 1280-dimensional feature vector

### Custom Classifier Head
```python
nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 2)  # Binary classification
)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 32
- **Epochs**: 50
- **Augmentation**: Rotation, flip, brightness, contrast, crop

## 📈 Training Process

### Data Augmentation
- **Rotation**: Random 90-degree rotations
- **Flip**: Horizontal and vertical flips
- **Brightness/Contrast**: Random adjustments
- **Hue/Saturation**: Color variations
- **Crop**: Random cropping with resize

### Training Strategy
1. **Data Splitting**: 80% train, 10% validation, 10% test
2. **Class Balancing**: Augment lantern fly images (4x factor)
3. **Transfer Learning**: Use pretrained EfficientNet-B0
4. **Fine-tuning**: Train custom classifier head
5. **Early Stopping**: Save best model based on validation accuracy

## 🚀 Inference API

### FastAPI Server
- **Endpoint**: `/classify` (POST)
- **Input**: Image file upload
- **Output**: Classification result with confidence

### Example Usage

```python
import requests

# Upload image for classification
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        files={'file': f}
    )

result = response.json()
print(f"Is Lantern Fly: {result['is_lantern_fly']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### API Endpoints
- `GET /health` - Health check
- `POST /classify` - Classify uploaded image
- `POST /classify_base64` - Classify base64 image
- `GET /model_info` - Model information
- `GET /docs` - API documentation

## 📋 Scripts Overview

### `data_processing.py`
- Data loading and preprocessing
- Image augmentation pipeline
- Train/validation/test splitting
- Class balancing

### `train_model.py`
- Model training and validation
- Training history tracking
- Model evaluation and metrics
- Model saving and loading

### `download_insecta.py`
- Download Insecta dataset from Hugging Face
- Organize and sample images
- Create training-ready dataset

### `photo_collection_guide.py`
- Generate comprehensive photo collection guide
- Quality requirements and standards
- Collection strategy and tips
- Progress tracking

## 🎯 Performance Metrics

### Target Performance
- **Accuracy**: >90% on test set
- **Precision**: >85% for lantern fly detection
- **Recall**: >85% for lantern fly detection
- **F1-Score**: >85% overall

### Evaluation
- **Test Set**: 10% of total data
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Visual performance analysis
- **Classification Report**: Detailed metrics per class

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set device for training
export CUDA_VISIBLE_DEVICES=0

# Optional: Set number of workers
export NUM_WORKERS=4
```

### Model Parameters
```python
# Training configuration
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
AUGMENTATION_FACTOR = 4

# Model configuration
NUM_CLASSES = 2
INPUT_SIZE = (224, 224)
PRETRAINED = True
```

## 📚 Dependencies

### Core ML Libraries
- `torch>=2.1.0` - PyTorch framework
- `torchvision>=0.16.0` - Computer vision utilities
- `tensorflow>=2.15.0` - TensorFlow (optional)
- `transformers>=4.35.0` - Hugging Face transformers

### Data Processing
- `albumentations>=1.3.1` - Image augmentation
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.1.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning utilities

### API and Utilities
- `fastapi>=0.104.0` - Web API framework
- `uvicorn>=0.24.0` - ASGI server
- `python-multipart>=0.0.6` - File upload support
- `requests>=2.31.0` - HTTP client

### Visualization
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `imageio>=2.31.0` - Image I/O

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Use CPU instead
device = "cpu"
```

#### 2. Data Loading Errors
```bash
# Check data paths
ls data/processed/train/lantern_fly/
ls data/processed/train/non_lantern_fly/

# Verify image formats
file data/processed/train/lantern_fly/*.jpg
```

#### 3. Model Loading Issues
```bash
# Check model file exists
ls models/*.pth

# Verify model architecture matches
python -c "import torch; print(torch.load('models/best_model.pth').keys())"
```

### Performance Optimization

#### 1. Faster Training
- Use GPU if available
- Increase batch size (if memory allows)
- Use mixed precision training
- Optimize data loading with more workers

#### 2. Better Accuracy
- Collect more diverse lantern fly photos
- Increase augmentation factor
- Train for more epochs
- Use data balancing techniques

## 📞 Support

### Getting Help
1. Check the troubleshooting section
2. Review the photo collection guide
3. Verify data requirements are met
4. Check model training logs

### Contributing
1. Follow the data collection guidelines
2. Test with your own lantern fly photos
3. Report issues and improvements
4. Share performance results

## 📄 License

This project is part of the Lantern Fly Tracker application. See the main project README for license information.

## 🎯 Next Steps

1. **Collect Data**: Follow the photo collection guide
2. **Train Model**: Run the training script
3. **Evaluate Performance**: Check test results
4. **Deploy API**: Start the inference server
5. **Integrate**: Connect with the main application

---

**Ready to start?** Run `python scripts/photo_collection_guide.py` to begin!
