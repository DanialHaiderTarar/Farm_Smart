# Wheat Grain Classification System

A state-of-the-art computer vision system for classifying wheat grains as bad_seed, healthy_seed, or impurity using Vision Transformers (ViT).

## 🌾 Overview

This project provides a complete solution for automated wheat grain quality assessment:
- **Classification**: Vision Transformer-based classification of grain quality (bad_seed/healthy_seed/impurity)
- **API**: RESTful API built with FastAPI for easy integration
- **Features**: Advanced ViT feature extraction plus traditional image features
- **Preprocessing**: Enhanced image processing optimized for grain analysis
- **Batch Processing**: Efficient batch inference for multiple grains

## 🚀 Features

- ✅ Vision Transformer-based grain classification
- ✅ Three-class quality assessment (bad_seed, healthy_seed, impurity)
- ✅ Batch processing support for multiple grain images
- ✅ Dual feature extraction (ViT features + traditional image features)
- ✅ Quality scoring and grading (A-F)
- ✅ RESTful API with automatic documentation
- ✅ GPU acceleration support with automatic CPU fallback
- ✅ Enhanced preprocessing for optimal grain analysis

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (optional, for faster processing)
- 8GB+ RAM recommended
- 5GB+ free disk space

## 🛠️ Installation

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/wheat-classification-system.git
cd wheat-classification-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 📊 Data Preparation

### Dataset Structure
```
data/
├── train/
│   ├── images/          # Original annotated images
│   │   ├── image1.jpg
│   │   └── ...
│   └── labels/          # YOLO format labels (3 classes)
│       ├── image1.txt   # Class 0: bad_seed, 1: healthy_seed, 2: impurity
│       └── ...
└── classification/      # Processed individual grains (auto-generated)
    ├── train/
    │   ├── bad_seed/
    │   ├── healthy_seed/
    │   └── impurity/
    └── val/
        ├── bad_seed/
        ├── healthy_seed/
        └── impurity/
```

### Prepare Dataset for Training

1. **Extract individual grains from annotated images**
```bash
python scripts/prepare_dataset.py \
    --data-dir data/train/images \
    --labels-dir data/train/labels \
    --output-dir data/classification
```

This will:
- Extract individual grains from images using YOLO bounding boxes
- Enhance and preprocess grain images for ViT training
- Organize grains into class-specific folders
- Create balanced train/val/test splits
- Generate dataset statistics and visualizations

## 🏋️ Training

### Train Vision Transformer Classifier
```bash
python scripts/train_classifier.py \
    --model vit_base_patch16_224 \
    --data-dir data/classification \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

**Available Models:**
- `vit_base_patch16_224` (Recommended)
- `vit_small_patch16_224` (Faster)
- `efficientnet_b0` (Alternative)
- `convnext_tiny` (Modern CNN)

**Training Features:**
- Automatic mixed precision
- Learning rate scheduling
- Data augmentation
- Early stopping
- Model checkpointing
- Comprehensive metrics

## 🚀 Running the API

### Local Development
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access the API
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health
- **Model Info**: http://localhost:8000/api/v1/model-info

## 📡 API Usage

### Single Grain Classification

```python
import requests

# Classify a single grain image
with open('grain_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/classify-grain',
        files={'file': f},
        params={
            'extract_features': True,
            'enhance_image': True
        }
    )
    
result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Probabilities: {result['probabilities']}")
```

### Batch Processing

```python
# Process multiple grain images
files = [
    ('files', open('grain1.jpg', 'rb')),
    ('files', open('grain2.jpg', 'rb')),
    ('files', open('grain3.jpg', 'rb')),
]

response = requests.post(
    'http://localhost:8000/api/v1/classify-grains-batch',
    files=files,
    params={'extract_features': True}
)

results = response.json()
print(f"Processed {results['total_files']} grains")
print(f"Aggregate statistics: {results['aggregate_statistics']}")
```

### Feature Analysis

```python
# Extract detailed features from a grain
with open('grain_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/analyze-features',
        files={'file': f},
        params={'feature_type': 'both'}  # 'vit', 'traditional', or 'both'
    )
    
features = response.json()
print(f"ViT features length: {features['feature_analysis']['vit']['feature_vector_length']}")
print(f"Traditional features: {len(features['feature_analysis']['traditional']['features'])}")
```

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and status |
| `/api/v1/classify-grain` | POST | Classify a single grain image |
| `/api/v1/classify-grains-batch` | POST | Classify multiple grain images |
| `/api/v1/analyze-features` | POST | Extract detailed features from grain |
| `/api/v1/health` | GET | Health check with model status |
| `/api/v1/model-info` | GET | Get Vision Transformer model information |

## 🎯 Classification Results

### Response Format
```json
{
  "status": "success",
  "classification": "healthy_seed",
  "confidence": 0.94,
  "probabilities": {
    "bad_seed": 0.02,
    "healthy_seed": 0.94,
    "impurity": 0.04
  },
  "model_type": "Vision Transformer",
  "vit_features": [...],
  "traditional_features": {...}
}
```

### Quality Classes
- **healthy_seed**: Good quality grains suitable for consumption/planting
- **bad_seed**: Damaged, diseased, or poor quality grains
- **impurity**: Foreign particles, debris, or other contaminants

### Quality Grading (Batch Processing)
- **A**: ≥ 90% quality score (Excellent)
- **B**: ≥ 70% quality score (Good)
- **C**: ≥ 50% quality score (Average)
- **D**: ≥ 30% quality score (Below Average)
- **F**: < 30% quality score (Poor)

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | ViT model architecture | vit_base_patch16_224 |
| `USE_GPU` | Enable GPU acceleration | true |
| `BATCH_SIZE` | Processing batch size | 32 |
| `EXTRACT_FEATURES` | Extract ViT features | true |
| `IMAGE_SIZE` | Input image size | 224 |

### Model Configuration

Models are stored in the `models/` directory:
- `wheat_classifier_vit_*.pth`: Trained Vision Transformer model
- `training_results_*.json`: Training metrics and configuration

## 🧪 Testing

### Quick Test
```bash
# Test with sample images (place test images in a test folder)
curl -X POST "http://localhost:8000/api/v1/classify-grain" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_grain.jpg" \
     -F "extract_features=true"
```

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA not available**
   - System automatically falls back to CPU
   - To force CPU: set `USE_GPU=False` in `.env`

2. **Model files not found**
   - Ensure model is trained: `python scripts/train_classifier.py`
   - Check model path in `models/` directory

3. **Out of memory**
   - Reduce `BATCH_SIZE` in configuration
   - Use smaller model: `vit_small_patch16_224`

4. **Low accuracy**
   - Ensure proper data preparation
   - Increase training epochs
   - Use data augmentation
   - Check class balance in dataset

## 📈 Performance

### Benchmarks (ViT Base)
- **Classification**: ~20ms per grain (GPU) / ~50ms (CPU)
- **Feature Extraction**: ~25ms per grain (GPU) / ~60ms (CPU)
- **Batch Processing**: ~5ms per grain (GPU batch of 32)
- **API Overhead**: ~2-5ms

### Optimization Tips
- Use GPU for faster processing
- Process multiple grains in batches
- Use smaller ViT model for speed: `vit_small_patch16_224`
- Enable feature caching for repeated analysis

## 🧠 Model Architecture

### Vision Transformer Features
- **Self-Attention**: Captures global relationships in grain images
- **Patch-based Processing**: Analyzes grain texture at multiple scales
- **Transfer Learning**: Pre-trained on ImageNet, fine-tuned on grains
- **Feature Rich**: 768-dimensional feature vectors (ViT Base)

### Advantages over CNNs
- Better handling of grain texture variations
- Superior performance on fine-grained classification
- More interpretable attention patterns
- Robust to image orientation and scale
## python "F:\wheat-classification-system\scripts\test_wheat_classifier.py" --model-path "F:\wheat-classification-system\models\wheat_classifier_vit_small_patch16_224_20250620_095332.pth" --test-data-dir "F:\wheat-classification-system\data\classification\test" --batch-size 96 --img-size 224 --output-dir "F:\wheat-classification-system\test_results" --mixed-precision --num-workers 12
## 
## python scripts/prepare_dataset.py --data-dir data/train/images --labels-dir data/train/labels --output-dir data/classification
## python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
## ./test.sh
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with Vision Transformers for better wheat quality assessment