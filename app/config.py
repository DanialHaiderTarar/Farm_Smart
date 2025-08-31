"""
Configuration settings for the Wheat Grain Classification System
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List, Tuple

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    APP_NAME: str = "Wheat Grain Classification System"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Vision Transformer-based wheat grain quality classification"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Model Settings - Updated for new pipeline
    CLASSIFIER_MODEL_PATH: str = "models/wheat_classifier_vit.pth"
    
    # Classification Settings - Updated for your dataset
    CLASSES: List[str] = ["bad_seed", "healthy_seed", "impurity"]
    NUM_CLASSES: int = 3
    CLASS_COLORS: dict = {
        "bad_seed": (128, 0, 128),      # Purple
        "healthy_seed": (0, 255, 0),    # Green
        "impurity": (255, 255, 0)       # Yellow
    }
    
    # Model Architecture Settings
    MODEL_NAME: str = "vit_small_patch16_224"  # Vision Transformer
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    PRETRAINED: bool = True
    
    # Alternative models (can be changed via config)
    AVAILABLE_MODELS: List[str] = [
        "vit_small_patch16_224",
        "vit_small_patch16_224", 
        "efficientnet_b0",
        "efficientnet_b3",
        "convnext_tiny",
        "convnext_small"
    ]
    
    # Training Settings
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.01
    
    # Data Augmentation
    TRAIN_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.5
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "classification"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    TEMP_DIR: Path = BASE_DIR / "temp"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    RESULTS_DIR: Path = BASE_DIR / "results"
    
    # File Settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    
    # Processing Settings
    USE_GPU: bool = True
    GPU_DEVICE: int = 0
    NUM_WORKERS: int = 4
    
    # Feature Extraction
    EXTRACT_FEATURES: bool = True
    FEATURE_VECTOR_SIZE: int = 768  # ViT base feature size
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Quality Score Weights
    QUALITY_WEIGHTS: dict = {
        "healthy_seed": 1.0,
        "bad_seed": 0.3,
        "impurity": 0.0
    }
    
    # Quality Grade Thresholds
    QUALITY_GRADES: dict = {
        "A": 0.9,
        "B": 0.7,
        "C": 0.5,
        "D": 0.3,
        "F": 0.0
    }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Create necessary directories
for directory in [settings.UPLOAD_DIR, settings.TEMP_DIR, settings.MODELS_DIR, 
                 settings.LOGS_DIR, settings.RESULTS_DIR, settings.PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)