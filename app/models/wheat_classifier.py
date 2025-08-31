"""
Vision Transformer-based wheat grain classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
import cv2

logger = logging.getLogger(__name__)

class WheatGrainClassifier:
    """Vision Transformer-based classifier for wheat grain quality assessment"""
    
    def __init__(
        self, 
        model_path: str, 
        model_name: str = 'vit_small_patch16_224',
        num_classes: int = 3, 
        device: Optional[torch.device] = None,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the wheat grain classifier
        
        Args:
            model_path: Path to saved model weights
            model_name: Model architecture name
            num_classes: Number of classification classes
            device: Torch device for computation
            input_size: Input image size
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.class_names = ['bad_seed', 'healthy_seed', 'impurity']
        
        # Define image transforms
        self.transform = self._get_transforms()
        
        # Build and load model
        self.model = self._build_model()
        self._load_weights()
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info(f"ViT Classifier initialized on device: {self.device}")
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_model(self) -> nn.Module:
        """Build the Vision Transformer model"""
        try:
            # Create model using timm
            model = timm.create_model(
                self.model_name, 
                pretrained=False,  # We'll load our trained weights
                num_classes=self.num_classes
            )
            
            logger.info(f"Created {self.model_name} with {self.num_classes} classes")
            
        except Exception as e:
            logger.error(f"Error creating model {self.model_name}: {e}")
            # Fallback to ViT base
            model = timm.create_model(
                'vit_small_patch16_224', 
                pretrained=False, 
                num_classes=self.num_classes
            )
            logger.info("Using fallback ViT small model")
        
        return model.to(self.device)
    
    def _load_weights(self):
        """Load pretrained weights if available"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(
                    self.model_path, 
                    map_location=self.device
                )
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded model weights from {self.model_path}")
                    
                    # Load additional info if available
                    if 'class_to_idx' in checkpoint:
                        self.class_to_idx = checkpoint['class_to_idx']
                    if 'config' in checkpoint:
                        self.config = checkpoint['config']
                        if 'class_names' in self.config:
                            self.class_names = self.config['class_names']
                            
                else:
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"Loaded model weights from {self.model_path}")
                    
            except Exception as e:
                logger.warning(f"Could not load model weights: {e}")
                logger.info("Using randomly initialized weights")
        else:
            logger.warning(f"Model file not found at {self.model_path}")
            logger.info("Using randomly initialized weights")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Preprocessed tensor
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Ensure RGB format
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            
            # Convert BGR to RGB if needed (OpenCV format)
            if image.shape[2] == 3:
                # Simple heuristic to detect BGR
                if np.mean(image[:, :, 0]) > np.mean(image[:, :, 2]) * 1.1:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(image.astype('uint8'))
        
        # Apply transforms
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def extract_features(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract feature vector from image using the model
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        # Get features from the model (before final classification layer)
        self.model.eval()
        
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Extract features
        with torch.no_grad():
            if hasattr(self.model, 'forward_features'):
                # For ViT models
                features = self.model.forward_features(img_tensor)
                # Global average pooling if needed
                if len(features.shape) == 3:  # [batch, seq_len, embed_dim]
                    features = features.mean(dim=1)  # Average over sequence
            else:
                # For other models, use hook or forward until penultimate layer
                # This is a simplified approach
                features = self.model(img_tensor)
            
            features = features.squeeze().cpu().numpy()
        
        return features
    
    def predict_single_grain(
        self, 
        grain_image: Union[np.ndarray, Image.Image],
        return_features: bool = False
    ) -> Dict:
        """
        Predict class for a single grain image
        
        Args:
            grain_image: Input grain image
            return_features: Whether to return feature vector
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_tensor = self.preprocess_image(grain_image)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = predicted_idx.item()
            confidence_score = confidence.item()
            
            # Get all class probabilities
            probs = probabilities[0].cpu().numpy()
        
        result = {
            'class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence_score,
            'probabilities': probs.tolist(),
            'probability_dict': {
                self.class_names[i]: float(probs[i]) 
                for i in range(self.num_classes)
            }
        }
        
        # Add features if requested
        if return_features:
            features = self.extract_features(grain_image)
            result['features'] = features.tolist()
        
        return result
    
    def predict_batch(
        self, 
        grain_images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 32,
        return_features: bool = False
    ) -> List[Dict]:
        """
        Predict classes for multiple grain images
        
        Args:
            grain_images: List of grain images
            batch_size: Batch size for processing
            return_features: Whether to return feature vectors
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(grain_images), batch_size):
            batch = grain_images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            # Stack tensors
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get predictions for each image
                for j in range(len(batch)):
                    probs = probabilities[j].cpu().numpy()
                    predicted_class = np.argmax(probs)
                    confidence = probs[predicted_class]
                    
                    result = {
                        'class': int(predicted_class),
                        'class_name': self.class_names[predicted_class],
                        'confidence': float(confidence),
                        'probabilities': probs.tolist(),
                        'probability_dict': {
                            self.class_names[k]: float(probs[k]) 
                            for k in range(self.num_classes)
                        }
                    }
                    
                    # Add features if requested
                    if return_features:
                        features = self.extract_features(batch[j])
                        result['features'] = features.tolist()
                    
                    results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Vision Transformer',
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'input_size': self.input_size,
            'device': str(self.device),
            'model_path': str(self.model_path),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': self._get_architecture_info()
        }
    
    def _get_architecture_info(self) -> Dict:
        """Get detailed architecture information"""
        if hasattr(self.model, 'patch_embed'):
            # ViT model
            return {
                'patch_size': getattr(self.model.patch_embed, 'patch_size', None),
                'embed_dim': getattr(self.model, 'embed_dim', None),
                'num_heads': getattr(self.model, 'num_heads', None),
                'depth': getattr(self.model, 'depth', None)
            }
        else:
            return {'type': 'Custom architecture'}

# Enhanced ViT model with additional features
class EnhancedWheatViT(nn.Module):
    """Enhanced Vision Transformer for wheat grain classification"""
    
    def __init__(self, small_model_name='vit_small_patch16_224', num_classes=3, dropout=0.1):
        super(EnhancedWheatViT, self).__init__()
        
        # Load base ViT model
        self.backbone = timm.create_model(small_model_name, pretrained=True, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim // 2),
            nn.Dropout(dropout / 2),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # Attention visualization support
        self.attention_weights = None
        
    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def get_attention_weights(self):
        """Get attention weights for visualization"""
        return self.attention_weights