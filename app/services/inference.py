"""
Inference service for wheat grain classification using Vision Transformer
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Union, Tuple
import torch
import logging
from pathlib import Path

from app.models.wheat_classifier import WheatGrainClassifier
from app.services.preprocessing import ImagePreprocessor
from app.config import settings

logger = logging.getLogger(__name__)

class WheatInferenceService:
    """Vision Transformer-based inference service for wheat grain classification"""
    
    def __init__(self):
        """Initialize the inference service with ViT classifier"""
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu")
        logger.info(f"Initializing ViT inference service on device: {self.device}")
        
        # Initialize components
        self._initialize_classifier()
        self.preprocessor = ImagePreprocessor()
        
        # Cache for model warmup
        self._warmed_up = False
    
    def _initialize_classifier(self):
        """Initialize the Vision Transformer classifier"""
        try:
            self.classifier = WheatGrainClassifier(
                model_path=str(settings.CLASSIFIER_MODEL_PATH),
                model_name=settings.MODEL_NAME,
                num_classes=settings.NUM_CLASSES,
                device=self.device,
                input_size=settings.IMAGE_SIZE
            )
            logger.info("ViT classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ViT classifier: {e}")
            self.classifier = None
    
    def warmup(self):
        """Warm up model with dummy data"""
        if self._warmed_up or not self.classifier:
            return
        
        logger.info("Warming up Vision Transformer...")
        
        try:
            # Create dummy grain image
            dummy_grain = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Warm up classifier
            self.classifier.predict_single_grain(dummy_grain)
            
            self._warmed_up = True
            logger.info("ViT warmup completed")
            
        except Exception as e:
            logger.error(f"Error during warmup: {e}")
    
    def check_models_loaded(self) -> bool:
        """Check if classifier is loaded successfully"""
        return self.classifier is not None
    
    def classify_single_grain(
        self, 
        grain_image: np.ndarray,
        extract_features: bool = True,
        enhance: bool = True
    ) -> Dict:
        """
        Classify a single grain image using Vision Transformer
        
        Args:
            grain_image: Input grain image
            extract_features: Whether to extract ViT features
            enhance: Whether to enhance image
            
        Returns:
            Classification result with ViT predictions
        """
        # Warm up on first use
        if not self._warmed_up:
            self.warmup()
        
        if not self.classifier:
            raise RuntimeError("ViT classifier not properly loaded")
        
        # Enhance if requested
        if enhance:
            grain_image = self.preprocessor.enhance_grain_image(grain_image)
        
        # Classify using Vision Transformer
        classification = self.classifier.predict_single_grain(
            grain_image, 
            return_features=extract_features
        )
        
        result = {
            'classification': classification['class_name'],
            'confidence': classification['confidence'],
            'probabilities': classification['probability_dict'],
            'model_type': 'Vision Transformer'
        }
        
        # Add ViT features if extracted
        if extract_features and 'features' in classification:
            result['vit_features'] = classification['features']
            result['feature_dimension'] = len(classification['features'])
        
        # Add traditional features for comparison
        if extract_features:
            traditional_features = {}  # Empty dict if method doesn't exist
            result['traditional_features'] = traditional_features
        
        return result
    
    def process_image_for_classification(
        self, 
        image: Union[bytes, np.ndarray],
        grain_regions: List[Dict],
        extract_features: bool = True,
        enhance_grains: bool = True
    ) -> Dict:
        """
        Process image with pre-detected grain regions for classification
        
        Args:
            image: Input image
            grain_regions: List of grain bounding boxes and metadata
            extract_features: Whether to extract features
            enhance_grains: Whether to enhance grain images
            
        Returns:
            Classification results for all grains
        """
        # Warm up on first use
        if not self._warmed_up:
            self.warmup()
        
        if not self.classifier:
            raise RuntimeError("ViT classifier not properly loaded")
        
        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(image)
        
        # Extract and classify each grain
        results = []
        quality_counts = {'healthy_seed': 0, 'bad_seed': 0, 'impurity': 0}
        
        for grain_data in grain_regions:
            # Extract grain region
            bbox = grain_data['bbox']
            x1, y1, x2, y2 = bbox
            grain_image = processed_image[y1:y2, x1:x2]
            
            if grain_image.size == 0:
                continue
            
            # Classify grain
            classification_result = self.classify_single_grain(
                grain_image, 
                extract_features=extract_features,
                enhance=enhance_grains
            )
            
            # Compile grain result
            grain_result = {
                'grain_id': grain_data.get('grain_id', len(results) + 1),
                'bbox': bbox,
                'classification': classification_result['classification'],
                'confidence': classification_result['confidence'],
                'probabilities': classification_result['probabilities'],
                'model_type': 'Vision Transformer',
                'area': (x2 - x1) * (y2 - y1),
                'center': [(x1 + x2) // 2, (y1 + y2) // 2]
            }
            
            # Add features if extracted
            if extract_features:
                if 'vit_features' in classification_result:
                    grain_result['vit_features'] = classification_result['vit_features']
                if 'traditional_features' in classification_result:
                    grain_result['traditional_features'] = classification_result['traditional_features']
            
            results.append(grain_result)
            quality_counts[classification_result['classification']] += 1
        
        # Calculate overall metrics
        total_grains = len(results)
        quality_score = self._calculate_quality_score(quality_counts)
        quality_grade = self._get_quality_grade(quality_score)
        
        return {
            'grains': results,
            'summary': {
                'total': total_grains,
                'healthy_seed': quality_counts['healthy_seed'],
                'bad_seed': quality_counts['bad_seed'],
                'impurity': quality_counts['impurity']
            },
            'quality_score': quality_score,
            'quality_grade': quality_grade,
            'model_info': {
                'classifier': 'Vision Transformer',
                'model_name': settings.MODEL_NAME,
                'feature_extraction': extract_features
            },
            'image_dimensions': {
                'width': processed_image.shape[1],
                'height': processed_image.shape[0]
            }
        }
    
    def batch_classify_grains(
        self,
        grain_images: List[np.ndarray],
        extract_features: bool = True,
        enhance_grains: bool = True
    ) -> List[Dict]:
        """Classify multiple grain images using ViT batch processing"""
        if not self.classifier:
            raise RuntimeError("ViT classifier not properly loaded")
        
        # Warm up on first use
        if not self._warmed_up:
            self.warmup()
        
        # Enhance grains if requested
        if enhance_grains:
            enhanced_grains = [
                self.preprocessor.enhance_grain_image(grain) 
                for grain in grain_images
            ]
        else:
            enhanced_grains = grain_images
        
        # Batch classify using ViT
        batch_results = self.classifier.predict_batch(
            enhanced_grains,
            batch_size=settings.BATCH_SIZE,
            return_features=extract_features
        )
        
        # Format results
        formatted_results = []
        for i, classification in enumerate(batch_results):
            result = {
                'grain_id': i + 1,
                'classification': classification['class_name'],
                'confidence': classification['confidence'],
                'probabilities': classification['probability_dict'],
                'model_type': 'Vision Transformer'
            }
            
            if extract_features and 'features' in classification:
                result['vit_features'] = classification['features']
                # Add traditional features
                traditional_features = {}
                result['traditional_features'] = traditional_features
            
            formatted_results.append(result)
        
        return formatted_results
    
    def _calculate_quality_score(self, quality_counts: Dict[str, int]) -> float:
        """Calculate overall quality score based on grain distribution"""
        total = sum(quality_counts.values())
        if total == 0:
            return 0.0
        
        score = 0.0
        for quality, count in quality_counts.items():
            weight = settings.QUALITY_WEIGHTS.get(quality, 0.0)
            score += (count * weight) / total
        
        return round(score, 3)
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        for grade, threshold in settings.QUALITY_GRADES.items():
            if score >= threshold:
                return grade
        return 'F'
    
    def get_model_info(self) -> Dict:
        """Get information about the ViT model"""
        if not self.classifier:
            return {'error': 'Classifier not loaded'}
        
        return self.classifier.get_model_info()
    
    def create_visualization(
        self, 
        image: Union[bytes, np.ndarray],
        results: Dict,
        show_confidence: bool = True,
        show_probabilities: bool = False
    ) -> np.ndarray:
        """
        Create visualization with ViT predictions
        
        Args:
            image: Original image
            results: ViT classification results
            show_confidence: Whether to show confidence scores
            show_probabilities: Whether to show all class probabilities
            
        Returns:
            Annotated image with ViT predictions
        """
        # Preprocess image
        vis_image = self.preprocessor.preprocess_image(image)
        
        # Color mapping
        color_map = {
            'healthy_seed': (0, 255, 0),     # Green
            'bad_seed': (128, 0, 128),       # Purple
            'impurity': (255, 255, 0)        # Yellow
        }
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
       
       # Draw grain annotations
        for grain in results['grains']:
            x1, y1, x2, y2 = grain['bbox']
            classification = grain['classification']
            confidence = grain['confidence']
           
            # Get color
            color = color_map.get(classification, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label_parts = [f"#{grain['grain_id']}: {classification}"]
            
            if show_confidence:
                label_parts.append(f"({confidence:.2f})")
            
            label = " ".join(label_parts)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
            
            cv2.rectangle(
                vis_image,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 5, label_y + 5),
                color,
                -1
            )
            
            # Draw label
            cv2.putText(
                vis_image,
                label,
                (x1 + 2, label_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness - 1
            )
            
            # Show probabilities if requested
            if show_probabilities and 'probabilities' in grain:
                prob_y_offset = 25
                for class_name, prob in grain['probabilities'].items():
                    prob_text = f"{class_name}: {prob:.2f}"
                    cv2.putText(
                        vis_image,
                        prob_text,
                        (x1, y1 + prob_y_offset),
                        font,
                        0.4,
                        color,
                        1
                    )
                    prob_y_offset += 15
        
        # Add summary information
        self._add_vit_summary_overlay(vis_image, results)
        
        return vis_image
   
    def _add_vit_summary_overlay(self, image: np.ndarray, results: Dict):
        """Add ViT-specific summary statistics overlay"""
        summary = results['summary']
        h, w = image.shape[:2]
       
        # Create overlay background
        overlay_h = 140
        overlay = np.zeros((overlay_h, w, 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)  # Black background
       
        # Add to top of image
        image[0:overlay_h, :] = cv2.addWeighted(image[0:overlay_h, :], 0.3, overlay, 0.7, 0)
       
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)
       
        # Title
        cv2.putText(image, "Wheat Grain Analysis (Vision Transformer)", (10, 30), font, 1.0, color, 2)
       
        # Model info
        model_info = results.get('model_info', {})
        model_text = f"Model: {model_info.get('model_name', 'ViT')}"
        cv2.putText(image, model_text, (10, 50), font, 0.6, (200, 200, 200), 1)
       
        # Statistics
        y_offset = 80
        cv2.putText(image, f"Total Grains: {summary['total']}", (10, y_offset), font, font_scale, color, 1)
        cv2.putText(image, f"Healthy: {summary['healthy_seed']}", (200, y_offset), font, font_scale, (0, 255, 0), 1)
        cv2.putText(image, f"Bad: {summary['bad_seed']}", (350, y_offset), font, font_scale, (128, 0, 128), 1)
        cv2.putText(image, f"Impurity: {summary['impurity']}", (450, y_offset), font, font_scale, (255, 255, 0), 1)
       
        # Quality score and grade
        y_offset = 110
        cv2.putText(image, f"Quality Score: {results['quality_score']:.2f}", (10, y_offset), font, font_scale, color, 1)
        cv2.putText(image, f"Grade: {results['quality_grade']}", (250, y_offset), font, 1.0, color, 2)