"""
Image preprocessing service for wheat grain classification.
Optimized for Vision Transformer models.
"""

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Enhanced image preprocessing for grain classification."""
    
    def __init__(self, 
                 image_size: int = 224,
                 normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        Initialize image preprocessor.
        
        Args:
            image_size: Target image size for model input
            normalize_mean: ImageNet normalization means
            normalize_std: ImageNet normalization stds
        """
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Standard ViT preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        
        # Enhanced preprocessing pipeline
        self.enhanced_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        
    def preprocess_image(self, 
                        image: Union[np.ndarray, Image.Image], 
                        enhance: bool = True) -> torch.Tensor:
        """
        Preprocess single image for model inference.
        
        Args:
            image: Input image (PIL Image or numpy array)
            enhance: Whether to apply enhancement
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if image.dtype == np.uint8:
                    image = Image.fromarray(image)
                else:
                    image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply enhancement if requested
            if enhance:
                image = self.enhance_grain_image(image)
            
            # Apply preprocessing transform
            tensor = self.transform(image)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def preprocess_batch(self, 
                        images: list, 
                        enhance: bool = True) -> torch.Tensor:
        """
        Preprocess batch of images.
        
        Args:
            images: List of images (PIL Images or numpy arrays)
            enhance: Whether to apply enhancement
            
        Returns:
            Batch tensor of preprocessed images
        """
        try:
            processed_images = []
            
            for image in images:
                tensor = self.preprocess_image(image, enhance=enhance)
                processed_images.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(processed_images)
            
            return batch_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing batch: {str(e)}")
            raise ValueError(f"Failed to preprocess batch: {str(e)}")
    
    def enhance_grain_image(self, image: Image.Image) -> Image.Image:
        """
        Apply enhancement techniques optimized for grain images.
        
        Args:
            image: PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert to numpy for OpenCV processing
            img_array = np.array(image)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            img_array = self._apply_clahe(img_array)
            
            # Noise reduction
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
            
            # Convert back to PIL
            enhanced_image = Image.fromarray(img_array)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(1.2)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(1.1)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Enhancement failed, using original image: {str(e)}")
            return image
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to improve local contrast.
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image array
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"CLAHE failed: {str(e)}")
            return image
    
    def extract_grain_roi(self, 
                         image: np.ndarray, 
                         bbox: Tuple[int, int, int, int],
                         padding: int = 10) -> np.ndarray:
        """
        Extract grain region of interest from image.
        
        Args:
            image: Full image array
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Additional padding around bbox
            
        Returns:
            Cropped grain image
        """
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract ROI
            roi = image[y1:y2, x1:x2]
            
            return roi
            
        except Exception as e:
            logger.error(f"Error extracting ROI: {str(e)}")
            raise ValueError(f"Failed to extract grain ROI: {str(e)}")
    
    def resize_with_aspect_ratio(self, 
                               image: Image.Image, 
                               target_size: int,
                               fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Resize image maintaining aspect ratio with padding.
        
        Args:
            image: Input PIL Image
            target_size: Target size (square)
            fill_color: Padding color
            
        Returns:
            Resized image with padding
        """
        try:
            # Calculate resize dimensions
            w, h = image.size
            max_dim = max(w, h)
            
            if max_dim <= target_size:
                # Image is smaller, just center it
                new_image = Image.new('RGB', (target_size, target_size), fill_color)
                paste_x = (target_size - w) // 2
                paste_y = (target_size - h) // 2
                new_image.paste(image, (paste_x, paste_y))
                return new_image
            
            # Calculate scale factor
            scale = target_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Create new image with padding
            new_image = Image.new('RGB', (target_size, target_size), fill_color)
            paste_x = (target_size - new_w) // 2
            paste_y = (target_size - new_h) // 2
            new_image.paste(resized, (paste_x, paste_y))
            
            return new_image
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            # Fallback to simple resize
            return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    def create_augmentation_pipeline(self, training: bool = True) -> transforms.Compose:
        """
        Create data augmentation pipeline for training.
        
        Args:
            training: Whether this is for training (enables augmentation)
            
        Returns:
            Augmentation transform pipeline
        """
        if training:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), 
                                interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        else:
            return self.transform
    
    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor for visualization.
        
        Args:
            tensor: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.normalize_std).view(3, 1, 1)
        
        denormalized = tensor * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized