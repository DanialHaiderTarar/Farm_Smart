"""
Utility functions for the wheat grain classification API.
"""

import os
import json
import logging
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import torch


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/wheat_classification_api.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def validate_image_file(file_content: bytes, filename: str) -> bool:
    """
    Validate if the uploaded file is a valid image.
    
    Args:
        file_content: File content bytes
        filename: Original filename
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in valid_extensions:
            return False
        
        # Try to open with PIL
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Verify it's a valid image
        
        return True
        
    except Exception:
        return False
def load_image_from_bytes(file_content: bytes) -> Image.Image:
    """
    Load PIL Image from bytes content.
    
    Args:
        file_content: Image file content as bytes
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        image = Image.open(io.BytesIO(file_content))
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from bytes: {str(e)}")

def get_file_hash(file_content: bytes) -> str:
    """
    Generate hash for file content.
    
    Args:
        file_content: File content bytes
        
    Returns:
        SHA256 hash of the file
    """
    return hashlib.sha256(file_content).hexdigest()


def save_results(results: Dict[str, Any], 
                filename: Optional[str] = None,
                results_dir: str = "results") -> str:
    """
    Save classification results to JSON file.
    
    Args:
        results: Results dictionary
        filename: Optional filename (auto-generated if None)
        results_dir: Directory to save results
        
    Returns:
        Path to saved file
    """
    # Create results directory if it doesn't exist
    Path(results_dir).mkdir(exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_id = generate_request_id()
        filename = f"{timestamp}_{request_id}.json"
    
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    filepath = Path(results_dir) / filename
    
    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'filename': filename,
        'api_version': '2.0.0'
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return str(filepath)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load results from {filepath}: {str(e)}")


def generate_request_id() -> str:
    """
    Generate unique request ID.
    
    Returns:
        Unique request ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    return f"req_{timestamp}_{random_suffix}"


def calculate_quality_score(probabilities: Dict[str, float]) -> float:
    """
    Calculate overall quality score based on class probabilities.
    
    Args:
        probabilities: Class probabilities dictionary
        
    Returns:
        Quality score between 0 and 1
    """
    # Weight the classes: healthy_seed=1.0, bad_seed=0.3, impurity=0.0
    weights = {
        'healthy_seed': 1.0,
        'bad_seed': 0.3,
        'impurity': 0.0
    }
    
    score = 0.0
    for class_name, prob in probabilities.items():
        if class_name in weights:
            score += prob * weights[class_name]
    
    return score


def get_quality_grade(score: float) -> str:
    """
    Convert quality score to letter grade.
    
    Args:
        score: Quality score (0-1)
        
    Returns:
        Quality grade (A-F)
    """
    if score >= 0.9:
        return 'A'
    elif score >= 0.7:
        return 'B'
    elif score >= 0.5:
        return 'C'
    elif score >= 0.3:
        return 'D'
    else:
        return 'F'


def calculate_batch_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate statistics for batch processing.
    
    Args:
        results: List of individual classification results
        
    Returns:
        Aggregate statistics dictionary
    """
    if not results:
        return {}
    
    # Count classifications
    class_counts = {'healthy_seed': 0, 'bad_seed': 0, 'impurity': 0}
    confidences = []
    quality_scores = []
    
    for result in results:
        classification = result.get('classification', '')
        confidence = result.get('confidence', 0.0)
        probabilities = result.get('probabilities', {})
        
        if classification in class_counts:
            class_counts[classification] += 1
        
        confidences.append(confidence)
        
        # Calculate quality score for this grain
        quality_score = calculate_quality_score(probabilities)
        quality_scores.append(quality_score)
    
    total_grains = len(results)
    avg_confidence = np.mean(confidences) if confidences else 0.0
    avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
    
    # Calculate percentages
    class_percentages = {
        class_name: (count / total_grains) * 100 if total_grains > 0 else 0.0
        for class_name, count in class_counts.items()
    }
    
    return {
        'total_grains': total_grains,
        'class_counts': class_counts,
        'class_percentages': class_percentages,
        'average_confidence': round(avg_confidence, 3),
        'average_quality_score': round(avg_quality_score, 3),
        'overall_grade': get_quality_grade(avg_quality_score),
        'confidence_std': round(np.std(confidences), 3) if len(confidences) > 1 else 0.0,
        'quality_distribution': {
            'excellent': sum(1 for s in quality_scores if s >= 0.9),
            'good': sum(1 for s in quality_scores if 0.7 <= s < 0.9),
            'average': sum(1 for s in quality_scores if 0.5 <= s < 0.7),
            'poor': sum(1 for s in quality_scores if s < 0.5)
        }
    }


def cleanup_temp_files(temp_dir: str = "uploads/temp", max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age.
    
    Args:
        temp_dir: Temporary files directory
        max_age_hours: Maximum age in hours before deletion
        
    Returns:
        Number of files deleted
    """
    try:
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return 0
        
        deleted_count = 0
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in temp_path.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1
        
        return deleted_count
        
    except Exception as e:
        logging.error(f"Error cleaning up temp files: {str(e)}")
        return 0


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model information dictionary
    """
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get model type
        model_type = type(model).__name__
        
        # Try to get model config if available
        config = {}
        if hasattr(model, 'config'):
            config = {
                'num_classes': getattr(model.config, 'num_labels', None),
                'hidden_size': getattr(model.config, 'hidden_size', None),
                'num_attention_heads': getattr(model.config, 'num_attention_heads', None),
                'num_hidden_layers': getattr(model.config, 'num_hidden_layers', None)
            }
        
        return {
            'model_type': model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(total_params * 4 / (1024 * 1024), 2),  # Assuming float32
            'config': config
        }
        
    except Exception as e:
        return {'error': f"Failed to get model info: {str(e)}"}


def create_error_response(error_message: str, 
                         error_code: str = "PROCESSING_ERROR",
                         details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        error_message: Human readable error message
        error_code: Error code for programmatic handling
        details: Additional error details
        
    Returns:
        Standardized error response
    """
    response = {
        'status': 'error',
        'error': {
            'code': error_code,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    if details:
        response['error']['details'] = details
    
    return response


def validate_model_path(model_path: str) -> bool:
    """
    Validate if model path exists and is accessible.
    
    Args:
        model_path: Path to model file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(model_path)
        return path.exists() and path.is_file() and path.suffix in ['.pth', '.pt']
    except Exception:
        return False


def ensure_directory(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to ensure
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Numpy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    return tensor.numpy()


def confidence_to_category(confidence: float) -> str:
    """
    Convert confidence score to categorical description.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Confidence category
    """
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    elif confidence >= 0.3:
        return "Low"
    else:
        return "Very Low"


# Import io at the top if not already imported
import io