"""
Vision Transformer-based Classification API endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import numpy as np
from datetime import datetime
import uuid
import io
import cv2
from PIL import Image
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback

from app.services.inference import WheatInferenceService
from app.config import settings
from app.utils.helpers import validate_image_file, save_results, load_image_from_bytes

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["ViT Classification"])

# Initialize ViT inference service - Add error handling
try:
    inference_service = WheatInferenceService()
    logger.info("ViT inference service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ViT inference service: {e}", exc_info=True)
    inference_service = None

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)

@router.post("/classify-grain", response_model=dict)
async def classify_single_grain(
    file: UploadFile = File(..., description="Single grain image file"),
    extract_features: bool = Query(True, description="Extract ViT features"),
    enhance_image: bool = Query(True, description="Apply image enhancement"),
    return_probabilities: bool = Query(True, description="Return class probabilities")
):
    """
    Classify a single grain image using Vision Transformer
    
    - **file**: Single grain image file (JPEG, PNG, etc.)
    - **extract_features**: Extract ViT feature vectors
    - **enhance_image**: Apply preprocessing enhancement
    - **return_probabilities**: Include class probabilities
    
    Returns ViT-based classification with confidence scores
    """
    # Check if inference service is available
    if inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service not available. Please check model loading."
        )
    
    # Validate file
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        if not validate_image_file(contents, file.filename):
            raise HTTPException(status_code=400, detail="Invalid image format")
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file validation: {e}")
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    
    try:
        # Load image with error handling
        try:
            image = load_image_from_bytes(contents)
        except Exception as e:
            logger.error(f"Failed to load image from bytes: {e}")
            raise HTTPException(status_code=400, detail="Could not process image file")
        
        logger.info(f"Processing ViT classification request {request_id} for file: {file.filename}")
        
        # Process with ViT in thread pool with timeout
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    inference_service.classify_single_grain,
                    image,
                    extract_features,
                    enhance_image
                ),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            raise HTTPException(status_code=408, detail="Request timeout")
        except Exception as e:
            logger.error(f"Error in inference service: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
        
        # Validate result
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid model response format")
        
        required_keys = ['classification', 'confidence']
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise HTTPException(
                status_code=500, 
                detail=f"Missing required fields in model response: {missing_keys}"
            )
        
        # Format response
        response_data = {
            "status": "success",
            "request_id": request_id,
            "filename": file.filename,
            "classification": result['classification'],
            "confidence": float(result['confidence']),
            "model_type": "Vision Transformer",
            "processing_time": (datetime.utcnow() - timestamp).total_seconds(),
            "timestamp": timestamp.isoformat()
        }
        
        if return_probabilities and 'probabilities' in result:
            response_data["probabilities"] = result['probabilities']
        
        if extract_features:
            feature_info = {
                "vit_features_length": len(result.get('vit_features', []))
            }
            response_data["features"] = feature_info
            
            # Only include actual features if settings allow
            if hasattr(settings, 'EXTRACT_FEATURES') and settings.EXTRACT_FEATURES:
                if 'vit_features' in result:
                    response_data["vit_features"] = result['vit_features']
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing ViT request {request_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/classify-grains-batch", response_model=dict)
async def classify_grains_batch(
    files: List[UploadFile] = File(..., description="Multiple grain image files"),
    extract_features: bool = Query(True, description="Extract ViT features"),
    enhance_images: bool = Query(True, description="Apply image enhancement"),
    aggregate_results: bool = Query(True, description="Include aggregate statistics")
):
    """
    Classify multiple grain images using Vision Transformer batch processing
    
    - **files**: Multiple grain image files
    - **extract_features**: Extract ViT feature vectors
    - **enhance_images**: Apply preprocessing enhancement
    - **aggregate_results**: Include aggregate statistics
    
    Returns batch ViT classification results with aggregation
    """
    # Check if inference service is available
    if inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service not available. Please check model loading."
        )
    
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 50:  # Limit batch size for ViT processing
        raise HTTPException(
            status_code=400,
            detail=f"Too many files for ViT processing. Maximum allowed: 50, received: {len(files)}"
        )
    
    batch_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    
    try:
        # Load all images
        grain_images = []
        file_names = []
        failed_files = []
        
        for file in files:
            try:
                contents = await file.read()
                if len(contents) == 0:
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Empty file"
                    })
                    continue
                
                if not validate_image_file(contents, file.filename):
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Invalid image format"
                    })
                    continue
                    
                image = load_image_from_bytes(contents)
                grain_images.append(image)
                file_names.append(file.filename)
            except Exception as e:
                logger.error(f"Failed to process file {file.filename}: {e}")
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        if not grain_images:
            raise HTTPException(status_code=400, detail="No valid images to process")
        
        logger.info(f"Processing ViT batch {batch_id} with {len(grain_images)} images")
        
        # Batch process with ViT with timeout
        loop = asyncio.get_event_loop()
        try:
            batch_results = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    inference_service.batch_classify_grains,
                    grain_images,
                    extract_features,
                    enhance_images
                ),
                timeout=120.0  # 2 minute timeout for batch
            )
        except asyncio.TimeoutError:
            logger.error(f"Batch {batch_id} timed out")
            raise HTTPException(status_code=408, detail="Batch processing timeout")
        except Exception as e:
            logger.error(f"Error in batch inference: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
        
        # Validate batch results
        if not isinstance(batch_results, list) or len(batch_results) != len(grain_images):
            raise HTTPException(status_code=500, detail="Invalid batch processing results")
        
        # Format results
        formatted_results = []
        for i, result in enumerate(batch_results):
            try:
                formatted_result = {
                    "filename": file_names[i],
                    "grain_id": i + 1,
                    "classification": result['classification'],
                    "confidence": float(result['confidence']),
                    "model_type": "Vision Transformer"
                }
                
                if 'probabilities' in result:
                    formatted_result["probabilities"] = result['probabilities']
                
                if extract_features and 'vit_features' in result:
                    formatted_result["vit_features_length"] = len(result['vit_features'])
                    if hasattr(settings, 'EXTRACT_FEATURES') and settings.EXTRACT_FEATURES:
                        formatted_result["vit_features"] = result['vit_features']
                
                formatted_results.append(formatted_result)
            except Exception as e:
                logger.error(f"Error formatting result {i}: {e}")
                failed_files.append({
                    "filename": file_names[i],
                    "error": f"Result formatting failed: {str(e)}"
                })
        
        # Calculate aggregate statistics
        aggregate_stats = None
        if aggregate_results and formatted_results:
            try:
                aggregate_stats = calculate_vit_aggregate_statistics(batch_results)
            except Exception as e:
                logger.error(f"Error calculating aggregate stats: {e}")
                # Continue without aggregate stats
        
        response_data = {
            "status": "success",
            "batch_id": batch_id,
            "model_type": "Vision Transformer",
            "total_files": len(files),
            "successful_files": len(formatted_results),
            "failed_files": len(failed_files),
            "results": formatted_results,
            "failed_details": failed_files if failed_files else None,
            "aggregate_statistics": aggregate_stats,
            "processing_time": (datetime.utcnow() - timestamp).total_seconds(),
            "timestamp": timestamp.isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing ViT batch {batch_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing error: {str(e)}"
        )

@router.get("/model-info")
async def get_vit_model_info():
    """Get information about the Vision Transformer model"""
    try:
        if inference_service is None:
            raise HTTPException(
                status_code=503,
                detail="Inference service not available"
            )
        
        model_info = inference_service.get_model_info()
        
        return {
            "model_architecture": "Vision Transformer",
            "model_details": model_info,
            "available_models": getattr(settings, 'AVAILABLE_MODELS', ['vit-base']),
            "current_model": getattr(settings, 'MODEL_NAME', 'vit-base'),
            "input_size": getattr(settings, 'IMAGE_SIZE', [224, 224]),
            "classes": getattr(settings, 'CLASSES', ['healthy_seed', 'bad_seed', 'impurity']),
            "feature_extraction": getattr(settings, 'EXTRACT_FEATURES', True),
            "preprocessing": {
                "enhancement": True,
                "normalization": "ImageNet",
                "augmentation": "Training only"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ViT model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model information: {str(e)}"
        )

@router.get("/health")
async def vit_health_check():
    """Check if the ViT classification service is healthy"""
    try:
        # Check if model is loaded
        if inference_service is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": "Inference service not initialized",
                    "timestamp": datetime.utcnow().isoformat(),
                    "service": "vit_classification"
                }
            )
        
        model_status = inference_service.check_models_loaded()
        model_info = {}
        if model_status:
            try:
                model_info = inference_service.get_model_info()
            except Exception as e:
                logger.warning(f"Could not get model info: {e}")
        
        return {
            "status": "healthy" if model_status else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "version": getattr(settings, 'APP_VERSION', '1.0.0'),
            "model_loaded": model_status,
            "model_type": "Vision Transformer",
            "model_name": getattr(settings, 'MODEL_NAME', 'vit-base'),
            "device": str(getattr(inference_service, 'device', 'unknown')),
            "feature_extraction": getattr(settings, 'EXTRACT_FEATURES', True),
            "service": "vit_classification",
            "model_parameters": model_info.get('total_parameters', 'Unknown'),
            "classes": getattr(settings, 'CLASSES', ['healthy_seed', 'bad_seed', 'impurity'])
        }
    except Exception as e:
        logger.error(f"ViT health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "service": "vit_classification"
            }
        )

@router.post("/analyze-features")
async def analyze_grain_features(
    file: UploadFile = File(..., description="Grain image for feature analysis"),
    feature_type: str = Query("both", description="Feature type: 'vit', 'traditional', or 'both'")
):
    """
    Analyze and extract features from a grain image
    
    Provides detailed feature analysis using ViT and traditional methods
    """
    # Check if inference service is available
    if inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service not available. Please check model loading."
        )
    
    # Validate file
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        if not validate_image_file(contents, file.filename):
            raise HTTPException(status_code=400, detail="Invalid image format")
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in feature analysis: {e}")
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    try:
        # Load image with error handling
        try:
            image = load_image_from_bytes(contents)
        except Exception as e:
            logger.error(f"Failed to load image from bytes: {e}")
            raise HTTPException(status_code=400, detail="Could not process image file")
        
        logger.info(f"Feature analysis request for {file.filename}")
        
        # Extract features with timeout
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    inference_service.classify_single_grain,
                    image,
                    True,  # extract_features
                    True   # enhance
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Feature analysis timeout")
        except Exception as e:
            logger.error(f"Error in feature analysis inference: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Feature analysis failed: {str(e)}")
        
        # Format feature response based on type
        feature_response = {
            "filename": file.filename,
            "classification": result['classification'],
            "confidence": float(result['confidence']),
            "feature_analysis": {}
        }
        
        if feature_type in ['vit', 'both'] and 'vit_features' in result:
            vit_features = result['vit_features']
            if isinstance(vit_features, (list, np.ndarray)):
                vit_array = np.array(vit_features)
                feature_response["feature_analysis"]["vit"] = {
                    "feature_vector_length": len(vit_features),
                    "feature_statistics": {
                        "mean": float(np.mean(vit_array)),
                        "std": float(np.std(vit_array)),
                        "min": float(np.min(vit_array)),
                        "max": float(np.max(vit_array))
                    },
                    "features": vit_features[:50] if len(vit_features) > 50 else vit_features  # Truncate for response size
                }
        
        
        return JSONResponse(content=feature_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in feature analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Feature analysis error: {str(e)}"
        )

def calculate_vit_aggregate_statistics(batch_results: List[dict]) -> dict:
    """Calculate aggregate statistics for ViT batch results"""
    if not batch_results:
        return {}
    
    try:
        # Count classifications
        classification_counts = {'healthy_seed': 0, 'bad_seed': 0, 'impurity': 0}
        confidences = []
        
        for result in batch_results:
            classification = result.get('classification', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            if classification in classification_counts:
                classification_counts[classification] += 1
            confidences.append(float(confidence))
        
        total_grains = len(batch_results)
        
        # Calculate percentages
        percentages = {
            class_name: (count / total_grains) * 100 
            for class_name, count in classification_counts.items()
        }
        
        # Calculate quality score with safe defaults
        quality_weights = getattr(settings, 'QUALITY_WEIGHTS', {
            'healthy_seed': 1.0,
            'bad_seed': 0.5,
            'impurity': 0.0
        })
        
        quality_score = (
            classification_counts['healthy_seed'] * quality_weights.get('healthy_seed', 1.0) +
            classification_counts['bad_seed'] * quality_weights.get('bad_seed', 0.5) +
            classification_counts['impurity'] * quality_weights.get('impurity', 0.0)
        ) / total_grains
        
        # Determine grade with safe defaults
        quality_grades = getattr(settings, 'QUALITY_GRADES', {
            'A': 0.9,
            'B': 0.8,
            'C': 0.7,
            'D': 0.6,
            'F': 0.0
        })
        
        grade = 'F'
        for g, threshold in sorted(quality_grades.items(), key=lambda x: x[1], reverse=True):
            if quality_score >= threshold:
                grade = g
                break
        
        return {
            'total_grains': total_grains,
            'classification_distribution': classification_counts,
            'classification_percentages': {k: round(v, 2) for k, v in percentages.items()},
            'confidence_statistics': {
                'mean': round(np.mean(confidences), 3),
                'std': round(np.std(confidences), 3),
                'min': round(np.min(confidences), 3),
                'max': round(np.max(confidences), 3)
            },
            'quality_score': round(quality_score, 3),
            'quality_grade': grade,
            'model_type': 'Vision Transformer'
        }
    except Exception as e:
        logger.error(f"Error calculating aggregate statistics: {e}")
        return {
            'error': 'Failed to calculate statistics',
            'total_grains': len(batch_results),
            'model_type': 'Vision Transformer'
        }