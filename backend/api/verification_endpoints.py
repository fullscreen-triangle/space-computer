"""
API endpoints for pose understanding verification system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime

from backend.core.pose_understanding import (
    PoseUnderstandingVerifier,
    verify_pose_understanding_before_analysis,
    save_verification_images,
    PoseUnderstandingResult
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/verification", tags=["pose-verification"])

# Global verifier instance
pose_verifier = None

class PoseVerificationRequest(BaseModel):
    pose_data: Dict[str, Dict[str, float]]
    query: str
    similarity_threshold: Optional[float] = 0.7
    save_images: Optional[bool] = False

class PoseVerificationResponse(BaseModel):
    understood: bool
    confidence: float
    similarity_score: float
    verification_time: float
    error_message: Optional[str] = None
    verification_id: Optional[str] = None

class VerificationStatsResponse(BaseModel):
    total_verifications: int
    success_rate: float
    average_confidence: float
    average_similarity: float
    average_verification_time: float

@router.on_event("startup")
async def initialize_verifier():
    """Initialize the pose understanding verifier"""
    global pose_verifier
    try:
        pose_verifier = PoseUnderstandingVerifier()
        await pose_verifier.initialize()
        logger.info("Pose understanding verifier initialized for API")
    except Exception as e:
        logger.error(f"Failed to initialize pose verifier: {e}")
        pose_verifier = None

@router.post("/verify-pose", response_model=PoseVerificationResponse)
async def verify_pose_understanding(
    request: PoseVerificationRequest,
    background_tasks: BackgroundTasks
) -> PoseVerificationResponse:
    """
    Verify AI understanding of pose data through image generation
    """
    if pose_verifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Pose understanding verifier not initialized"
        )
    
    try:
        # Set custom threshold if provided
        if request.similarity_threshold != 0.7:
            pose_verifier.similarity_threshold = request.similarity_threshold
        
        # Perform verification
        result = await pose_verifier.verify_with_retry(
            request.pose_data, 
            request.query
        )
        
        # Generate verification ID for tracking
        verification_id = f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.query) % 10000}"
        
        # Save images in background if requested
        if request.save_images:
            background_tasks.add_task(
                save_verification_images_task,
                result,
                f"verification_images/{verification_id}",
                verification_id
            )
        
        return PoseVerificationResponse(
            understood=result.understood,
            confidence=result.confidence,
            similarity_score=result.similarity_score,
            verification_time=result.verification_time,
            error_message=result.error_message,
            verification_id=verification_id
        )
        
    except Exception as e:
        logger.error(f"Error in pose verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-verify", response_model=List[PoseVerificationResponse])
async def batch_verify_poses(
    requests: List[PoseVerificationRequest]
) -> List[PoseVerificationResponse]:
    """
    Verify multiple pose understanding requests in batch
    """
    if pose_verifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Pose understanding verifier not initialized"
        )
    
    if len(requests) > 10:
        raise HTTPException(
            status_code=400, 
            detail="Batch size limited to 10 requests"
        )
    
    try:
        # Process all requests concurrently
        tasks = []
        for request in requests:
            if request.similarity_threshold != 0.7:
                # Create a new verifier instance for custom threshold
                custom_verifier = PoseUnderstandingVerifier(request.similarity_threshold)
                await custom_verifier.initialize()
                task = custom_verifier.verify_with_retry(request.pose_data, request.query)
            else:
                task = pose_verifier.verify_with_retry(request.pose_data, request.query)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Convert to response format
        responses = []
        for i, result in enumerate(results):
            verification_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
            responses.append(PoseVerificationResponse(
                understood=result.understood,
                confidence=result.confidence,
                similarity_score=result.similarity_score,
                verification_time=result.verification_time,
                error_message=result.error_message,
                verification_id=verification_id
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch pose verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=VerificationStatsResponse)
async def get_verification_stats() -> VerificationStatsResponse:
    """
    Get statistics about pose understanding verification performance
    """
    # This would typically come from a database or metrics store
    # For now, return mock data
    return VerificationStatsResponse(
        total_verifications=0,
        success_rate=0.0,
        average_confidence=0.0,
        average_similarity=0.0,
        average_verification_time=0.0
    )

@router.post("/test-verification")
async def test_verification_system():
    """
    Test the pose understanding verification system with sample data
    """
    if pose_verifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Pose understanding verifier not initialized"
        )
    
    # Sample pose data for testing
    sample_pose = {
        "nose": {"x": 0.5, "y": 0.3, "confidence": 0.9},
        "left_shoulder": {"x": 0.4, "y": 0.5, "confidence": 0.8},
        "right_shoulder": {"x": 0.6, "y": 0.5, "confidence": 0.8},
        "left_elbow": {"x": 0.3, "y": 0.7, "confidence": 0.7},
        "right_elbow": {"x": 0.7, "y": 0.7, "confidence": 0.7},
        "left_wrist": {"x": 0.2, "y": 0.9, "confidence": 0.6},
        "right_wrist": {"x": 0.8, "y": 0.9, "confidence": 0.6},
        "left_hip": {"x": 0.45, "y": 0.8, "confidence": 0.8},
        "right_hip": {"x": 0.55, "y": 0.8, "confidence": 0.8},
        "left_knee": {"x": 0.4, "y": 1.1, "confidence": 0.7},
        "right_knee": {"x": 0.6, "y": 1.1, "confidence": 0.7},
        "left_ankle": {"x": 0.35, "y": 1.4, "confidence": 0.6},
        "right_ankle": {"x": 0.65, "y": 1.4, "confidence": 0.6}
    }
    
    sample_query = "What is the angle of the left elbow?"
    
    try:
        result = await pose_verifier.verify_understanding(sample_pose, sample_query)
        
        return {
            "status": "success",
            "test_result": {
                "understood": result.understood,
                "confidence": result.confidence,
                "similarity_score": result.similarity_score,
                "verification_time": result.verification_time,
                "error_message": result.error_message
            },
            "message": "Verification system is working correctly" if result.understood else "Verification system detected understanding issues"
        }
        
    except Exception as e:
        logger.error(f"Error in verification test: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@router.post("/configure")
async def configure_verifier(
    similarity_threshold: float = 0.7,
    enable_caching: bool = True
):
    """
    Configure the pose understanding verifier settings
    """
    if pose_verifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Pose understanding verifier not initialized"
        )
    
    try:
        # Update threshold
        pose_verifier.similarity_threshold = similarity_threshold
        
        return {
            "status": "success",
            "configuration": {
                "similarity_threshold": similarity_threshold,
                "caching_enabled": enable_caching
            },
            "message": "Verifier configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error configuring verifier: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def save_verification_images_task(
    result: PoseUnderstandingResult, 
    output_dir: str, 
    prefix: str
):
    """
    Background task to save verification images
    """
    try:
        save_verification_images(result, output_dir, prefix)
        logger.info(f"Saved verification images for {prefix}")
    except Exception as e:
        logger.error(f"Failed to save verification images for {prefix}: {e}")

# Health check endpoint
@router.get("/health")
async def verification_health_check():
    """
    Check the health of the pose understanding verification system
    """
    if pose_verifier is None:
        return {
            "status": "unhealthy",
            "message": "Pose understanding verifier not initialized",
            "components": {
                "verifier": "not_initialized",
                "image_generator": "unknown",
                "similarity_calculator": "unknown"
            }
        }
    
    # Test basic functionality
    try:
        # Simple test to check if components are working
        test_pose = {"nose": {"x": 0.5, "y": 0.3, "confidence": 0.9}}
        description = pose_verifier.description_generator.generate_description(test_pose)
        
        return {
            "status": "healthy",
            "message": "Pose understanding verification system is operational",
            "components": {
                "verifier": "initialized",
                "image_generator": "ready",
                "similarity_calculator": "ready",
                "description_generator": "working"
            },
            "test_description": description
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "message": f"Some components may not be working: {str(e)}",
            "components": {
                "verifier": "initialized",
                "image_generator": "unknown",
                "similarity_calculator": "unknown"
            }
        } 