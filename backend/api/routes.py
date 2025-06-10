import os
import uuid
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from backend.llm.model_loader import get_biomech_llm
from backend.core.pose_extraction import extract_pose_from_video
from backend.core.biomech_analysis import analyze_biomechanics
from backend.core.visualization import generate_visualization_data
from backend.config.settings import TEMP_UPLOAD_DIR, PROCESSED_DATA_DIR, SUPPORTED_VIDEO_FORMATS

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Models ---

class AnalysisRequest(BaseModel):
    video_id: str
    query: Optional[str] = None

class QueryRequest(BaseModel):
    query: str

class AnalysisResponse(BaseModel):
    analysis_id: str
    text: str
    key_points: List[str]
    recommendations: List[str]
    visualization_url: str

class VisualizationData(BaseModel):
    frames: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# --- Helper Functions ---

def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename"""
    return Path(filename).suffix.lower().lstrip(".")

def validate_video_file(file: UploadFile) -> bool:
    """Validate that the uploaded file is a supported video format"""
    ext = get_file_extension(file.filename)
    return ext in SUPPORTED_VIDEO_FORMATS

def process_video_background(video_path: str, video_id: str, sport_type: str):
    """Background task to process a video after upload"""
    try:
        # Extract pose data from the video
        pose_data = extract_pose_from_video(video_path, sport_type=sport_type)
        
        # Save the processed pose data
        output_path = PROCESSED_DATA_DIR / f"{video_id}.json"
        with open(output_path, "w") as f:
            import json
            json.dump(pose_data, f)
        
        logger.info(f"Successfully processed video {video_id}")
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        # Could implement notification or status update here

# --- Routes ---

@router.post("/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    sport_type: str = Form("general")
):
    """Upload a video for biomechanical analysis"""
    if not validate_video_file(video):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )
    
    # Generate a unique ID for this video
    video_id = str(uuid.uuid4())
    
    # Create a path to save the uploaded video
    video_path = TEMP_UPLOAD_DIR / f"{video_id}.{get_file_extension(video.filename)}"
    
    # Save the uploaded video
    with open(video_path, "wb") as f:
        f.write(await video.read())
    
    # Process the video in the background
    background_tasks.add_task(
        process_video_background, 
        str(video_path), 
        video_id, 
        sport_type
    )
    
    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Video uploaded successfully and is being processed"
    }

@router.post("/analysis", response_model=AnalysisResponse)
async def analyze_video(request: AnalysisRequest):
    """Analyze a processed video"""
    video_id = request.video_id
    
    # Check if the processed data exists
    processed_path = PROCESSED_DATA_DIR / f"{video_id}.json"
    if not processed_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video not found or still processing"
        )
    
    # Load the processed pose data
    with open(processed_path, "r") as f:
        import json
        pose_data = json.load(f)
    
    # Get the biomechanical LLM
    biomech_llm = get_biomech_llm()
    
    # Analyze the biomechanics
    analysis_id = str(uuid.uuid4())
    analysis_result = analyze_biomechanics(
        pose_data=pose_data,
        llm=biomech_llm,
        query=request.query
    )
    
    # Generate visualization data
    visualization_data = generate_visualization_data(
        pose_data=pose_data,
        analysis=analysis_result
    )
    
    # Save the visualization data
    viz_path = PROCESSED_DATA_DIR / f"{analysis_id}_viz.json"
    with open(viz_path, "w") as f:
        import json
        json.dump(visualization_data, f)
    
    # Extract key points and recommendations
    key_points = analysis_result.get("key_points", [])
    recommendations = analysis_result.get("recommendations", [])
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        text=analysis_result["text"],
        key_points=key_points,
        recommendations=recommendations,
        visualization_url=f"/api/analysis/{analysis_id}/visualization"
    )

@router.get("/analysis/{analysis_id}/visualization", response_model=VisualizationData)
async def get_visualization(analysis_id: str):
    """Get visualization data for an analysis"""
    viz_path = PROCESSED_DATA_DIR / f"{analysis_id}_viz.json"
    if not viz_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Visualization data not found"
        )
    
    # Load the visualization data
    with open(viz_path, "r") as f:
        import json
        visualization_data = json.load(f)
    
    return visualization_data

@router.post("/analysis/{analysis_id}/query")
async def query_analysis(analysis_id: str, request: QueryRequest):
    """Query an existing analysis with a specific question"""
    # Check if the visualization data exists (to confirm analysis exists)
    viz_path = PROCESSED_DATA_DIR / f"{analysis_id}_viz.json"
    if not viz_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    
    # Get the biomechanical LLM
    biomech_llm = get_biomech_llm()
    
    # Generate a response to the query
    # We'd typically load the original analysis for context
    response = biomech_llm.generate(
        prompt=f"Analysis ID: {analysis_id}\nQuery: {request.query}",
        max_tokens=1024
    )
    
    return {
        "query": request.query,
        "response": response["text"]
    }
