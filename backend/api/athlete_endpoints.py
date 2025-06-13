"""
API endpoints for athlete data integration with Space Computer frontend.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import logging

from backend.core.data_ingestion import AthleteDataIngestion
from backend.llm.model_loader import get_biomech_llm
from backend.core.biomechanical_analysis import format_prompt
from orchestration.rag_engine import BiomechanicalRAG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/athletes", tags=["athletes"])

# Initialize services
data_ingestion = AthleteDataIngestion()
biomech_llm = get_biomech_llm()

@router.get("/list")
async def list_athletes() -> Dict[str, Any]:
    """
    Get list of all available athletes.
    
    Returns:
        Dictionary containing athlete information
    """
    try:
        athletes = []
        for athlete_id, athlete_info in data_ingestion.athlete_mapping.items():
            athletes.append({
                'id': athlete_id,
                'name': athlete_info['name'],
                'sport': athlete_info['sport'],
                'discipline': athlete_info['discipline'],
                'video_url': f"/datasources/annotated/{athlete_id}.mp4",
                'model_url': f"/datasources/models/{athlete_id}.json"
            })
        
        return {
            'athletes': athletes,
            'total_count': len(athletes)
        }
    except Exception as e:
        logger.error(f"Error listing athletes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{athlete_id}")
async def get_athlete_data(athlete_id: str) -> Dict[str, Any]:
    """
    Get complete data for a specific athlete.
    
    Args:
        athlete_id: The athlete identifier
        
    Returns:
        Complete athlete data including pose, metadata, and metrics
    """
    try:
        if athlete_id not in data_ingestion.athlete_mapping:
            raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")
        
        athlete_data = data_ingestion.process_athlete(athlete_id)
        if not athlete_data:
            raise HTTPException(status_code=404, detail=f"Data not available for athlete {athlete_id}")
        
        return athlete_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting athlete data for {athlete_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{athlete_id}/frame/{frame_number}")
async def get_frame_data(athlete_id: str, frame_number: int) -> Dict[str, Any]:
    """
    Get pose data for a specific frame.
    
    Args:
        athlete_id: The athlete identifier
        frame_number: Frame number to retrieve
        
    Returns:
        Frame-specific pose and analysis data
    """
    try:
        athlete_data = data_ingestion.process_athlete(athlete_id)
        if not athlete_data:
            raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")
        
        frames = athlete_data.get('frames', {})
        if frame_number not in frames:
            raise HTTPException(status_code=404, detail=f"Frame {frame_number} not found")
        
        frame_data = frames[frame_number]
        
        # Add athlete context
        frame_data['athlete_info'] = {
            'id': athlete_id,
            'name': athlete_data['name'],
            'sport': athlete_data['sport']
        }
        
        return frame_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting frame data for {athlete_id}, frame {frame_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{athlete_id}/analyze")
async def analyze_athlete_movement(
    athlete_id: str,
    frame_number: Optional[int] = None,
    query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform AI analysis on athlete movement.
    
    Args:
        athlete_id: The athlete identifier
        frame_number: Optional specific frame to analyze
        query: Optional specific question about the movement
        
    Returns:
        AI-generated biomechanical analysis
    """
    try:
        athlete_data = data_ingestion.process_athlete(athlete_id)
        if not athlete_data:
            raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")
        
        # Prepare analysis context
        if frame_number is not None:
            frames = athlete_data.get('frames', {})
            if frame_number not in frames:
                raise HTTPException(status_code=404, detail=f"Frame {frame_number} not found")
            
            # Analyze specific frame
            frame_data = frames[frame_number]
            analysis_context = {
                'joints': frame_data.get('joints', {}),
                'angles': frame_data.get('angles', {}),
                'forces': frame_data.get('forces', {}),
                'metadata': {
                    'sport_type': athlete_data['sport'],
                    'athlete_name': athlete_data['name'],
                    'frame_number': frame_number,
                    'timestamp': frame_data.get('timestamp', 0)
                }
            }
        else:
            # Analyze overall movement
            analysis_context = {
                'motion_metrics': athlete_data.get('motion_metrics', {}),
                'metadata': {
                    'sport_type': athlete_data['sport'],
                    'athlete_name': athlete_data['name'],
                    'total_frames': len(athlete_data.get('frames', {}))
                }
            }
        
        # Generate AI analysis
        prompt = format_prompt(analysis_context, query)
        
        response = biomech_llm.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7
        )
        
        return {
            'athlete_id': athlete_id,
            'athlete_name': athlete_data['name'],
            'sport': athlete_data['sport'],
            'frame_number': frame_number,
            'query': query,
            'analysis': response['text'],
            'model_info': response.get('model', 'BiomechLLM'),
            'usage': response.get('usage', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing athlete {athlete_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{athlete_id}/metrics")
async def get_athlete_metrics(athlete_id: str) -> Dict[str, Any]:
    """
    Get calculated motion metrics for an athlete.
    
    Args:
        athlete_id: The athlete identifier
        
    Returns:
        Motion metrics and performance indicators
    """
    try:
        athlete_data = data_ingestion.process_athlete(athlete_id)
        if not athlete_data:
            raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")
        
        metrics = athlete_data.get('motion_metrics', {})
        
        return {
            'athlete_id': athlete_id,
            'athlete_name': athlete_data['name'],
            'sport': athlete_data['sport'],
            'metrics': metrics,
            'metadata': athlete_data.get('metadata', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics for athlete {athlete_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare")
async def compare_athletes(
    athlete_ids: List[str],
    comparison_type: str = Query("technique", description="Type of comparison: technique, performance, biomechanics")
) -> Dict[str, Any]:
    """
    Compare multiple athletes' biomechanical data.
    
    Args:
        athlete_ids: List of athlete identifiers to compare
        comparison_type: Type of comparison to perform
        
    Returns:
        Comparative analysis results
    """
    try:
        if len(athlete_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 athletes required for comparison")
        
        # Get data for all athletes
        athletes_data = {}
        for athlete_id in athlete_ids:
            if athlete_id not in data_ingestion.athlete_mapping:
                raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")
            
            data = data_ingestion.process_athlete(athlete_id)
            if data:
                athletes_data[athlete_id] = data
        
        if len(athletes_data) < 2:
            raise HTTPException(status_code=404, detail="Insufficient athlete data for comparison")
        
        # Generate comparison analysis
        comparison_prompt = f"""
        Compare the following elite athletes in terms of {comparison_type}:
        
        """
        
        for athlete_id, data in athletes_data.items():
            comparison_prompt += f"""
            {data['name']} ({data['sport']}):
            - Sport: {data['sport']}
            - Performance metrics: {data.get('motion_metrics', {})}
            
            """
        
        comparison_prompt += f"""
        Provide a detailed {comparison_type} comparison focusing on:
        1. Key differences in movement patterns
        2. Strengths and weaknesses of each athlete
        3. Technical insights and recommendations
        4. What can be learned from each athlete's approach
        """
        
        response = biomech_llm.generate(
            prompt=comparison_prompt,
            max_tokens=1024,
            temperature=0.7
        )
        
        return {
            'comparison_type': comparison_type,
            'athletes': [
                {
                    'id': athlete_id,
                    'name': data['name'],
                    'sport': data['sport']
                }
                for athlete_id, data in athletes_data.items()
            ],
            'analysis': response['text'],
            'model_info': response.get('model', 'BiomechLLM')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing athletes {athlete_ids}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_athlete_data() -> Dict[str, Any]:
    """
    Trigger ingestion of all athlete data into the system.
    
    Returns:
        Ingestion results and status
    """
    try:
        results = data_ingestion.ingest_all_athletes()
        return {
            'status': 'completed',
            'results': results,
            'message': f"Successfully processed {results['total_processed']} athletes"
        }
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 