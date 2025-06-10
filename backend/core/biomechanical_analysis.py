import logging
from typing import Dict, Any, List, Optional

from backend.llm.model_loader import BiomechLLM

logger = logging.getLogger(__name__)

def format_prompt(pose_data: Dict[str, Any], query: Optional[str] = None) -> str:
    """Format a prompt for the biomechanical LLM based on pose data"""
    # Extract relevant information from pose data
    joints = pose_data.get("joints", {})
    angles = pose_data.get("angles", {})
    velocities = pose_data.get("velocities", {})
    sport_type = pose_data.get("metadata", {}).get("sport_type", "general")
    
    # Build a descriptive prompt
    prompt = f"Analyze the biomechanics of this {sport_type} movement:\n\n"
    
    # Add joint position information
    prompt += "Joint Positions:\n"
    for joint, positions in joints.items():
        # Only include a sample of positions to avoid token limits
        if isinstance(positions, list) and len(positions) > 10:
            # Sample at regular intervals
            sample_indices = [int(i * len(positions) / 10) for i in range(10)]
            sampled_positions = [positions[i] for i in sample_indices]
            prompt += f"- {joint}: {sampled_positions} (sampled from {len(positions)} frames)\n"
        else:
            prompt += f"- {joint}: {positions}\n"
    
    # Add joint angle information
    if angles:
        prompt += "\nJoint Angles:\n"
        for joint, angle_data in angles.items():
            prompt += f"- {joint}: {angle_data}\n"
    
    # Add velocity information
    if velocities:
        prompt += "\nVelocities:\n"
        for part, velocity_data in velocities.items():
            prompt += f"- {part}: {velocity_data}\n"
    
    # Add the specific query if provided
    if query:
        prompt += f"\nSpecific Question: {query}\n"
    else:
        prompt += "\nProvide a detailed biomechanical analysis of this movement. Include:\n"
        prompt += "1. Overall technique assessment\n"
        prompt += "2. Key strengths and weaknesses\n"
        prompt += "3. Specific recommendations for improvement\n"
        prompt += "4. Potential injury risks if any are present\n"
    
    return prompt

def extract_key_points(analysis_text: str) -> List[str]:
    """Extract key points from the analysis text"""
    # This could be done with a more sophisticated approach,
    # but for now we'll use a simple heuristic
    key_points = []
    lines = analysis_text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for bullet points, numbered lists, or sentences with key terms
        if (line.startswith('- ') or 
            (line[0:2].isdigit() and line[2:3] == '.') or
            any(term in line.lower() for term in ['key', 'important', 'critical', 'significant'])):
            # Clean up the line
            if line.startswith('- '):
                line = line[2:]
            elif line[0:2].isdigit() and line[2:3] == '.':
                line = line[3:]
            key_points.append(line.strip())
    
    # Limit to reasonable number of key points
    return key_points[:10]

def extract_recommendations(analysis_text: str) -> List[str]:
    """Extract recommendations from the analysis text"""
    recommendations = []
    in_recommendation_section = False
    lines = analysis_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
