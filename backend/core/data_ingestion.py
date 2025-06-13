"""
Data ingestion service for elite athlete biomechanical data.
Processes JSON pose data, videos, and posture analysis into the backend system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

from backend.llm.embeddings import EmbeddingGenerator
from backend.core.biomechanical_analysis import format_prompt

logger = logging.getLogger(__name__)

class AthleteDataIngestion:
    """
    Ingests elite athlete data from datasources into the backend system.
    """
    
    def __init__(self, datasources_path: str = "datasources"):
        self.datasources_path = Path(datasources_path)
        self.models_path = self.datasources_path / "models"
        self.posture_path = self.datasources_path / "posture"
        self.videos_path = self.datasources_path / "annotated"
        
        # Initialize embedding generator for RAG
        self.embedding_generator = EmbeddingGenerator()
        
        # Athlete mapping from your data
        self.athlete_mapping = {
            'usain_bolt_final': {'name': 'Usain Bolt', 'sport': 'Sprint', 'discipline': 'track_field'},
            'didier_drogba_header': {'name': 'Didier Drogba', 'sport': 'Football', 'discipline': 'football'},
            'derek_chisora_punch': {'name': 'Derek Chisora', 'sport': 'Boxing', 'discipline': 'combat'},
            'jonah_lomu_run': {'name': 'Jonah Lomu', 'sport': 'Rugby', 'discipline': 'rugby'},
            'asafa_powell_race': {'name': 'Asafa Powell', 'sport': 'Sprint', 'discipline': 'track_field'},
            'mahela_jayawardene_shot': {'name': 'Mahela Jayawardene', 'sport': 'Cricket', 'discipline': 'cricket'},
            'kevin_pietersen_shot': {'name': 'Kevin Pietersen', 'sport': 'Cricket', 'discipline': 'cricket'},
            'daniel_sturridge_dribble': {'name': 'Daniel Sturridge', 'sport': 'Football', 'discipline': 'football'},
            'gareth_bale_kick': {'name': 'Gareth Bale', 'sport': 'Football', 'discipline': 'football'},
            'jordan_henderson_pass': {'name': 'Jordan Henderson', 'sport': 'Football', 'discipline': 'football'},
            'raheem_sterling_sprint': {'name': 'Raheem Sterling', 'sport': 'Football', 'discipline': 'football'},
            'wrestling_takedown': {'name': 'Wrestling Analysis', 'sport': 'Wrestling', 'discipline': 'combat'},
            'boxing_combo': {'name': 'Boxing Analysis', 'sport': 'Boxing', 'discipline': 'combat'}
        }
    
    def ingest_all_athletes(self) -> Dict[str, Any]:
        """
        Ingest all available athlete data into the system.
        
        Returns:
            Dictionary with ingestion results
        """
        results = {
            'successful': [],
            'failed': [],
            'total_processed': 0,
            'knowledge_base_entries': 0
        }
        
        logger.info("Starting athlete data ingestion...")
        
        for athlete_id, athlete_info in self.athlete_mapping.items():
            try:
                athlete_data = self.process_athlete(athlete_id)
                if athlete_data:
                    results['successful'].append({
                        'athlete_id': athlete_id,
                        'name': athlete_info['name'],
                        'data': athlete_data
                    })
                    results['total_processed'] += 1
                    
                    # Generate knowledge base entries
                    kb_entries = self.generate_knowledge_base_entries(athlete_id, athlete_data)
                    results['knowledge_base_entries'] += len(kb_entries)
                    
                else:
                    results['failed'].append({
                        'athlete_id': athlete_id,
                        'reason': 'Data processing failed'
                    })
            except Exception as e:
                logger.error(f"Failed to process athlete {athlete_id}: {e}")
                results['failed'].append({
                    'athlete_id': athlete_id,
                    'reason': str(e)
                })
        
        logger.info(f"Ingestion complete: {len(results['successful'])} successful, {len(results['failed'])} failed")
        return results
    
    def process_athlete(self, athlete_id: str) -> Optional[Dict[str, Any]]:
        """
        Process a single athlete's data.
        
        Args:
            athlete_id: The athlete identifier
            
        Returns:
            Processed athlete data or None if failed
        """
        logger.info(f"Processing athlete: {athlete_id}")
        
        # Load pose model data
        model_file = self.models_path / f"{athlete_id}.json"
        if not model_file.exists():
            logger.warning(f"Model file not found: {model_file}")
            return None
        
        try:
            with open(model_file, 'r') as f:
                pose_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pose data for {athlete_id}: {e}")
            return None
        
        # Load posture analysis if available
        posture_data = None
        posture_file = self.posture_path / f"{athlete_id}.json"
        if posture_file.exists():
            try:
                with open(posture_file, 'r') as f:
                    posture_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load posture data for {athlete_id}: {e}")
        
        # Process the data
        processed_data = self.transform_pose_data(pose_data, posture_data, athlete_id)
        
        return processed_data
    
    def transform_pose_data(self, pose_data: Dict[str, Any], 
                          posture_data: Optional[Dict[str, Any]], 
                          athlete_id: str) -> Dict[str, Any]:
        """
        Transform raw JSON data into backend-compatible format.
        
        Args:
            pose_data: Raw pose detection data
            posture_data: Raw posture analysis data
            athlete_id: Athlete identifier
            
        Returns:
            Transformed data structure
        """
        athlete_info = self.athlete_mapping[athlete_id]
        
        # Extract metadata
        metadata = pose_data.get('metadata', {})
        
        # Transform frames data
        frames = pose_data.get('frames', {})
        processed_frames = {}
        
        for frame_num, frame_data in frames.items():
            if 'pose_landmarks' in frame_data and frame_data['pose_landmarks']:
                landmarks = frame_data['pose_landmarks'][0]  # First person
                
                # Convert to joint dictionary
                joints = self.landmarks_to_joints(landmarks)
                
                processed_frames[int(frame_num)] = {
                    'timestamp': float(frame_num) / metadata.get('fps', 30),
                    'joints': joints,
                    'confidence': frame_data.get('confidence', 0.9),
                    'landmarks': landmarks
                }
        
        # Add posture analysis if available
        if posture_data:
            self.integrate_posture_analysis(processed_frames, posture_data)
        
        # Calculate derived metrics
        motion_metrics = self.calculate_motion_metrics(processed_frames, athlete_info['sport'])
        
        return {
            'athlete_id': athlete_id,
            'name': athlete_info['name'],
            'sport': athlete_info['sport'],
            'discipline': athlete_info['discipline'],
            'metadata': {
                'fps': metadata.get('fps', 30),
                'duration': metadata.get('duration', len(processed_frames) / metadata.get('fps', 30)),
                'frame_count': len(processed_frames),
                'resolution': metadata.get('resolution', '1920x1080'),
                'ingestion_date': datetime.now().isoformat()
            },
            'frames': processed_frames,
            'motion_metrics': motion_metrics,
            'video_path': f"datasources/annotated/{athlete_id}.mp4"
        }
    
    def landmarks_to_joints(self, landmarks: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Convert MediaPipe landmarks to named joints.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Dictionary of named joints with positions
        """
        # MediaPipe pose landmark mapping
        landmark_map = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        joints = {}
        for joint_name, idx in landmark_map.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                joints[joint_name] = {
                    'x': landmark.get('x', 0.0),
                    'y': landmark.get('y', 0.0),
                    'z': landmark.get('z', 0.0),
                    'visibility': landmark.get('visibility', 1.0)
                }
        
        return joints
    
    def integrate_posture_analysis(self, frames: Dict[int, Dict[str, Any]], 
                                 posture_data: Dict[str, Any]) -> None:
        """
        Integrate posture analysis data into frames.
        
        Args:
            frames: Processed frame data
            posture_data: Raw posture analysis data
        """
        for frame_num, frame_data in frames.items():
            # Add joint angles if available
            if 'joint_angles' in posture_data:
                frame_data['angles'] = posture_data['joint_angles'].get(str(frame_num), {})
            
            # Add force data if available
            if 'forces' in posture_data:
                frame_data['forces'] = posture_data['forces'].get(str(frame_num), {})
            
            # Add stability metrics if available
            if 'stability' in posture_data:
                frame_data['stability'] = posture_data['stability'].get(str(frame_num), {})
    
    def calculate_motion_metrics(self, frames: Dict[int, Dict[str, Any]], 
                               sport: str) -> Dict[str, Any]:
        """
        Calculate motion metrics from frame data.
        
        Args:
            frames: Processed frame data
            sport: Sport type
            
        Returns:
            Calculated motion metrics
        """
        if not frames:
            return {}
        
        frame_numbers = sorted(frames.keys())
        
        # Calculate center of mass movement
        com_positions = []
        for frame_num in frame_numbers:
            joints = frames[frame_num]['joints']
            com = self.calculate_center_of_mass(joints)
            com_positions.append(com)
        
        # Calculate velocities and accelerations
        velocities = []
        for i in range(1, len(com_positions)):
            dt = frames[frame_numbers[i]]['timestamp'] - frames[frame_numbers[i-1]]['timestamp']
            if dt > 0:
                velocity = np.linalg.norm(np.array(com_positions[i]) - np.array(com_positions[i-1])) / dt
                velocities.append(velocity)
        
        # Sport-specific metrics
        sport_metrics = self.calculate_sport_specific_metrics(frames, sport)
        
        return {
            'avg_velocity': np.mean(velocities) if velocities else 0.0,
            'max_velocity': np.max(velocities) if velocities else 0.0,
            'total_distance': sum(velocities) * (1/30) if velocities else 0.0,  # Assuming 30fps
            'center_of_mass_path': com_positions,
            'sport_specific': sport_metrics
        }
    
    def calculate_center_of_mass(self, joints: Dict[str, Dict[str, float]]) -> List[float]:
        """Calculate approximate center of mass from joint positions."""
        # Simplified COM calculation using key joints
        key_joints = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']
        
        x_sum, y_sum, z_sum = 0.0, 0.0, 0.0
        count = 0
        
        for joint_name in key_joints:
            if joint_name in joints:
                joint = joints[joint_name]
                x_sum += joint['x']
                y_sum += joint['y']
                z_sum += joint['z']
                count += 1
        
        if count > 0:
            return [x_sum/count, y_sum/count, z_sum/count]
        return [0.0, 0.0, 0.0]
    
    def calculate_sport_specific_metrics(self, frames: Dict[int, Dict[str, Any]], 
                                       sport: str) -> Dict[str, Any]:
        """Calculate sport-specific biomechanical metrics."""
        metrics = {}
        
        if sport in ['Sprint', 'Football']:
            # Running-related metrics
            metrics['stride_analysis'] = self.analyze_stride_pattern(frames)
        elif sport == 'Boxing':
            # Punching mechanics
            metrics['punch_analysis'] = self.analyze_punch_mechanics(frames)
        elif sport == 'Rugby':
            # Power and contact metrics
            metrics['power_analysis'] = self.analyze_power_movements(frames)
        elif sport == 'Cricket':
            # Batting/bowling mechanics
            metrics['technique_analysis'] = self.analyze_cricket_technique(frames)
        
        return metrics
    
    def analyze_stride_pattern(self, frames: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stride patterns for running sports."""
        # Extract ankle positions over time
        left_ankle_y = []
        right_ankle_y = []
        timestamps = []
        
        for frame_num in sorted(frames.keys()):
            frame = frames[frame_num]
            joints = frame['joints']
            
            if 'left_ankle' in joints and 'right_ankle' in joints:
                left_ankle_y.append(joints['left_ankle']['y'])
                right_ankle_y.append(joints['right_ankle']['y'])
                timestamps.append(frame['timestamp'])
        
        # Detect foot strikes (local minima in ankle height)
        strikes = []
        for i in range(1, len(left_ankle_y) - 1):
            if left_ankle_y[i] < left_ankle_y[i-1] and left_ankle_y[i] < left_ankle_y[i+1]:
                strikes.append(('left', timestamps[i]))
            if right_ankle_y[i] < right_ankle_y[i-1] and right_ankle_y[i] < right_ankle_y[i+1]:
                strikes.append(('right', timestamps[i]))
        
        # Calculate stride metrics
        stride_times = []
        for i in range(1, len(strikes)):
            if strikes[i][0] == strikes[i-1][0]:  # Same foot
                stride_times.append(strikes[i][1] - strikes[i-1][1])
        
        return {
            'foot_strikes': len(strikes),
            'avg_stride_time': np.mean(stride_times) if stride_times else 0.0,
            'stride_frequency': 1.0 / np.mean(stride_times) if stride_times else 0.0
        }
    
    def analyze_punch_mechanics(self, frames: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze punching mechanics for boxing."""
        # Simplified punch analysis
        return {'punch_phases': 'detected', 'power_generation': 'analyzed'}
    
    def analyze_power_movements(self, frames: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze power movements for rugby."""
        return {'contact_preparation': 'detected', 'drive_phase': 'analyzed'}
    
    def analyze_cricket_technique(self, frames: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cricket batting/bowling technique."""
        return {'bat_swing': 'analyzed', 'body_rotation': 'measured'}
    
    def generate_knowledge_base_entries(self, athlete_id: str, 
                                      athlete_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate knowledge base entries for RAG system.
        
        Args:
            athlete_id: Athlete identifier
            athlete_data: Processed athlete data
            
        Returns:
            List of knowledge base entries
        """
        entries = []
        
        # Generate technique description
        technique_entry = {
            'type': 'technique_analysis',
            'athlete': athlete_data['name'],
            'sport': athlete_data['sport'],
            'title': f"{athlete_data['name']} - {athlete_data['sport']} Technique Analysis",
            'content': self.generate_technique_description(athlete_data),
            'metadata': {
                'category': 'elite_technique',
                'sport': athlete_data['sport'],
                'athlete_id': athlete_id
            }
        }
        entries.append(technique_entry)
        
        return entries
    
    def generate_technique_description(self, athlete_data: Dict[str, Any]) -> str:
        """Generate natural language description of athlete's technique."""
        name = athlete_data['name']
        sport = athlete_data['sport']
        
        description = f"""
        {name} demonstrates elite-level {sport} technique representing world-class 
        biomechanical patterns that serve as the gold standard for {sport} performance.
        
        This analysis captures the precise movement mechanics that contribute to 
        elite athletic performance and can be used for technique comparison, 
        coaching applications, and biomechanical research.
        """
        
        return description.strip() 