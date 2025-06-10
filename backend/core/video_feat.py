"""
Video feature extraction module for Spectacular.
Implements video feature extraction using Video Swin Transformer.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import cv2

class VideoFeatureExtractor:
    """
    Video feature extraction using Video Swin Transformer from Hugging Face.
    Extracts motion features and can perform phase segmentation.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the video feature extractor.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the Video Swin Transformer model from Hugging Face."""
        from transformers import AutoImageProcessor, AutoModel
        
        model_name = "Tonic/video-swin-transformer"
        print(f"Loading {model_name} model...")
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            device_map=self.device
        )
        self.model.eval()
    
    def extract_features(self, video_frames: List[np.ndarray]) -> Dict:
        """
        Extract features from a sequence of video frames.
        
        Args:
            video_frames: List of video frames (numpy arrays)
        
        Returns:
            Dictionary containing video features and other information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize the VideoFeatureExtractor first.")
        
        # Process the frames
        with torch.no_grad():
            # Prepare input for the model
            inputs = self.processor(images=video_frames, return_tensors="pt").to(self.device)
            
            # Run the model
            outputs = self.model(**inputs)
            
            # Extract features
            features = outputs.last_hidden_state.cpu().numpy()
            pooled_features = outputs.pooler_output.cpu().numpy() if hasattr(outputs, 'pooler_output') else None
            
            return {
                "features": features,
                "pooled_features": pooled_features
            }
    
    def segment_phases(self, video_frames: List[np.ndarray], threshold: float = 0.5) -> Dict:
        """
        Segment the video into different motion phases.
        
        Args:
            video_frames: List of video frames
            threshold: Threshold for phase change detection
            
        Returns:
            Dictionary containing phase information and boundaries
        """
        # Extract features first
        features = self.extract_features(video_frames)
        
        # Perform phase segmentation based on feature similarity
        features_tensor = torch.tensor(features["features"], device=self.device)
        
        # Compute similarity between adjacent frames
        sim_scores = []
        for i in range(1, len(features_tensor)):
            similarity = torch.cosine_similarity(
                features_tensor[i-1].unsqueeze(0), 
                features_tensor[i].unsqueeze(0)
            ).item()
            sim_scores.append(similarity)
        
        # Detect phase boundaries based on similarity drops
        phase_boundaries = []
        for i in range(1, len(sim_scores)):
            if sim_scores[i-1] - sim_scores[i] > threshold:
                phase_boundaries.append(i)
        
        # Convert to numpy for return
        return {
            "phase_boundaries": np.array(phase_boundaries),
            "similarity_scores": np.array(sim_scores)
        } 