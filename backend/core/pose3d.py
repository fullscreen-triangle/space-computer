"""
3D pose estimation and motion embedding module for Spectacular.
Implements 3D pose lifting using MotionBERT-Lite.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class Pose3DEstimator:
    """
    3D pose estimation using MotionBERT-Lite model from Hugging Face.
    Also generates motion embeddings that can be used for action recognition.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the 3D pose estimator.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the MotionBERT-Lite model from Hugging Face."""
        from transformers import AutoProcessor, AutoModel
        
        model_name = "walterzhu/MotionBERT-Lite"
        print(f"Loading {model_name} model...")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            device_map=self.device
        )
        self.model.eval()
    
    def lift_2d_to_3d(self, keypoints_2d: np.ndarray, temporal_window: int = 27) -> Dict:
        """
        Lift 2D pose keypoints to 3D using MotionBERT-Lite.
        
        Args:
            keypoints_2d: Array of 2D keypoints, shape [frame_count, joint_count, 2]
            temporal_window: Number of frames to use for temporal context
        
        Returns:
            Dictionary containing 3D keypoints and motion embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize the Pose3DEstimator first.")
        
        # Ensure we have enough frames
        frame_count = keypoints_2d.shape[0]
        if frame_count < temporal_window:
            # Pad with copies of first and last frames if needed
            needed_padding = temporal_window - frame_count
            if needed_padding > 0:
                pad_front = needed_padding // 2
                pad_back = needed_padding - pad_front
                if pad_front > 0:
                    front_padding = np.repeat(keypoints_2d[0:1], pad_front, axis=0)
                    keypoints_2d = np.concatenate([front_padding, keypoints_2d], axis=0)
                if pad_back > 0:
                    back_padding = np.repeat(keypoints_2d[-1:], pad_back, axis=0)
                    keypoints_2d = np.concatenate([keypoints_2d, back_padding], axis=0)
        
        # Process the 2D keypoints
        with torch.no_grad():
            # Prepare input for the model
            inputs = self.processor(
                keypoints=keypoints_2d,
                return_tensors="pt"
            ).to(self.device)
            
            # Run the model
            outputs = self.model(**inputs)
            
            # Extract 3D keypoints and motion embeddings
            keypoints_3d = outputs.keypoints_3d.cpu().numpy()
            motion_embeddings = outputs.motion_embeddings.cpu().numpy()
            
            return {
                "keypoints_3d": keypoints_3d,
                "motion_embeddings": motion_embeddings
            }
    
    def get_motion_embedding(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Extract motion embeddings from 2D keypoints.
        
        Args:
            keypoints_2d: Array of 2D keypoints
            
        Returns:
            Motion embeddings that can be used for action recognition
        """
        result = self.lift_2d_to_3d(keypoints_2d)
        return result["motion_embeddings"] 