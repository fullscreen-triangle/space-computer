"""
Pose pipeline implementation for Spectacular.
Implements 2D human pose estimation using YOLOv8 and RTMPose.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class PosePipeline:
    """
    Pipeline for 2D pose estimation using Hugging Face models.
    Primary model: ultralytics/yolov8s-pose
    Fallback model: qualcomm/RTMPose_Body2d for mobile devices
    """
    
    def __init__(self, use_fallback: bool = False, device: str = None):
        """
        Initialize the pose pipeline with the appropriate model.
        
        Args:
            use_fallback: Whether to use the fallback RTMPose model instead of YOLOv8
            device: Device to run inference on ('cuda', 'cpu', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fallback = use_fallback
        self.model = None
        self.processor = None
        
        # Load the appropriate model
        if self.use_fallback:
            self._load_rtmpose()
        else:
            self._load_yolov8()
    
    def _load_yolov8(self):
        """Load the YOLOv8 pose estimation model from Hugging Face."""
        from transformers import AutoModelForObjectDetection
        
        model_name = "ultralytics/yolov8s-pose"
        print(f"Loading {model_name} model...")
        
        # Load the model using the transformers library
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_name, 
            trust_remote_code=True,
            device_map=self.device
        )
        self.model.eval()
    
    def _load_rtmpose(self):
        """Load the RTMPose model from Hugging Face for mobile applications."""
        from transformers import AutoImageProcessor, AutoModelForPoseEstimation
        
        model_name = "qualcomm/RTMPose_Body2d"
        print(f"Loading {model_name} model...")
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForPoseEstimation.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=self.device
        )
        self.model.eval()
    
    def infer_2d(self, image: np.ndarray) -> Dict:
        """
        Run 2D pose estimation on the input image.
        
        Args:
            image: Input image (BGR format from OpenCV)
        
        Returns:
            Dictionary containing keypoints and pose information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize the PosePipeline first.")
            
        with torch.no_grad():
            if self.use_fallback:  # RTMPose
                # Process the image
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                
                # Process the outputs to get keypoints
                keypoints = outputs.keypoints.squeeze().cpu().numpy()
                scores = outputs.scores.squeeze().cpu().numpy()
                
                return {
                    "keypoints": keypoints,  # 133 keypoints for RTMPose
                    "scores": scores,
                    "model": "RTMPose_Body2d"
                }
            else:  # YOLOv8
                # Convert image to RGB for YOLOv8
                if len(image.shape) == 3 and image.shape[2] == 3:
                    rgb_image = image[..., ::-1]  # BGR to RGB
                else:
                    rgb_image = image
                
                # Run detection (YOLOv8 expects a different format)
                results = self.model(rgb_image)
                
                # Extract keypoints from the results
                if hasattr(results, 'keypoints') and len(results.keypoints) > 0:
                    keypoints = results.keypoints[0].data.cpu().numpy()  # 17 keypoints for YOLOv8
                    return {
                        "keypoints": keypoints,
                        "scores": results.boxes[0].conf.cpu().numpy() if len(results.boxes) > 0 else None,
                        "model": "YOLOv8-pose"
                    }
                else:
                    return {
                        "keypoints": None,
                        "scores": None,
                        "model": "YOLOv8-pose"
                    } 