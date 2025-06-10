"""
RGB-based action recognition module for Spectacular.
Implements action recognition using fine-tuned Video Swin Transformer.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class RGBActionClassifier:
    """
    RGB-based action classification using fine-tuned Video Swin Transformer.
    Provides context-rich phase and technique classification.
    """
    
    def __init__(self, num_classes: int = 10, device: str = None):
        """
        Initialize the RGB action classifier.
        
        Args:
            num_classes: Number of action classes to classify
            device: Device to run inference on ('cuda', 'cpu', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the fine-tuned Video Swin Transformer for action recognition.
        """
        from transformers import AutoImageProcessor, AutoModelForVideoClassification
        
        model_name = "Tonic/video-swin-transformer"
        print(f"Loading fine-tuned {model_name} model for action recognition...")
        
        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Load base model
        base_model = AutoModelForVideoClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=self.num_classes
        )
        
        # In practice, you would fine-tune this model with your specific action classes
        # For demonstration, we'll use the base model
        self.model = base_model
        self.model.to(self.device)
        self.model.eval()
        
        print("Note: Using base Video Swin Transformer. For production use, fine-tune with your action data.")
    
    def classify_action(self, video_frames: List[np.ndarray]) -> Dict:
        """
        Classify action from a sequence of video frames.
        
        Args:
            video_frames: List of video frames (numpy arrays)
            
        Returns:
            Dictionary containing action classification results
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Initialize the RGBActionClassifier first.")
        
        # Process the frames
        with torch.no_grad():
            # Prepare input for the model
            inputs = self.processor(images=video_frames, return_tensors="pt").to(self.device)
            
            # Run the model
            outputs = self.model(**inputs)
            
            # Extract classification results
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
            predicted_class = np.argmax(probabilities)
            
            return {
                "predicted_class": int(predicted_class),
                "probabilities": probabilities,
                "confidence": float(probabilities[predicted_class])
            }
    
    def classify_phases(self, video_frames: List[np.ndarray], window_size: int = 16, stride: int = 8) -> List[Dict]:
        """
        Classify phases in a longer video by sliding a window over frames.
        
        Args:
            video_frames: List of video frames
            window_size: Size of sliding window
            stride: Step size for sliding window
            
        Returns:
            List of classification results for each window
        """
        results = []
        for i in range(0, len(video_frames) - window_size + 1, stride):
            window = video_frames[i:i+window_size]
            result = self.classify_action(window)
            result["start_frame"] = i
            result["end_frame"] = i + window_size - 1
            results.append(result)
        
        return results 