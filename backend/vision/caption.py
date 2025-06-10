"""
Vision-Language Captioning module for Spectacular.
Implements automatic captioning of frames using BLIP2-FLAN-T5-XL.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import cv2

class FrameCaptioner:
    """
    Frame captioning using BLIP2-FLAN-T5-XL model from Hugging Face.
    Auto-captions frames to provide richer LLM context.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the frame captioner.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the BLIP2-FLAN-T5-XL model from Hugging Face."""
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        
        model_name = "Salesforce/blip2-flan-t5-xl"
        print(f"Loading {model_name} model...")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.eval()
    
    def caption_image(self, image: np.ndarray, prompt: str = "Describe this image:") -> str:
        """
        Generate a caption for the input image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            prompt: Optional prompt to guide the captioning
            
        Returns:
            Caption text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize the FrameCaptioner first.")
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image[..., ::-1]  # BGR to RGB
        else:
            rgb_image = image
        
        # Process the image
        with torch.no_grad():
            # Prepare input for the model
            inputs = self.processor(
                images=rgb_image, 
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate caption
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            
            # Decode the caption
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption
    
    def caption_video(self, video_frames: List[np.ndarray], sample_rate: int = 5) -> List[Dict]:
        """
        Generate captions for a sequence of video frames.
        
        Args:
            video_frames: List of video frames
            sample_rate: Number of frames to skip between captions
            
        Returns:
            List of dictionaries with frame index and caption
        """
        captions = []
        
        # Process frames at the specified sample rate
        for i in range(0, len(video_frames), sample_rate):
            caption = self.caption_image(video_frames[i])
            captions.append({
                "frame_idx": i,
                "caption": caption
            })
        
        return captions
    
    def generate_llm_context(self, video_frames: List[np.ndarray], frame_interval: int = 10) -> str:
        """
        Generate a rich context description for an LLM based on video frames.
        
        Args:
            video_frames: List of video frames
            frame_interval: Interval between frames to sample for captioning
            
        Returns:
            Formatted text context for LLM
        """
        # Sample frames at the specified interval
        sampled_frames = [video_frames[i] for i in range(0, len(video_frames), frame_interval)]
        
        # Caption each sampled frame
        captions = []
        for i, frame in enumerate(sampled_frames):
            caption = self.caption_image(frame)
            frame_idx = i * frame_interval
            time_seconds = frame_idx / 30  # Assuming 30 FPS
            minutes = int(time_seconds // 60)
            seconds = int(time_seconds % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            
            captions.append(f"[{timestamp}] {caption}")
        
        # Format as a single text
        context = "Video Content Summary:\n" + "\n".join(captions)
        
        return context 