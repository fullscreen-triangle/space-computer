"""
Text-to-Speech (TTS) module for Spectacular.
Implements voice cloning and text synthesis using XTTS-v2.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import tempfile

class TTSEngine:
    """
    Text-to-Speech engine using XTTS-v2 with voice cloning capabilities.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the TTS engine.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the XTTS model from Hugging Face."""
        from transformers import pipeline
        
        model_name = "coqui/XTTS-v2"
        print(f"Loading {model_name} model...")
        
        # Load the model using pipeline for simplicity
        self.model = pipeline(
            "text-to-speech",
            model=model_name,
            device_map=self.device
        )
    
    def synthesize(self, text: str, speaker_embedding: Optional[np.ndarray] = None,
                   language: str = "en") -> Dict:
        """
        Synthesize speech from text with optional voice cloning.
        
        Args:
            text: Text to synthesize
            speaker_embedding: Optional speaker embedding for voice cloning
            language: Language code for synthesis ("en", "fr", "de", etc.)
            
        Returns:
            Dictionary containing audio data and sample rate
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize the TTSEngine first.")
        
        # Prepare inputs
        inputs = {
            "text": text,
            "language": language
        }
        
        # Add speaker embedding if provided
        if speaker_embedding is not None:
            inputs["speaker_embeddings"] = speaker_embedding
        
        # Generate speech
        outputs = self.model(**inputs)
        
        # Return audio data and sample rate
        return {
            "audio": outputs["audio"],
            "sample_rate": outputs["sampling_rate"]
        }
    
    def clone_voice(self, reference_audio: Union[str, np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        """
        Generate speaker embedding from reference audio for voice cloning.
        
        Args:
            reference_audio: Path to audio file or audio data as numpy array
            sample_rate: Sample rate of the audio (if providing array)
            
        Returns:
            Speaker embedding for use with synthesize()
        """
        import librosa
        
        # Load audio if path is provided
        if isinstance(reference_audio, str):
            audio_data, sample_rate = librosa.load(reference_audio, sr=16000, mono=True)
        else:
            audio_data = reference_audio
        
        # Ensure we have enough audio (XTTS needs about 6 seconds)
        min_length = 6 * sample_rate
        if len(audio_data) < min_length:
            # Pad by repeating if needed
            repeats = int(np.ceil(min_length / len(audio_data)))
            audio_data = np.tile(audio_data, repeats)[:min_length]
        
        # Extract speaker embedding
        speaker_embedding = self.model.extract_speaker_embedding(
            audio_data, 
            sample_rate=sample_rate
        ).cpu().numpy()
        
        return speaker_embedding
    
    def save_audio(self, audio_data: np.ndarray, sample_rate: int, output_path: str) -> str:
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            output_path: Path to save the audio file
            
        Returns:
            Path to the saved audio file
        """
        import soundfile as sf
        
        # Save audio file
        sf.write(output_path, audio_data, sample_rate)
        
        return output_path 