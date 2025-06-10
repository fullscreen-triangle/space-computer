"""
Automatic Speech Recognition (ASR) module for Spectacular.
Implements streaming speech-to-text using Whisper-large-v3.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import tempfile

class ASRProcessor:
    """
    ASR processor using Whisper-large-v3 for streaming speech-to-text.
    """
    
    def __init__(self, device: str = None, model_size: str = "large-v3"):
        """
        Initialize the ASR processor.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', etc.)
            model_size: Size of Whisper model to use ('tiny', 'base', 'small', 'medium', 'large-v3')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_size = model_size
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model from Hugging Face."""
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        model_name = f"openai/whisper-{self.model_size}"
        print(f"Loading {model_name} model...")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=self.device
        )
        self.model.eval()
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary containing transcription results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize the ASRProcessor first.")
        
        # Process the audio
        with torch.no_grad():
            # Prepare input for the model
            inputs = self.processor(
                audio_data, 
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            predicted_ids = self.model.generate(**inputs)
            
            # Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return {
                "text": transcription
            }
    
    def transcribe_file(self, audio_file_path: str) -> Dict:
        """
        Transcribe audio from a file.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary containing transcription results
        """
        # Load audio file
        import librosa
        audio_data, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)
        
        # Transcribe
        return self.transcribe_audio(audio_data, sample_rate)
    
    def streaming_transcribe(self, audio_stream, chunk_size: int = 4000, overlap: int = 400) -> Dict:
        """
        Perform streaming transcription on an audio stream.
        
        Args:
            audio_stream: Audio stream to transcribe
            chunk_size: Size of audio chunks to process
            overlap: Overlap between chunks
            
        Returns:
            Dictionary containing streaming transcription results
        """
        import librosa
        
        # Initialize buffer and results
        buffer = np.zeros(chunk_size)
        buffer_pos = 0
        full_text = ""
        
        # Process audio in chunks
        for chunk in audio_stream:
            # Add chunk to buffer
            chunk_len = len(chunk)
            if buffer_pos + chunk_len > chunk_size:
                # Buffer full, process it
                transcription = self.transcribe_audio(buffer)
                full_text += transcription["text"] + " "
                
                # Keep overlap for context continuity
                buffer = np.concatenate([buffer[-overlap:], np.zeros(chunk_size - overlap)])
                buffer_pos = overlap
            
            # Add current chunk to buffer
            buffer[buffer_pos:buffer_pos+chunk_len] = chunk
            buffer_pos += chunk_len
        
        # Process remaining audio
        if buffer_pos > 0:
            transcription = self.transcribe_audio(buffer[:buffer_pos])
            full_text += transcription["text"]
        
        return {
            "text": full_text.strip()
        } 