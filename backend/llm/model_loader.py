import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import ollama

from backend.config.settings import LLM_MODEL_PATH, LLM_QUANTIZATION

logger = logging.getLogger(__name__)

class BiomechLLM:
    """Biomechanical LLM model loader and interface"""
    
    def __init__(
        self,
        model_path: str = LLM_MODEL_PATH,
        quantization: str = LLM_QUANTIZATION,
        use_ollama: bool = True,
    ):
        self.model_path = model_path
        self.quantization = quantization
        self.use_ollama = use_ollama
        self.model = None
        self.model_info = {}
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the biomechanical LLM"""
        try:
            if self.use_ollama:
                logger.info(f"Initializing Ollama model from {self.model_path}")
                # Check if model exists in Ollama
                models = ollama.list()
                model_name = Path(self.model_path).name
                
                if not any(m['name'] == model_name for m in models.get('models', [])):
                    # Import the model to Ollama
                    logger.info(f"Importing model {model_name} to Ollama")
                    ollama.create(
                        model=model_name,
                        path=self.model_path,
                        quantization=self.quantization
                    )
                
                # Set the model reference
                self.model = model_name
                self.model_info = {
                    'name': model_name,
                    'quantization': self.quantization,
                    'type': 'ollama'
                }
            else:
                # Load using direct model loading (PyTorch)
                logger.info(f"Loading model directly from {self.model_path}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = torch.load(self.model_path, map_location=device)
                self.model_info = {
                    'name': Path(self.model_path).name,
                    'device': device,
                    'type': 'torch'
                }
            
            logger.info(f"Successfully loaded biomechanical LLM: {self.model_info}")
        except Exception as e:
            logger.error(f"Failed to load biomechanical LLM: {e}")
            raise RuntimeError(f"Failed to load biomechanical LLM: {e}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from the biomechanical LLM"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            if self.use_ollama:
                # Generate using Ollama
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': temperature,
                        'top_p': top_p,
                        'num_predict': max_tokens,
                        **kwargs
                    }
                )
                return {
                    'text': response['response'],
                    'model': self.model_info['name'],
                    'usage': {
                        'prompt_tokens': response.get('prompt_eval_count', 0),
                        'completion_tokens': response.get('eval_count', 0),
                        'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                    }
                }
            else:
                # Generate using PyTorch model
                # Implementation depends on the specific model format
                # This is a placeholder for direct model inference
                outputs = self.model.generate(
                    prompt=prompt,
                    max_length=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
                
                return {
                    'text': outputs,
                    'model': self.model_info['name'],
                    'usage': {}  # Would need custom tracking for PyTorch model
                }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise RuntimeError(f"Error generating response: {e}")

# Singleton instance for reuse
_biomech_llm_instance = None

def get_biomech_llm() -> BiomechLLM:
    """Get or create the biomechanical LLM instance"""
    global _biomech_llm_instance
    if _biomech_llm_instance is None:
        _biomech_llm_instance = BiomechLLM()
    return _biomech_llm_instance
