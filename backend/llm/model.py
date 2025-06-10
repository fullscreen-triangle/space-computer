"""
LLM module for Spectacular.
Implements LLM functionality with an option to use Meta-Llama-3-8B-Instruct.
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
import os

class LLMProcessor:
    """
    LLM processor with support for Mistral-7B and Meta-Llama-3-8B-Instruct models.
    """
    
    def __init__(self, device: str = None, use_llama3: bool = False):
        """
        Initialize the LLM processor.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', etc.)
            use_llama3: Whether to use Meta-Llama-3-8B-Instruct instead of Mistral-7B
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_llama3 = use_llama3
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if self.use_llama3:
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            print(f"Loading {model_name} model...")
            
            # Check if we have the model access
            try:
                # Load tokenizer and model with 4-bit quantization for efficiency
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=self.device,
                    load_in_4bit=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            except Exception as e:
                print(f"Error loading Llama-3: {e}")
                print("Falling back to Mistral-7B...")
                self.use_llama3 = False
                self._load_mistral()
        else:
            self._load_mistral()
    
    def _load_mistral(self):
        """Load the Mistral-7B model as fallback."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print(f"Loading {model_name} model...")
        
        # Load tokenizer and model with 4-bit quantization for efficiency
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            load_in_4bit=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                 system_prompt: Optional[str] = None) -> str:
        """
        Generate text using the loaded LLM.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt for instruction
            
        Returns:
            Generated text response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Initialize the LLMProcessor first.")
        
        # Format prompt based on model type
        if self.use_llama3:
            # Llama-3 chat format
            if system_prompt:
                formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
            else:
                formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
        else:
            # Mistral chat format
            if system_prompt:
                formatted_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
            else:
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                repetition_penalty=1.1
            )
        
        # Decode and clean up response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if self.use_llama3:
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[1].strip()
        else:
            if "[/INST]" in response:
                response = response.split("[/INST]")[1].strip()
        
        return response
    
    def generate_with_context(self, prompt: str, context: str, max_tokens: int = 512) -> str:
        """
        Generate text with additional context.
        
        Args:
            prompt: User prompt
            context: Additional context information
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        system_prompt = "You are a helpful AI assistant with expertise in sports and movement analysis. Use the context information provided to give accurate and insightful responses."
        
        # Combine context and prompt
        full_prompt = f"Context information:\n{context}\n\nQuestion: {prompt}"
        
        return self.generate(full_prompt, max_tokens, system_prompt=system_prompt) 