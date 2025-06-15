"""
Core model abstractions and registry for the Diadochi framework.

This module provides the base Model interface and ModelRegistry for managing
multiple LLM instances from different providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from a model with metadata."""
    content: str
    model_name: str
    domain: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Model(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: str, domain: Optional[str] = None):
        self.name = name
        self.domain = domain
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response to the given prompt."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the underlying model name."""
        pass
    
    def supports_embedding(self) -> bool:
        """Check if this model supports embedding generation."""
        try:
            self.embed("test")
            return True
        except NotImplementedError:
            return False


class OpenAIModel(Model):
    """OpenAI model implementation."""
    
    def __init__(self, name: str, model_name: str = "gpt-4", api_key: Optional[str] = None, 
                 domain: Optional[str] = None, **kwargs):
        super().__init__(name, domain)
        self._model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            
            content = response.choices[0].message.content
            return ModelResponse(
                content=content,
                model_name=self._model_name,
                domain=self.domain,
                metadata={"usage": response.usage.dict() if response.usage else None}
            )
        except Exception as e:
            logger.error(f"Error generating response with OpenAI model {self._model_name}: {e}")
            raise
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise
    
    @property
    def model_name(self) -> str:
        return self._model_name


class AnthropicModel(Model):
    """Anthropic Claude model implementation."""
    
    def __init__(self, name: str, model_name: str = "claude-3-sonnet-20240229", 
                 api_key: Optional[str] = None, domain: Optional[str] = None, **kwargs):
        super().__init__(name, domain)
        self._model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self._model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            
            content = response.content[0].text
            return ModelResponse(
                content=content,
                model_name=self._model_name,
                domain=self.domain,
                metadata={"usage": {"input_tokens": response.usage.input_tokens, 
                                  "output_tokens": response.usage.output_tokens}}
            )
        except Exception as e:
            logger.error(f"Error generating response with Anthropic model {self._model_name}: {e}")
            raise
    
    def embed(self, text: str) -> List[float]:
        """Anthropic doesn't provide embeddings - raise NotImplementedError."""
        raise NotImplementedError("Anthropic models don't support embeddings")
    
    @property
    def model_name(self) -> str:
        return self._model_name


class OllamaModel(Model):
    """Ollama local model implementation."""
    
    def __init__(self, name: str, model_name: str, base_url: str = "http://localhost:11434", 
                 domain: Optional[str] = None, **kwargs):
        super().__init__(name, domain)
        self._model_name = model_name
        self.base_url = base_url
        self.kwargs = kwargs
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError("Ollama library not installed. Install with: pip install ollama")
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using Ollama API."""
        try:
            response = self.client.chat(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            
            content = response['message']['content']
            return ModelResponse(
                content=content,
                model_name=self._model_name,
                domain=self.domain,
                metadata={"done": response.get('done', False)}
            )
        except Exception as e:
            logger.error(f"Error generating response with Ollama model {self._model_name}: {e}")
            raise
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings using Ollama API."""
        try:
            response = self.client.embeddings(
                model=self._model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embeddings with Ollama: {e}")
            raise
    
    @property
    def model_name(self) -> str:
        return self._model_name


class ModelRegistry:
    """Registry for managing multiple models."""
    
    def __init__(self):
        self.models: Dict[str, Model] = {}
        self.domains: Dict[str, List[str]] = {}  # domain -> list of model names
    
    def add_model(self, name: str, engine: str, model_name: str, domain: Optional[str] = None, **kwargs) -> None:
        """Register a new model."""
        if engine == "openai":
            model = OpenAIModel(name=name, model_name=model_name, domain=domain, **kwargs)
        elif engine == "anthropic":
            model = AnthropicModel(name=name, model_name=model_name, domain=domain, **kwargs)
        elif engine == "ollama":
            model = OllamaModel(name=name, model_name=model_name, domain=domain, **kwargs)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
        
        self.models[name] = model
        
        # Track domain associations
        if domain:
            if domain not in self.domains:
                self.domains[domain] = []
            self.domains[domain].append(name)
        
        logger.info(f"Registered model '{name}' with engine '{engine}' for domain '{domain}'")
    
    def get(self, name: str) -> Model:
        """Retrieve a registered model by name."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self.models[name]
    
    def get_by_domain(self, domain: str) -> List[Model]:
        """Get all models for a specific domain."""
        if domain not in self.domains:
            return []
        return [self.models[name] for name in self.domains[domain]]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def list_domains(self) -> List[str]:
        """List all registered domains."""
        return list(self.domains.keys())
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the registry."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in registry")
        
        model = self.models[name]
        domain = model.domain
        
        # Remove from models dict
        del self.models[name]
        
        # Remove from domain tracking
        if domain and domain in self.domains:
            self.domains[domain].remove(name)
            if not self.domains[domain]:  # Remove empty domain
                del self.domains[domain]
        
        logger.info(f"Removed model '{name}' from registry")
    
    def get_models_dict(self) -> Dict[str, Model]:
        """Get all models as a dictionary."""
        return self.models.copy() 