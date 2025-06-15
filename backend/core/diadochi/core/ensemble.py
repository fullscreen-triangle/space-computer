"""
Ensemble implementation for router-based model selection.

This module provides router-based ensemble systems that direct queries
to the most appropriate domain expert models.
"""

from typing import Dict, List, Optional, Any
import logging
from .models import Model, ModelRegistry, ModelResponse
from .routers import Router, RoutingResult
from .mixers import Mixer, DefaultMixer, WeightedResponse

logger = logging.getLogger(__name__)


class Ensemble:
    """Router-based ensemble that directs queries to appropriate experts."""
    
    def __init__(self, router: Router, models: ModelRegistry, 
                 default_model: Optional[str] = None, mixer: Optional[Mixer] = None):
        """
        Initialize the ensemble.
        
        Args:
            router: Router to determine which model(s) to use
            models: Registry of available models
            default_model: Default model name if routing fails
            mixer: Optional mixer for combining multiple responses
        """
        self.router = router
        self.models = models
        self.default_model = default_model
        self.mixer = mixer or DefaultMixer()
    
    def generate(self, query: str, top_k: int = 1, **kwargs) -> ModelResponse:
        """
        Generate response using router-based ensemble.
        
        Args:
            query: Input query
            top_k: Number of top models to use (1 for single routing, >1 for mixing)
            **kwargs: Additional arguments passed to model generation
        """
        available_models = self.models.list_models()
        
        if not available_models:
            raise ValueError("No models available in registry")
        
        if top_k == 1:
            # Single model routing
            routing_result = self.router.route(query, available_models)
            
            if routing_result is None:
                # Fall back to default model
                if self.default_model and self.default_model in available_models:
                    model = self.models.get(self.default_model)
                    response = model.generate(query, **kwargs)
                    response.metadata = response.metadata or {}
                    response.metadata["routing_failed"] = True
                    response.metadata["used_default"] = True
                    return response
                else:
                    raise ValueError("Routing failed and no valid default model available")
            
            # Use routed model
            model = self.models.get(routing_result.model_name)
            response = model.generate(query, **kwargs)
            
            # Add routing metadata
            response.metadata = response.metadata or {}
            response.metadata["routing_result"] = {
                "model_name": routing_result.model_name,
                "confidence": routing_result.confidence,
                "domain": routing_result.domain,
                "routing_metadata": routing_result.metadata
            }
            
            return response
        
        else:
            # Multiple model routing with mixing
            routing_results = self.router.route_multiple(query, available_models, top_k)
            
            if not routing_results:
                # Fall back to default model
                if self.default_model and self.default_model in available_models:
                    model = self.models.get(self.default_model)
                    response = model.generate(query, **kwargs)
                    response.metadata = response.metadata or {}
                    response.metadata["routing_failed"] = True
                    response.metadata["used_default"] = True
                    return response
                else:
                    raise ValueError("Multi-routing failed and no valid default model available")
            
            # Generate responses from all routed models
            weighted_responses = []
            for routing_result in routing_results:
                try:
                    model = self.models.get(routing_result.model_name)
                    response = model.generate(query, **kwargs)
                    
                    weighted_response = WeightedResponse(
                        response=response,
                        weight=routing_result.confidence,
                        metadata=routing_result.metadata
                    )
                    weighted_responses.append(weighted_response)
                    
                except Exception as e:
                    logger.error(f"Error generating response from {routing_result.model_name}: {e}")
                    continue
            
            if not weighted_responses:
                raise ValueError("No successful responses from routed models")
            
            # Mix the responses
            mixed_response = self.mixer.mix(query, weighted_responses)
            
            # Add ensemble metadata
            mixed_response.metadata = mixed_response.metadata or {}
            mixed_response.metadata["ensemble_type"] = "router_based"
            mixed_response.metadata["top_k"] = top_k
            mixed_response.metadata["routing_results"] = [
                {
                    "model_name": r.model_name,
                    "confidence": r.confidence,
                    "domain": r.domain
                }
                for r in routing_results
            ]
            
            return mixed_response
    
    def add_model(self, name: str, engine: str, model_name: str, domain: Optional[str] = None, **kwargs) -> None:
        """Add a new model to the ensemble."""
        self.models.add_model(name, engine, model_name, domain, **kwargs)
    
    def list_models(self) -> List[str]:
        """List all available models in the ensemble."""
        return self.models.list_models()
    
    def list_domains(self) -> List[str]:
        """List all available domains in the ensemble."""
        return self.models.list_domains()
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all models in the ensemble."""
        model_info = {}
        for model_name in self.models.list_models():
            model = self.models.get(model_name)
            model_info[model_name] = {
                "name": model.name,
                "domain": model.domain,
                "model_name": model.model_name,
                "supports_embedding": model.supports_embedding()
            }
        return model_info 