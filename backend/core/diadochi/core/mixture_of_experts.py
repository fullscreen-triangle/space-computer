"""
Mixture of Experts implementation for parallel processing with weighted combination.

This module processes queries through multiple domain experts in parallel
and combines their outputs based on relevance or confidence.
"""

from typing import Dict, List, Optional, Any, Callable
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models import Model, ModelRegistry, ModelResponse
from .mixers import Mixer, SynthesisMixer, WeightedResponse

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """Base class for estimating model confidence for queries."""
    
    def estimate(self, query: str, models: List[Model]) -> Dict[str, float]:
        """
        Estimate confidence scores for each model given a query.
        
        Args:
            query: Input query
            models: List of models to evaluate
            
        Returns:
            Dict mapping model names to confidence scores (0-1)
        """
        raise NotImplementedError


class EmbeddingConfidenceEstimator(ConfidenceEstimator):
    """Confidence estimator based on embedding similarity."""
    
    def __init__(self, embedding_model: Model, temperature: float = 1.0):
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.domain_descriptions: Dict[str, str] = {}
        self.domain_embeddings: Dict[str, List[float]] = {}
    
    def add_domain(self, domain: str, description: str) -> None:
        """Add domain description for confidence estimation."""
        self.domain_descriptions[domain] = description
        try:
            embedding = self.embedding_model.embed(description)
            self.domain_embeddings[domain] = embedding
            logger.info(f"Added domain '{domain}' for confidence estimation")
        except Exception as e:
            logger.error(f"Failed to generate embedding for domain '{domain}': {e}")
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        if len(a) != len(b):
            return 0.0
        
        a_np = np.array(a)
        b_np = np.array(b)
        
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def _softmax(self, scores: List[float]) -> List[float]:
        """Apply softmax with temperature to scores."""
        import numpy as np
        
        if not scores:
            return []
        
        # Apply temperature scaling
        scaled_scores = [score / self.temperature for score in scores]
        
        # Compute softmax
        exp_scores = np.exp(np.array(scaled_scores) - np.max(scaled_scores))
        return (exp_scores / np.sum(exp_scores)).tolist()
    
    def estimate(self, query: str, models: List[Model]) -> Dict[str, float]:
        """Estimate confidence based on embedding similarity."""
        if not self.domain_embeddings:
            # No domain embeddings available, return equal confidence
            return {model.name: 1.0 / len(models) for model in models}
        
        try:
            query_embedding = self.embedding_model.embed(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return {model.name: 1.0 / len(models) for model in models}
        
        # Calculate similarities for each model's domain
        similarities = []
        model_names = []
        
        for model in models:
            model_names.append(model.name)
            
            if model.domain and model.domain in self.domain_embeddings:
                domain_embedding = self.domain_embeddings[model.domain]
                similarity = self._cosine_similarity(query_embedding, domain_embedding)
                similarities.append(max(similarity, 0.01))  # Minimum similarity
            else:
                similarities.append(0.01)  # Default low similarity
        
        # Apply softmax to get confidence scores
        confidence_scores = self._softmax(similarities)
        
        return dict(zip(model_names, confidence_scores))


class KeywordConfidenceEstimator(ConfidenceEstimator):
    """Confidence estimator based on keyword matching."""
    
    def __init__(self):
        self.domain_keywords: Dict[str, List[str]] = {}
        self.keyword_weights: Dict[str, Dict[str, float]] = {}
    
    def add_domain(self, domain: str, keywords: List[str], weights: Optional[Dict[str, float]] = None) -> None:
        """Add domain keywords for confidence estimation."""
        self.domain_keywords[domain] = [kw.lower() for kw in keywords]
        
        if weights:
            self.keyword_weights[domain] = {kw.lower(): weight for kw, weight in weights.items()}
        else:
            self.keyword_weights[domain] = {kw.lower(): 1.0 for kw in keywords}
    
    def estimate(self, query: str, models: List[Model]) -> Dict[str, float]:
        """Estimate confidence based on keyword matching."""
        query_lower = query.lower()
        domain_scores = {}
        
        # Calculate scores for each domain
        for domain, keywords in self.domain_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in query_lower:
                    weight = self.keyword_weights[domain].get(keyword, 1.0)
                    score += weight
            domain_scores[domain] = score
        
        # Map to model confidence scores
        model_confidences = {}
        total_score = sum(domain_scores.values())
        
        for model in models:
            if model.domain and model.domain in domain_scores:
                if total_score > 0:
                    confidence = domain_scores[model.domain] / total_score
                else:
                    confidence = 1.0 / len(models)
            else:
                confidence = 1.0 / len(models)  # Default equal weight
            
            model_confidences[model.name] = confidence
        
        return model_confidences


class MixtureOfExperts:
    """Mixture of Experts that processes queries through multiple experts in parallel."""
    
    def __init__(self, models: ModelRegistry, confidence_estimator: ConfidenceEstimator,
                 mixer: Mixer, threshold: float = 0.1, max_experts: int = 5):
        """
        Initialize the Mixture of Experts.
        
        Args:
            models: Registry of available models
            confidence_estimator: Component to estimate model relevance
            mixer: Component to combine expert responses
            threshold: Minimum confidence threshold for inclusion
            max_experts: Maximum number of experts to use
        """
        self.models = models
        self.confidence_estimator = confidence_estimator
        self.mixer = mixer
        self.threshold = threshold
        self.max_experts = max_experts
    
    def _select_experts(self, query: str) -> List[tuple[Model, float]]:
        """Select experts based on confidence estimation."""
        available_models = [self.models.get(name) for name in self.models.list_models()]
        
        if not available_models:
            raise ValueError("No models available in registry")
        
        # Get confidence scores
        confidence_scores = self.confidence_estimator.estimate(query, available_models)
        
        # Filter by threshold and sort by confidence
        expert_candidates = []
        for model in available_models:
            confidence = confidence_scores.get(model.name, 0.0)
            if confidence >= self.threshold:
                expert_candidates.append((model, confidence))
        
        # Sort by confidence (highest first) and limit to max_experts
        expert_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_experts = expert_candidates[:self.max_experts]
        
        if not selected_experts:
            # If no experts meet threshold, use the best one
            best_model = max(available_models, key=lambda m: confidence_scores.get(m.name, 0.0))
            best_confidence = confidence_scores.get(best_model.name, 0.0)
            selected_experts = [(best_model, best_confidence)]
        
        logger.info(f"Selected {len(selected_experts)} experts for query")
        return selected_experts
    
    def _generate_response_sync(self, model: Model, query: str, **kwargs) -> tuple[Model, ModelResponse, Exception]:
        """Generate response from a single model (synchronous)."""
        try:
            response = model.generate(query, **kwargs)
            return model, response, None
        except Exception as e:
            logger.error(f"Error generating response from {model.name}: {e}")
            return model, None, e
    
    def generate(self, query: str, parallel: bool = True, **kwargs) -> ModelResponse:
        """
        Generate response using mixture of experts.
        
        Args:
            query: Input query
            parallel: Whether to process experts in parallel
            **kwargs: Additional arguments passed to model generation
        """
        # Select experts
        selected_experts = self._select_experts(query)
        
        if parallel and len(selected_experts) > 1:
            # Parallel processing
            weighted_responses = []
            
            with ThreadPoolExecutor(max_workers=min(len(selected_experts), 4)) as executor:
                # Submit all tasks
                future_to_expert = {
                    executor.submit(self._generate_response_sync, model, query, **kwargs): (model, confidence)
                    for model, confidence in selected_experts
                }
                
                # Collect results
                for future in as_completed(future_to_expert):
                    model, confidence = future_to_expert[future]
                    model_result, response, error = future.result()
                    
                    if response is not None:
                        weighted_response = WeightedResponse(
                            response=response,
                            weight=confidence,
                            metadata={"expert_confidence": confidence}
                        )
                        weighted_responses.append(weighted_response)
        else:
            # Sequential processing
            weighted_responses = []
            for model, confidence in selected_experts:
                try:
                    response = model.generate(query, **kwargs)
                    weighted_response = WeightedResponse(
                        response=response,
                        weight=confidence,
                        metadata={"expert_confidence": confidence}
                    )
                    weighted_responses.append(weighted_response)
                except Exception as e:
                    logger.error(f"Error generating response from {model.name}: {e}")
                    continue
        
        if not weighted_responses:
            raise ValueError("No successful responses from expert models")
        
        # Mix the responses
        mixed_response = self.mixer.mix(query, weighted_responses)
        
        # Add MoE metadata
        mixed_response.metadata = mixed_response.metadata or {}
        mixed_response.metadata.update({
            "mixture_of_experts": True,
            "num_experts": len(weighted_responses),
            "parallel_processing": parallel,
            "threshold": self.threshold,
            "expert_info": [
                {
                    "model_name": wr.response.model_name,
                    "domain": wr.response.domain,
                    "confidence": wr.weight
                }
                for wr in weighted_responses
            ]
        })
        
        return mixed_response
    
    def add_model(self, name: str, engine: str, model_name: str, domain: Optional[str] = None, **kwargs) -> None:
        """Add a new model to the mixture of experts."""
        self.models.add_model(name, engine, model_name, domain, **kwargs)
    
    def list_models(self) -> List[str]:
        """List all available models."""
        return self.models.list_models()
    
    def get_confidence_scores(self, query: str) -> Dict[str, float]:
        """Get confidence scores for all models given a query."""
        available_models = [self.models.get(name) for name in self.models.list_models()]
        return self.confidence_estimator.estimate(query, available_models) 