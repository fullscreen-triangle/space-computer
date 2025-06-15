"""
Router implementations for directing queries to appropriate domain experts.

This module provides various routing strategies for determining which domain
expert models should handle specific queries.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import logging
import re
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of routing operation."""
    model_name: str
    confidence: float
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Router(ABC):
    """Abstract base class for all routers."""
    
    @abstractmethod
    def route(self, query: str, available_models: List[str]) -> Optional[RoutingResult]:
        """Route a query to the most appropriate model."""
        pass
    
    def route_multiple(self, query: str, available_models: List[str], k: int = 3) -> List[RoutingResult]:
        """Route a query to the k most appropriate models."""
        # Default implementation - subclasses can override for more efficient implementations
        results = []
        for model in available_models:
            result = self.route(query, [model])
            if result:
                results.append(result)
        
        # Sort by confidence and return top k
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:k]


class KeywordRouter(Router):
    """Router that uses keyword matching to determine domain."""
    
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        self.domain_keywords: Dict[str, List[str]] = {}
        self.keyword_weights: Dict[str, Dict[str, float]] = {}
    
    def add_domain(self, domain: str, keywords: List[str], weights: Optional[Dict[str, float]] = None) -> None:
        """Add domain with associated keywords and optional weights."""
        if not self.case_sensitive:
            keywords = [kw.lower() for kw in keywords]
        
        self.domain_keywords[domain] = keywords
        
        # Set weights (default to 1.0 for all keywords)
        if weights:
            self.keyword_weights[domain] = {
                (kw.lower() if not self.case_sensitive else kw): weight 
                for kw, weight in weights.items()
            }
        else:
            self.keyword_weights[domain] = {kw: 1.0 for kw in keywords}
        
        logger.info(f"Added domain '{domain}' with {len(keywords)} keywords")
    
    def route(self, query: str, available_models: List[str]) -> Optional[RoutingResult]:
        """Route query based on keyword matching."""
        if not self.case_sensitive:
            query_normalized = query.lower()
        else:
            query_normalized = query
        
        # Calculate scores for each domain
        domain_scores: Dict[str, float] = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                # Use word boundaries for more precise matching
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, query_normalized))
                
                if matches > 0:
                    weight = self.keyword_weights[domain].get(keyword, 1.0)
                    score += matches * weight
                    matched_keywords.append(keyword)
            
            if score > 0:
                domain_scores[domain] = score
                logger.debug(f"Domain '{domain}' scored {score} with keywords: {matched_keywords}")
        
        if not domain_scores:
            return None
        
        # Find the domain with highest score
        best_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])
        best_score = domain_scores[best_domain]
        
        # Find best model for this domain among available models
        best_model = None
        for model in available_models:
            if best_domain in model.lower():
                best_model = model
                break
        
        if not best_model:
            best_model = available_models[0] if available_models else None
        
        if best_model:
            # Normalize confidence to 0-1 range (simple heuristic)
            max_possible_score = sum(self.keyword_weights[best_domain].values())
            confidence = min(best_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
            
            return RoutingResult(
                model_name=best_model,
                confidence=confidence,
                domain=best_domain,
                metadata={"score": best_score, "domain_scores": domain_scores}
            )
        
        return None


class EmbeddingRouter(Router):
    """Router that uses embedding similarity to determine domain."""
    
    def __init__(self, embedding_model, threshold: float = 0.7, temperature: float = 1.0):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.temperature = temperature
        self.domain_descriptions: Dict[str, str] = {}
        self.domain_embeddings: Dict[str, List[float]] = {}
    
    def add_domain(self, domain: str, description: str) -> None:
        """Add domain with description for embedding-based routing."""
        self.domain_descriptions[domain] = description
        
        # Generate embedding for domain description
        try:
            embedding = self.embedding_model.embed(description)
            self.domain_embeddings[domain] = embedding
            logger.info(f"Added domain '{domain}' with embedding")
        except Exception as e:
            logger.error(f"Failed to generate embedding for domain '{domain}': {e}")
            raise
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            raise ValueError("Vectors must have the same length")
        
        a_np = np.array(a)
        b_np = np.array(b)
        
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _softmax(self, scores: List[float]) -> List[float]:
        """Apply softmax with temperature to scores."""
        if not scores:
            return []
        
        # Apply temperature scaling
        scaled_scores = [score / self.temperature for score in scores]
        
        # Compute softmax
        exp_scores = np.exp(np.array(scaled_scores) - np.max(scaled_scores))  # Subtract max for numerical stability
        return (exp_scores / np.sum(exp_scores)).tolist()
    
    def route(self, query: str, available_models: List[str]) -> Optional[RoutingResult]:
        """Route query based on embedding similarity."""
        if not self.domain_embeddings:
            logger.warning("No domain embeddings available for routing")
            return None
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None
        
        # Calculate similarities with all domains
        similarities = {}
        for domain, domain_embedding in self.domain_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, domain_embedding)
            similarities[domain] = similarity
        
        # Find best domain
        best_domain = max(similarities.keys(), key=lambda d: similarities[d])
        best_similarity = similarities[best_domain]
        
        # Check if similarity meets threshold
        if best_similarity < self.threshold:
            logger.debug(f"Best similarity {best_similarity} below threshold {self.threshold}")
            return None
        
        # Find best model for this domain among available models
        best_model = None
        for model in available_models:
            if best_domain in model.lower():
                best_model = model
                break
        
        if not best_model:
            best_model = available_models[0] if available_models else None
        
        if best_model:
            return RoutingResult(
                model_name=best_model,
                confidence=best_similarity,
                domain=best_domain,
                metadata={"similarities": similarities}
            )
        
        return None
    
    def route_multiple(self, query: str, available_models: List[str], k: int = 3) -> List[RoutingResult]:
        """Route query to k most similar domains."""
        if not self.domain_embeddings:
            return []
        
        try:
            query_embedding = self.embedding_model.embed(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
        
        # Calculate similarities
        similarities = {}
        for domain, domain_embedding in self.domain_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, domain_embedding)
            similarities[domain] = similarity
        
        # Sort domains by similarity
        sorted_domains = sorted(similarities.keys(), key=lambda d: similarities[d], reverse=True)
        
        # Apply softmax to get confidence scores
        similarity_scores = [similarities[domain] for domain in sorted_domains]
        confidence_scores = self._softmax(similarity_scores)
        
        results = []
        for i, domain in enumerate(sorted_domains[:k]):
            if similarities[domain] >= self.threshold:
                # Find model for this domain
                best_model = None
                for model in available_models:
                    if domain in model.lower():
                        best_model = model
                        break
                
                if not best_model and available_models:
                    best_model = available_models[0]
                
                if best_model:
                    results.append(RoutingResult(
                        model_name=best_model,
                        confidence=confidence_scores[i],
                        domain=domain,
                        metadata={"raw_similarity": similarities[domain], "all_similarities": similarities}
                    ))
        
        return results


class LLMRouter(Router):
    """Router that uses an LLM to determine the best domain."""
    
    def __init__(self, router_model, domains: List[str], prompt_template: Optional[str] = None):
        self.router_model = router_model
        self.domains = domains
        self.prompt_template = prompt_template or self._default_prompt_template()
    
    def _default_prompt_template(self) -> str:
        """Default prompt template for LLM routing."""
        return """Analyze this query and determine which domain it belongs to.

Query: {query}

Available domains:
{domains}

Respond with ONLY the domain name that best matches the query. If no domain is a good match, respond with "NONE".

Domain:"""
    
    def route(self, query: str, available_models: List[str]) -> Optional[RoutingResult]:
        """Route query using LLM analysis."""
        # Format domains list
        domains_list = "\n".join([f"- {domain}" for domain in self.domains])
        
        # Create prompt
        prompt = self.prompt_template.format(
            query=query,
            domains=domains_list
        )
        
        try:
            # Get LLM response
            response = self.router_model.generate(prompt)
            predicted_domain = response.content.strip()
            
            # Validate domain
            if predicted_domain == "NONE" or predicted_domain not in self.domains:
                logger.debug(f"LLM router returned invalid domain: {predicted_domain}")
                return None
            
            # Find best model for this domain
            best_model = None
            for model in available_models:
                if predicted_domain in model.lower():
                    best_model = model
                    break
            
            if not best_model and available_models:
                best_model = available_models[0]
            
            if best_model:
                # LLM routing doesn't provide explicit confidence, use 0.8 as default
                return RoutingResult(
                    model_name=best_model,
                    confidence=0.8,
                    domain=predicted_domain,
                    metadata={"llm_response": response.content}
                )
        
        except Exception as e:
            logger.error(f"Error in LLM routing: {e}")
            return None
        
        return None


class CompositeRouter(Router):
    """Router that combines multiple routing strategies."""
    
    def __init__(self, routers: List[Tuple[Router, float]]):
        """
        Initialize with list of (router, weight) tuples.
        Weights should sum to 1.0.
        """
        self.routers = routers
        total_weight = sum(weight for _, weight in routers)
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Router weights sum to {total_weight}, not 1.0")
    
    def route(self, query: str, available_models: List[str]) -> Optional[RoutingResult]:
        """Route using weighted combination of multiple routers."""
        # Get results from all routers
        router_results = []
        for router, weight in self.routers:
            try:
                result = router.route(query, available_models)
                if result:
                    router_results.append((result, weight))
            except Exception as e:
                logger.error(f"Error in composite router component: {e}")
                continue
        
        if not router_results:
            return None
        
        # Aggregate results by model/domain
        model_scores: Dict[str, float] = {}
        model_metadata: Dict[str, Dict[str, Any]] = {}
        
        for result, weight in router_results:
            key = f"{result.model_name}_{result.domain}"
            weighted_confidence = result.confidence * weight
            
            if key in model_scores:
                model_scores[key] += weighted_confidence
            else:
                model_scores[key] = weighted_confidence
                model_metadata[key] = {
                    "model_name": result.model_name,
                    "domain": result.domain,
                    "component_results": []
                }
            
            model_metadata[key]["component_results"].append({
                "router_type": type(result).__name__,
                "confidence": result.confidence,
                "weight": weight,
                "metadata": result.metadata
            })
        
        # Find best result
        best_key = max(model_scores.keys(), key=lambda k: model_scores[k])
        best_score = model_scores[best_key]
        best_meta = model_metadata[best_key]
        
        return RoutingResult(
            model_name=best_meta["model_name"],
            confidence=best_score,
            domain=best_meta["domain"],
            metadata={
                "composite_score": best_score,
                "all_scores": model_scores,
                "component_results": best_meta["component_results"]
            }
        ) 