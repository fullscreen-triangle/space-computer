"""
Evaluation metrics for multi-domain AI systems.

This module provides specialized metrics for evaluating the performance
of domain-expert combinations and cross-domain integration.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import statistics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of an evaluation metric."""
    score: float
    details: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class EvaluationMetric(ABC):
    """Base class for evaluation metrics."""
    
    @abstractmethod
    def evaluate(self, queries: List[str], responses: List[str], 
                ground_truth: Optional[List[str]] = None, **kwargs) -> EvaluationResult:
        """Evaluate the given responses."""
        pass


class DomainExpertiseRetention(EvaluationMetric):
    """
    Measures how well a multi-domain system maintains expertise in individual domains
    compared to domain-specific models.
    """
    
    def __init__(self, domain_models: Dict[str, Any]):
        self.domain_models = domain_models
    
    def evaluate(self, queries: List[str], responses: List[str], 
                ground_truth: Optional[List[str]] = None, 
                domain_labels: Optional[List[str]] = None, **kwargs) -> EvaluationResult:
        """Evaluate domain expertise retention."""
        if len(queries) != len(responses):
            raise ValueError("Queries and responses must have the same length")
        
        # Group by domain
        domain_groups = {}
        for i, (query, response) in enumerate(zip(queries, responses)):
            domain = domain_labels[i] if domain_labels else "general"
            if domain not in domain_groups:
                domain_groups[domain] = {"queries": [], "responses": []}
            
            domain_groups[domain]["queries"].append(query)
            domain_groups[domain]["responses"].append(response)
        
        # Calculate retention scores for each domain
        domain_scores = {}
        overall_retention = []
        
        for domain, group_data in domain_groups.items():
            if domain in self.domain_models:
                # Compare with domain expert (simplified)
                score = self._compare_with_domain_expert(group_data, domain)
                domain_scores[domain] = score
                overall_retention.append(score)
            else:
                domain_scores[domain] = 0.0
                overall_retention.append(0.0)
        
        overall_score = statistics.mean(overall_retention) if overall_retention else 0.0
        
        return EvaluationResult(
            score=overall_score,
            details={
                "domain_scores": domain_scores,
                "num_domains": len(domain_groups)
            },
            metadata={"metric": "domain_expertise_retention"}
        )
    
    def _compare_with_domain_expert(self, group_data: Dict, domain: str) -> float:
        """Compare responses with domain expert (simplified)."""
        # Simplified comparison - in practice would use more sophisticated methods
        return 0.8  # Placeholder score


class CrossDomainAccuracy(EvaluationMetric):
    """Measures accuracy on cross-domain queries."""
    
    def evaluate(self, queries: List[str], responses: List[str], 
                ground_truth: Optional[List[str]] = None, **kwargs) -> EvaluationResult:
        """Evaluate cross-domain accuracy."""
        if len(queries) != len(responses):
            raise ValueError("Queries and responses must have the same length")
        
        # Identify cross-domain queries (simplified heuristic)
        cross_domain_indices = self._identify_cross_domain_queries(queries)
        
        if not cross_domain_indices:
            return EvaluationResult(
                score=0.0,
                details={"num_cross_domain": 0},
                metadata={"metric": "cross_domain_accuracy"}
            )
        
        # Evaluate cross-domain responses
        accuracy_scores = []
        for i in cross_domain_indices:
            score = self._evaluate_cross_domain_response(queries[i], responses[i])
            accuracy_scores.append(score)
        
        overall_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.0
        
        return EvaluationResult(
            score=overall_accuracy,
            details={
                "num_cross_domain": len(cross_domain_indices),
                "total_queries": len(queries)
            },
            metadata={"metric": "cross_domain_accuracy"}
        )
    
    def _identify_cross_domain_queries(self, queries: List[str]) -> List[int]:
        """Identify queries that span multiple domains."""
        # Simplified heuristic
        cross_domain_indices = []
        domain_keywords = {
            "biomechanics": ["movement", "force", "velocity"],
            "physiology": ["muscle", "energy", "oxygen"],
            "nutrition": ["diet", "protein", "supplements"]
        }
        
        for i, query in enumerate(queries):
            query_lower = query.lower()
            domains_mentioned = sum(
                1 for keywords in domain_keywords.values()
                if any(kw in query_lower for kw in keywords)
            )
            if domains_mentioned >= 2:
                cross_domain_indices.append(i)
        
        return cross_domain_indices
    
    def _evaluate_cross_domain_response(self, query: str, response: str) -> float:
        """Evaluate a single cross-domain response."""
        # Simplified evaluation
        return 0.75  # Placeholder score


class IntegrationCoherence(EvaluationMetric):
    """Measures logical consistency of integrated responses."""
    
    def evaluate(self, queries: List[str], responses: List[str], 
                ground_truth: Optional[List[str]] = None, **kwargs) -> EvaluationResult:
        """Evaluate integration coherence."""
        if len(queries) != len(responses):
            raise ValueError("Queries and responses must have the same length")
        
        coherence_scores = []
        
        for query, response in zip(queries, responses):
            score = self._assess_coherence(query, response)
            coherence_scores.append(score)
        
        overall_coherence = statistics.mean(coherence_scores) if coherence_scores else 0.0
        
        return EvaluationResult(
            score=overall_coherence,
            details={
                "individual_scores": coherence_scores,
                "num_responses": len(responses)
            },
            metadata={"metric": "integration_coherence"}
        )
    
    def _assess_coherence(self, query: str, response: str) -> float:
        """Assess coherence of a single response."""
        if not response.strip():
            return 0.0
        
        # Simple coherence indicators
        coherence_factors = []
        
        # Keyword overlap with query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0.0
        coherence_factors.append(min(overlap * 2, 1.0))
        
        # Response length (not too short, not too long)
        word_count = len(response.split())
        if 50 <= word_count <= 300:
            length_score = 1.0
        elif 20 <= word_count <= 500:
            length_score = 0.7
        else:
            length_score = 0.3
        coherence_factors.append(length_score)
        
        return statistics.mean(coherence_factors) if coherence_factors else 0.0 