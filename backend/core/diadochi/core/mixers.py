"""
Mixer implementations for combining responses from multiple domain expert models.

This module provides various strategies for combining and synthesizing responses
from multiple domain experts into coherent, integrated responses.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from .models import Model, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class WeightedResponse:
    """Response with associated weight and metadata."""
    response: ModelResponse
    weight: float
    metadata: Optional[Dict[str, Any]] = None


class Mixer(ABC):
    """Abstract base class for all mixers."""
    
    @abstractmethod
    def mix(self, query: str, responses: List[WeightedResponse]) -> ModelResponse:
        """Mix multiple weighted responses into a single response."""
        pass


class DefaultMixer(Mixer):
    """Default mixer that returns the response with highest weight."""
    
    def mix(self, query: str, responses: List[WeightedResponse]) -> ModelResponse:
        """Return the response with the highest weight."""
        if not responses:
            raise ValueError("No responses provided to mix")
        
        # Find response with highest weight
        best_response = max(responses, key=lambda r: r.weight)
        
        return ModelResponse(
            content=best_response.response.content,
            model_name=f"DefaultMixer({best_response.response.model_name})",
            domain=best_response.response.domain,
            confidence=best_response.weight,
            metadata={
                "mixer_type": "default",
                "selected_model": best_response.response.model_name,
                "selected_weight": best_response.weight,
                "total_responses": len(responses)
            }
        )


class ConcatenationMixer(Mixer):
    """Mixer that concatenates responses with domain labels."""
    
    def __init__(self, include_weights: bool = True, separator: str = "\n\n"):
        self.include_weights = include_weights
        self.separator = separator
    
    def mix(self, query: str, responses: List[WeightedResponse]) -> ModelResponse:
        """Concatenate responses with domain labels."""
        if not responses:
            raise ValueError("No responses provided to mix")
        
        # Sort responses by weight (highest first)
        sorted_responses = sorted(responses, key=lambda r: r.weight, reverse=True)
        
        # Build concatenated response
        parts = []
        total_weight = 0.0
        domains = []
        models = []
        
        for weighted_resp in sorted_responses:
            resp = weighted_resp.response
            weight = weighted_resp.weight
            total_weight += weight
            
            # Format header
            if self.include_weights:
                header = f"[{resp.domain or 'Unknown'} ({weight:.1%})]:"
            else:
                header = f"[{resp.domain or 'Unknown'}]:"
            
            parts.append(f"{header}\n{resp.content}")
            
            if resp.domain and resp.domain not in domains:
                domains.append(resp.domain)
            if resp.model_name not in models:
                models.append(resp.model_name)
        
        combined_content = self.separator.join(parts)
        
        return ModelResponse(
            content=combined_content,
            model_name=f"ConcatenationMixer({'+'.join(models)})",
            domain="+".join(domains) if domains else None,
            confidence=total_weight / len(responses) if responses else 0.0,
            metadata={
                "mixer_type": "concatenation",
                "num_responses": len(responses),
                "total_weight": total_weight,
                "contributing_models": models,
                "contributing_domains": domains
            }
        )


class WeightedMixer(Mixer):
    """Mixer that combines responses using weighted averaging strategies."""
    
    def __init__(self, strategy: str = "confidence", min_weight: float = 0.1):
        """
        Initialize weighted mixer.
        
        Args:
            strategy: Weighting strategy ('confidence', 'equal', 'exponential')
            min_weight: Minimum weight threshold for inclusion
        """
        self.strategy = strategy
        self.min_weight = min_weight
    
    def _calculate_weights(self, responses: List[WeightedResponse]) -> List[float]:
        """Calculate normalized weights based on strategy."""
        if self.strategy == "equal":
            return [1.0 / len(responses)] * len(responses)
        
        elif self.strategy == "confidence":
            weights = [r.weight for r in responses]
            total = sum(weights)
            return [w / total if total > 0 else 0.0 for w in weights]
        
        elif self.strategy == "exponential":
            import math
            weights = [math.exp(r.weight) for r in responses]
            total = sum(weights)
            return [w / total if total > 0 else 0.0 for w in weights]
        
        else:
            raise ValueError(f"Unknown weighting strategy: {self.strategy}")
    
    def mix(self, query: str, responses: List[WeightedResponse]) -> ModelResponse:
        """Mix responses using weighted combination."""
        if not responses:
            raise ValueError("No responses provided to mix")
        
        # Filter responses by minimum weight
        filtered_responses = [r for r in responses if r.weight >= self.min_weight]
        if not filtered_responses:
            # If no responses meet threshold, use the best one
            filtered_responses = [max(responses, key=lambda r: r.weight)]
        
        # Calculate weights
        weights = self._calculate_weights(filtered_responses)
        
        # Create weighted combination summary
        summary_parts = []
        total_confidence = 0.0
        domains = []
        models = []
        
        for i, (weighted_resp, weight) in enumerate(zip(filtered_responses, weights)):
            resp = weighted_resp.response
            total_confidence += weight * weighted_resp.weight
            
            # Extract key points (simplified - could be enhanced with NLP)
            content_preview = resp.content[:200] + "..." if len(resp.content) > 200 else resp.content
            summary_parts.append(f"• {resp.domain or 'Unknown'} ({weight:.1%}): {content_preview}")
            
            if resp.domain and resp.domain not in domains:
                domains.append(resp.domain)
            if resp.model_name not in models:
                models.append(resp.model_name)
        
        combined_content = f"Weighted synthesis from {len(filtered_responses)} experts:\n\n" + "\n\n".join(summary_parts)
        
        return ModelResponse(
            content=combined_content,
            model_name=f"WeightedMixer({'+'.join(models)})",
            domain="+".join(domains) if domains else None,
            confidence=total_confidence,
            metadata={
                "mixer_type": "weighted",
                "strategy": self.strategy,
                "num_responses": len(filtered_responses),
                "weights": weights,
                "contributing_models": models,
                "contributing_domains": domains
            }
        )


class SynthesisMixer(Mixer):
    """Mixer that uses an LLM to synthesize multiple responses."""
    
    def __init__(self, synthesis_model: Model, prompt_template: Optional[str] = None):
        self.synthesis_model = synthesis_model
        self.prompt_template = prompt_template or self._default_prompt_template()
    
    def _default_prompt_template(self) -> str:
        """Default prompt template for synthesis."""
        return """You are tasked with synthesizing responses from multiple domain experts into a coherent, integrated response.

Original query: {query}

Expert responses:
{expert_responses}

Create a unified response that integrates insights from all relevant experts and provides a comprehensive answer.

Synthesized response:"""
    
    def mix(self, query: str, responses: List[WeightedResponse]) -> ModelResponse:
        """Synthesize responses using an LLM."""
        if not responses:
            raise ValueError("No responses provided to mix")
        
        # Format responses for synthesis
        expert_responses = []
        for i, weighted_resp in enumerate(responses, 1):
            resp = weighted_resp.response
            domain = resp.domain or "Unknown"
            expert_responses.append(f"Expert {i} - {domain}:\n{resp.content}")
        
        # Create synthesis prompt
        prompt = self.prompt_template.format(
            query=query,
            expert_responses="\n\n".join(expert_responses)
        )
        
        try:
            synthesis_response = self.synthesis_model.generate(prompt)
            
            # Calculate combined confidence
            avg_confidence = sum(r.weight for r in responses) / len(responses)
            
            domains = [r.response.domain for r in responses if r.response.domain]
            
            return ModelResponse(
                content=synthesis_response.content,
                model_name=f"SynthesisMixer({self.synthesis_model.name})",
                domain="+".join(set(domains)) if domains else None,
                confidence=avg_confidence,
                metadata={
                    "mixer_type": "synthesis",
                    "num_experts": len(responses),
                    "synthesis_model": self.synthesis_model.name
                }
            )
        
        except Exception as e:
            logger.error(f"Error in synthesis mixing: {e}")
            # Fallback to default mixer
            fallback_mixer = DefaultMixer()
            return fallback_mixer.mix(query, responses)


class HierarchicalMixer(Mixer):
    """Mixer that groups responses by domain and synthesizes hierarchically."""
    
    def __init__(self, synthesis_model: Model, group_synthesis_template: Optional[str] = None, 
                 final_synthesis_template: Optional[str] = None):
        self.synthesis_model = synthesis_model
        self.group_synthesis_template = group_synthesis_template or self._default_group_template()
        self.final_synthesis_template = final_synthesis_template or self._default_final_template()
    
    def _default_group_template(self) -> str:
        """Default template for synthesizing within domain groups."""
        return """Synthesize the following responses from the {domain} domain:

{responses}

Create a coherent summary that captures the key insights from this domain while maintaining technical accuracy.

Domain summary:"""
    
    def _default_final_template(self) -> str:
        """Default template for final cross-domain synthesis."""
        return """You are synthesizing insights from multiple domains to answer this query:

Query: {query}

Domain summaries:

{domain_summaries}

Create a comprehensive, integrated response that:
1. Combines insights from all domains
2. Shows how the domains relate to each other
3. Provides actionable conclusions
4. Addresses the original query completely

Final response:"""
    
    def mix(self, query: str, responses: List[WeightedResponse]) -> ModelResponse:
        """Mix responses using hierarchical synthesis."""
        if not responses:
            raise ValueError("No responses provided to mix")
        
        # Group responses by domain
        domain_groups: Dict[str, List[WeightedResponse]] = {}
        for resp in responses:
            domain = resp.response.domain or "general"
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(resp)
        
        # Synthesize within each domain group
        domain_summaries = {}
        for domain, group_responses in domain_groups.items():
            if len(group_responses) == 1:
                # Single response - use as-is
                domain_summaries[domain] = group_responses[0].response.content
            else:
                # Multiple responses - synthesize
                responses_text = "\n\n".join([
                    f"Response {i+1}: {resp.response.content}"
                    for i, resp in enumerate(group_responses)
                ])
                
                group_prompt = self.group_synthesis_template.format(
                    domain=domain,
                    responses=responses_text
                )
                
                try:
                    group_synthesis = self.synthesis_model.generate(group_prompt)
                    domain_summaries[domain] = group_synthesis.content
                except Exception as e:
                    logger.error(f"Error synthesizing domain {domain}: {e}")
                    # Fallback to concatenation
                    domain_summaries[domain] = "\n\n".join([resp.response.content for resp in group_responses])
        
        # Final cross-domain synthesis
        summaries_text = "\n\n".join([
            f"{domain.upper()}:\n{summary}"
            for domain, summary in domain_summaries.items()
        ])
        
        final_prompt = self.final_synthesis_template.format(
            query=query,
            domain_summaries=summaries_text
        )
        
        try:
            final_synthesis = self.synthesis_model.generate(final_prompt)
            
            # Calculate metadata
            total_weight = sum(r.weight for r in responses)
            avg_confidence = total_weight / len(responses) if responses else 0.0
            
            return ModelResponse(
                content=final_synthesis.content,
                model_name=f"HierarchicalMixer({self.synthesis_model.name})",
                domain="+".join(domain_groups.keys()),
                confidence=avg_confidence,
                metadata={
                    "mixer_type": "hierarchical",
                    "synthesis_model": self.synthesis_model.name,
                    "num_domains": len(domain_groups),
                    "domain_groups": {domain: len(group) for domain, group in domain_groups.items()},
                    "total_responses": len(responses)
                }
            )
        
        except Exception as e:
            logger.error(f"Error in final synthesis: {e}")
            # Fallback to simple synthesis mixer
            fallback_mixer = SynthesisMixer(self.synthesis_model)
            return fallback_mixer.mix(query, responses) 