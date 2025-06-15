"""
Diadochi: Intelligent Model Combination Framework

This module provides tools for combining domain-expert models to create 
better integrated AI systems. Named after Alexander the Great's successors
who divided his empire among themselves, this module intelligently divides
and combines domain expertise.

Key Components:
- ModelRegistry: Central repository for managing model instances
- Router: Components that direct queries to appropriate domain experts
- Chain: Sequential processing through multiple models
- Mixer: Components that combine responses from multiple models
- Ensemble: Router-based ensemble systems
- MixtureOfExperts: Parallel processing with weighted combination
"""

from .core.models import Model, ModelRegistry
from .core.routers import Router, KeywordRouter, EmbeddingRouter, LLMRouter
from .core.mixers import Mixer, DefaultMixer, SynthesisMixer, WeightedMixer
from .core.chains import Chain, SummarizingChain
from .core.ensemble import Ensemble
from .core.mixture_of_experts import MixtureOfExperts
from .evaluation.metrics import DomainExpertiseRetention, CrossDomainAccuracy, IntegrationCoherence

__version__ = "0.1.0"
__author__ = "Space Computer Team"

__all__ = [
    # Core Components
    "Model",
    "ModelRegistry",
    
    # Routing
    "Router",
    "KeywordRouter", 
    "EmbeddingRouter",
    "LLMRouter",
    
    # Mixing
    "Mixer",
    "DefaultMixer",
    "SynthesisMixer", 
    "WeightedMixer",
    
    # Patterns
    "Chain",
    "SummarizingChain",
    "Ensemble",
    "MixtureOfExperts",
    
    # Evaluation
    "DomainExpertiseRetention",
    "CrossDomainAccuracy", 
    "IntegrationCoherence",
] 