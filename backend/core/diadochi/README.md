# Diadochi: Intelligent Model Combination Framework

> Named after Alexander the Great's successors who divided his empire among themselves, Diadochi intelligently divides and combines domain expertise.

## Overview

Diadochi is a comprehensive framework for combining domain-expert LLMs to create better integrated AI systems. It addresses the challenge of leveraging specialized domain knowledge while enabling effective cross-domain integration for complex, interdisciplinary problems.

## Key Features

- **Multiple Architectural Patterns**: Router-based ensembles, sequential chaining, mixture of experts
- **Flexible Routing**: Keyword-based, embedding-based, and LLM-based routing strategies
- **Advanced Mixing**: Various strategies for combining and synthesizing expert responses
- **Multi-Provider Support**: OpenAI, Anthropic, Ollama, and extensible to other providers
- **Evaluation Metrics**: Specialized metrics for multi-domain system assessment
- **Extensible Design**: Clean abstractions for adding new components

## Architecture

### Core Components

1. **Models** (`core/models.py`): Abstractions for different LLM providers
2. **Routers** (`core/routers.py`): Direct queries to appropriate domain experts
3. **Mixers** (`core/mixers.py`): Combine responses from multiple models
4. **Chains** (`core/chains.py`): Sequential processing through multiple models
5. **Ensembles** (`core/ensemble.py`): Router-based model selection
6. **Mixture of Experts** (`core/mixture_of_experts.py`): Parallel processing with weighted combination

### Architectural Patterns

#### 1. Router-Based Ensembles
Routes queries to the most appropriate domain expert based on query analysis.

```python
from diadochi import Ensemble, ModelRegistry, EmbeddingRouter

# Setup
registry = ModelRegistry()
registry.add_model("biomechanics", "openai", "gpt-4", domain="biomechanics")
registry.add_model("nutrition", "anthropic", "claude-3-sonnet", domain="nutrition")

router = EmbeddingRouter(embedding_model)
router.add_domain("biomechanics", "Study of movement mechanics and forces")
router.add_domain("nutrition", "Dietary strategies for athletic performance")

ensemble = Ensemble(router, registry, default_model="general")
response = ensemble.generate("What is optimal stride frequency for sprinters?")
```

#### 2. Sequential Chaining
Passes queries through multiple experts in sequence, with each building on previous insights.

```python
from diadochi import Chain

models = [biomechanics_model, physiology_model, nutrition_model, synthesizer]
templates = {
    "biomechanics": "Analyze movement mechanics: {query}",
    "physiology": "Building on {prev_response}, analyze physiology: {query}",
    # ... more templates
}

chain = Chain(models, templates)
response = chain.generate("How to optimize sprint performance?")
```

#### 3. Mixture of Experts
Processes queries through multiple experts in parallel and combines outputs based on confidence.

```python
from diadochi import MixtureOfExperts, EmbeddingConfidenceEstimator, SynthesisMixer

confidence_estimator = EmbeddingConfidenceEstimator(embedding_model)
mixer = SynthesisMixer(synthesis_model)

moe = MixtureOfExperts(registry, confidence_estimator, mixer)
response = moe.generate("What factors affect endurance performance?")
```

## Quick Start

### Installation

```bash
# Install core dependencies
pip install openai anthropic ollama numpy

# Install the diadochi module (assuming it's in your Python path)
# Add backend/core to your PYTHONPATH or install as a package
```

### Basic Usage

```python
from diadochi import ModelRegistry, EmbeddingRouter, Ensemble

# 1. Set up model registry
registry = ModelRegistry()
registry.add_model("expert1", "openai", "gpt-4", domain="domain1")
registry.add_model("expert2", "ollama", "llama3.2", domain="domain2")

# 2. Create router
router = EmbeddingRouter(embedding_model, threshold=0.7)
router.add_domain("domain1", "Description of domain 1")
router.add_domain("domain2", "Description of domain 2")

# 3. Create ensemble
ensemble = Ensemble(router, registry, default_model="expert1")

# 4. Generate response
response = ensemble.generate("Your query here")
print(response.content)
```

## Model Support

### Supported Providers

- **OpenAI**: GPT-4, GPT-3.5, with embeddings support
- **Anthropic**: Claude 3 family (Sonnet, Opus, Haiku)
- **Ollama**: Local models (Llama, Mistral, etc.)
- **Extensible**: Easy to add new providers

### Model Configuration

```python
# OpenAI model
registry.add_model("gpt4", "openai", "gpt-4", 
                  domain="general", api_key="your-key")

# Anthropic model  
registry.add_model("claude", "anthropic", "claude-3-sonnet-20240229",
                  domain="analysis", api_key="your-key")

# Ollama model (local)
registry.add_model("llama", "ollama", "llama3.2", 
                  domain="coding", base_url="http://localhost:11434")
```

## Routing Strategies

### Embedding-Based Routing
Uses semantic similarity between queries and domain descriptions.

```python
router = EmbeddingRouter(embedding_model, threshold=0.7, temperature=0.5)
router.add_domain("biomechanics", "Study of movement and force mechanics")
```

### Keyword-Based Routing
Routes based on keyword matching with optional weights.

```python
router = KeywordRouter()
router.add_domain("nutrition", 
                 ["diet", "protein", "supplements"], 
                 weights={"protein": 2.0, "diet": 1.5})
```

### LLM-Based Routing
Uses a smaller LLM to analyze queries and determine routing.

```python
router = LLMRouter(router_model, domains=["domain1", "domain2"])
```

## Response Mixing

### Synthesis Mixer
Uses an LLM to synthesize multiple expert responses.

```python
mixer = SynthesisMixer(synthesis_model, prompt_template=custom_template)
```

### Weighted Mixer
Combines responses using confidence-based weighting.

```python
mixer = WeightedMixer(strategy="confidence", min_weight=0.1)
```

### Default Mixer
Returns the response with the highest confidence.

```python
mixer = DefaultMixer()
```

## Evaluation

Diadochi includes specialized metrics for evaluating multi-domain systems:

```python
from diadochi.evaluation import (
    DomainExpertiseRetention, 
    CrossDomainAccuracy, 
    IntegrationCoherence
)

# Evaluate how well the system maintains domain expertise
der = DomainExpertiseRetention(domain_models)
result = der.evaluate(queries, responses, domain_labels=labels)

# Evaluate cross-domain integration
cda = CrossDomainAccuracy()
result = cda.evaluate(queries, responses)

# Evaluate response coherence
ic = IntegrationCoherence()
result = ic.evaluate(queries, responses)
```

## Use Cases

### Sports Science
Combine expertise from biomechanics, exercise physiology, nutrition, and psychology:

```python
# Set up domain experts
registry.add_model("biomech", "openai", "gpt-4", domain="biomechanics")
registry.add_model("physio", "anthropic", "claude-3-sonnet", domain="physiology") 
registry.add_model("nutrition", "ollama", "llama3.2", domain="nutrition")

# Create mixture of experts for comprehensive analysis
moe = MixtureOfExperts(registry, confidence_estimator, synthesis_mixer)
response = moe.generate("How can a sprinter optimize their 100m performance?")
```

### Computer Vision
Combine pose analysis, biomechanical assessment, and performance optimization:

```python
# Chain experts for progressive analysis
chain = Chain([
    pose_expert,          # Analyze movement patterns
    biomech_expert,       # Assess mechanical efficiency  
    performance_expert    # Provide optimization recommendations
])
```

## Advanced Configuration

### Custom Prompt Templates

```python
templates = {
    "biomechanics": """You are a biomechanics expert. 
    Analyze the movement mechanics in: {query}
    Focus on forces, velocities, and efficiency.""",
    
    "physiology": """You are an exercise physiologist.
    Previous analysis: {prev_response}
    
    Building on the biomechanical analysis above, 
    examine the physiological aspects of: {query}"""
}

chain = Chain(models, templates)
```

### Context Management

```python
# For long chains, use summarizing chain to manage context
summarizing_chain = SummarizingChain(
    models=expert_models,
    summarizer=summarizer_model,
    max_length=2000
)
```

### Parallel Processing

```python
# Enable parallel processing in mixture of experts
response = moe.generate(query, parallel=True)

# Control parallelism
moe = MixtureOfExperts(
    models=registry,
    confidence_estimator=estimator,
    mixer=mixer,
    max_experts=3,  # Limit number of experts
    threshold=0.2   # Confidence threshold
)
```

## Error Handling

Diadochi includes robust error handling:

```python
try:
    response = ensemble.generate(query)
except ValueError as e:
    # Handle routing failures
    logger.error(f"Routing failed: {e}")
except Exception as e:
    # Handle model errors
    logger.error(f"Model error: {e}")

# Check response metadata for routing information
if response.metadata.get("routing_failed"):
    logger.warning("Used fallback model")
```

## Extending Diadochi

### Adding New Model Providers

```python
from diadochi.core.models import Model, ModelResponse

class CustomModel(Model):
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        # Implement your model logic
        return ModelResponse(content=result, model_name=self.name)
    
    def embed(self, text: str) -> List[float]:
        # Implement embedding logic
        return embedding_vector
```

### Custom Routers

```python
from diadochi.core.routers import Router, RoutingResult

class CustomRouter(Router):
    def route(self, query: str, available_models: List[str]) -> Optional[RoutingResult]:
        # Implement your routing logic
        return RoutingResult(model_name=best_model, confidence=score)
```

### Custom Mixers

```python
from diadochi.core.mixers import Mixer, WeightedResponse

class CustomMixer(Mixer):
    def mix(self, query: str, responses: List[WeightedResponse]) -> ModelResponse:
        # Implement your mixing logic
        return combined_response
```

## Performance Considerations

- **Parallel Processing**: Use `parallel=True` in MoE for faster inference
- **Context Management**: Use `SummarizingChain` for long chains
- **Caching**: Consider implementing response caching for repeated queries
- **Model Selection**: Balance between accuracy and computational cost

## Contributing

Diadochi is designed to be extensible. Key areas for contribution:

1. **New Model Providers**: Add support for additional LLM providers
2. **Routing Strategies**: Implement new routing algorithms
3. **Mixing Techniques**: Develop advanced response combination methods
4. **Evaluation Metrics**: Create domain-specific evaluation measures
5. **Optimization**: Improve performance and efficiency

## License

This project is part of the Space Computer framework. See the main project license for details.

## Citation

If you use Diadochi in your research, please cite:

```bibtex
@software{diadochi2024,
  title={Diadochi: Intelligent Model Combination Framework},
  author={Space Computer Team},
  year={2024},
  url={https://github.com/your-org/space-computer}
}
``` 