"""
Chain implementations for sequential processing through multiple domain expert models.

This module provides various chaining strategies for passing queries through
multiple domain experts in sequence, with each expert building on previous insights.
"""

from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from .models import Model, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class ChainContext:
    """Context object that accumulates information as it flows through the chain."""
    query: str
    responses: List[ModelResponse]
    metadata: Dict[str, Any]
    
    def add_response(self, response: ModelResponse) -> None:
        """Add a response to the chain context."""
        self.responses.append(response)
        self.metadata[f"response_{len(self.responses)}"] = response.content
        if response.domain:
            self.metadata[response.domain] = response.content


class Chain:
    """Sequential chain that processes queries through multiple models."""
    
    def __init__(self, models: List[Model], prompt_templates: Optional[Dict[str, str]] = None):
        """
        Initialize the chain.
        
        Args:
            models: List of models to chain together
            prompt_templates: Optional dict mapping model names to prompt templates
        """
        self.models = models
        self.prompt_templates = prompt_templates or {}
    
    def _get_prompt_template(self, model: Model, position: int) -> str:
        """Get prompt template for a model at a specific position in the chain."""
        # Try model-specific template first
        if model.name in self.prompt_templates:
            return self.prompt_templates[model.name]
        
        # Default templates based on position
        if position == 0:
            return "You are an expert in {domain}. Analyze this query: {query}"
        elif position == len(self.models) - 1:
            # Final model - synthesis
            return """You are tasked with synthesizing the following expert analyses into a comprehensive response.

Original query: {query}

Previous expert analyses:
{previous_responses}

Provide an integrated response that combines insights from all experts."""
        else:
            # Middle models
            return """You are an expert in {domain}.

Previous expert analysis:
{prev_response}

Original query: {query}

Building on the previous analysis, provide your expert perspective."""
    
    def _format_prompt(self, template: str, context: ChainContext, model: Model, position: int) -> str:
        """Format prompt template with context information."""
        format_kwargs = {
            "query": context.query,
            "domain": model.domain or "general",
            "model_name": model.name,
            "position": position
        }
        
        # Add previous responses
        if context.responses:
            format_kwargs["prev_response"] = context.responses[-1].content
            format_kwargs["previous_responses"] = "\n\n".join([
                f"Expert {i+1} ({resp.domain or 'Unknown'}): {resp.content}"
                for i, resp in enumerate(context.responses)
            ])
        else:
            format_kwargs["prev_response"] = ""
            format_kwargs["previous_responses"] = ""
        
        # Add all responses individually
        format_kwargs["responses"] = [resp.content for resp in context.responses]
        
        try:
            return template.format(**format_kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable {e}")
            return template.replace("{" + str(e).strip("'") + "}", "")
    
    def generate(self, query: str, **kwargs) -> ModelResponse:
        """Generate a response by chaining models sequentially."""
        context = ChainContext(
            query=query,
            responses=[],
            metadata={"chain_length": len(self.models)}
        )
        
        for i, model in enumerate(self.models):
            try:
                # Get and format prompt
                template = self._get_prompt_template(model, i)
                prompt = self._format_prompt(template, context, model, i)
                
                # Generate response
                response = model.generate(prompt, **kwargs)
                
                # Add to context
                context.add_response(response)
                
                logger.debug(f"Chain step {i+1}/{len(self.models)} completed with model {model.name}")
                
            except Exception as e:
                logger.error(f"Error in chain step {i+1} with model {model.name}: {e}")
                # Create error response
                error_response = ModelResponse(
                    content=f"Error in {model.name}: {str(e)}",
                    model_name=model.name,
                    domain=model.domain,
                    confidence=0.0,
                    metadata={"error": str(e)}
                )
                context.add_response(error_response)
        
        # Return final response
        if context.responses:
            final_response = context.responses[-1]
            
            chain_metadata = {
                "chain_type": "sequential",
                "chain_length": len(self.models),
                "model_sequence": [model.name for model in self.models]
            }
            
            return ModelResponse(
                content=final_response.content,
                model_name=f"Chain({' -> '.join([m.name for m in self.models])})",
                domain=final_response.domain,
                confidence=final_response.confidence,
                metadata=chain_metadata
            )
        else:
            return ModelResponse(
                content="Chain execution failed",
                model_name="Chain(failed)",
                domain=None,
                confidence=0.0,
                metadata={"chain_failed": True}
            )


class SummarizingChain(Chain):
    """Chain that summarizes intermediate responses to manage context length."""
    
    def __init__(self, models: List[Model], summarizer: Model, max_length: int = 2000, 
                 prompt_templates: Optional[Dict[str, str]] = None):
        """Initialize summarizing chain."""
        super().__init__(models, prompt_templates)
        self.summarizer = summarizer
        self.max_length = max_length
    
    def _should_summarize(self, response: ModelResponse) -> bool:
        """Check if response should be summarized."""
        return len(response.content) > self.max_length
    
    def _summarize_response(self, response: ModelResponse) -> ModelResponse:
        """Summarize a response using the summarizer model."""
        try:
            prompt = f"Summarize this analysis concisely while preserving key insights:\n\n{response.content}\n\nSummary:"
            summary_response = self.summarizer.generate(prompt)
            
            return ModelResponse(
                content=summary_response.content,
                model_name=f"Summarized({response.model_name})",
                domain=response.domain,
                confidence=response.confidence,
                metadata={
                    "summarized": True,
                    "original_length": len(response.content),
                    "summary_length": len(summary_response.content)
                }
            )
        except Exception as e:
            logger.error(f"Error summarizing response: {e}")
            return response
    
    def generate(self, query: str, **kwargs) -> ModelResponse:
        """Generate response with automatic summarization."""
        context = ChainContext(
            query=query,
            responses=[],
            metadata={"chain_length": len(self.models), "summarizations": 0}
        )
        
        for i, model in enumerate(self.models):
            try:
                template = self._get_prompt_template(model, i)
                prompt = self._format_prompt(template, context, model, i)
                response = model.generate(prompt, **kwargs)
                
                # Summarize if needed (but not for final response)
                if i < len(self.models) - 1 and self._should_summarize(response):
                    response = self._summarize_response(response)
                    context.metadata["summarizations"] += 1
                
                context.add_response(response)
                
            except Exception as e:
                logger.error(f"Error in summarizing chain step {i+1}: {e}")
                error_response = ModelResponse(
                    content=f"Error: {str(e)}",
                    model_name=model.name,
                    domain=model.domain,
                    confidence=0.0
                )
                context.add_response(error_response)
        
        # Return final response
        if context.responses:
            final_response = context.responses[-1]
            
            return ModelResponse(
                content=final_response.content,
                model_name=f"SummarizingChain({self.summarizer.name})",
                domain=final_response.domain,
                confidence=final_response.confidence,
                metadata={
                    "chain_type": "summarizing",
                    "summarizations": context.metadata["summarizations"]
                }
            )
        else:
            return ModelResponse(
                content="Chain execution failed",
                model_name="SummarizingChain(failed)",
                domain=None,
                confidence=0.0
            ) 