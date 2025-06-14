import os
import sys
import logging
from pathlib import Path
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple

# Add the backend directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Import backend components
from backend.api.routes import router as backend_router
from backend.api.athlete_endpoints import router as athlete_router
from backend.api.verification_endpoints import router as verification_router
from backend.config.settings import API_HOST, API_PORT, CORS_ORIGINS, DEBUG
from backend.core.biomechanical_analysis import BiomechanicalAnalyzer
from backend.core.data_ingestion import BiomechanicalDataLoader
from backend.ai.biomech_llm import BiomechLLM
from backend.ai.rag_system import BiomechRAG
from backend.core.pose_understanding import (
    PoseUnderstandingVerifier, 
    verify_pose_understanding_before_analysis,
    save_verification_images
)

# Import orchestration components
from orchestration.meta_orchestrator import MetaOrchestrator
from orchestration.query_processor import QueryProcessor
from orchestration.solver import BiomechanicalSolver
from orchestration.interpreter import GlbInterpreter
from orchestration.rag_engine import BiomechanicalRAG

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestration")

# Create FastAPI app
app = FastAPI(
    title="VisualKinetics API",
    description="API for biomechanical analysis of sports videos",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestration components
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "complexity_threshold": 0.7,
    "precision_threshold": 0.85,
    "knowledge_base_path": "data/knowledge_base"
}

meta_orchestrator = MetaOrchestrator(config)
query_processor = QueryProcessor(config)
biomech_solver = BiomechanicalSolver(config)
glb_interpreter = GlbInterpreter(config)
biomech_rag = BiomechanicalRAG(config)

# Define request models
class PostureQueryRequest(BaseModel):
    text_query: str
    glb_interaction: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None

class QueryAnalysisRequest(BaseModel):
    analysis_id: str
    query: str

class KnowledgeDocumentRequest(BaseModel):
    documents: List[Dict[str, Any]]
    category: Optional[str] = "user_added"

class FeedbackRequest(BaseModel):
    query_id: str
    feedback_text: str
    rating: int
    improved_parameters: Optional[Dict[str, float]] = None

# Include backend API routes
app.include_router(backend_router, prefix="/api")
app.include_router(athlete_router, prefix="/api")
app.include_router(verification_router, prefix="/api")

# Define posture query endpoint
@app.post("/api/posture/query", response_model=Dict[str, Any])
async def query_posture(request: PostureQueryRequest):
    """
    Process a posture query from the GLB model frontend and return optimized posture.
    """
    try:
        logger.info(f"Received posture query: {request.text_query[:100]}...")
        
        # Process through metacognitive orchestrator
        result = meta_orchestrator.process_query({
            "text": request.text_query,
            "model_params": request.glb_interaction.get("pose_data", {}),
            "constraints": request.constraints
        })
        
        return result
    except Exception as e:
        logger.error(f"Error processing posture query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/posture/process", response_model=Dict[str, Any])
async def process_posture_query(request: PostureQueryRequest):
    """
    Process a posture query step by step through the pipeline components.
    Shows the intermediate results for debugging and explanation.
    """
    try:
        logger.info(f"Processing posture query step by step: {request.text_query[:100]}...")
        
        # Step 1: Process the query
        structured_query = query_processor.process_user_input(
            request.text_query,
            request.glb_interaction,
            request.constraints
        )
        
        # Step 2: Retrieve relevant domain knowledge using RAG
        domain_context = biomech_rag.generate_context(
            request.text_query,
            structured_query.get("model_params", {})
        )
        structured_query["domain_context"] = domain_context
        
        # Step 3: Solve the optimization problem
        solution = biomech_solver.solve(structured_query)
        
        # Step 4: Interpret the solution back to GLB model
        glb_updates = glb_interpreter.interpret(solution, structured_query)
        
        # Return all intermediate results for transparency
        return {
            "structured_query": structured_query,
            "solution": solution,
            "glb_updates": glb_updates,
            "processing_time": {
                "total": solution.get("solving_time", 0)
            }
        }
    except Exception as e:
        logger.error(f"Error in step-by-step processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/add", response_model=Dict[str, Any])
async def add_knowledge(request: KnowledgeDocumentRequest):
    """
    Add new knowledge documents to the biomechanical knowledge base.
    """
    try:
        logger.info(f"Adding {len(request.documents)} new knowledge documents")
        
        # Add documents to the knowledge base
        biomech_rag.index_documents(request.documents, request.category)
        
        return {
            "success": True,
            "message": f"Added {len(request.documents)} documents to the knowledge base",
            "count": len(request.documents)
        }
    except Exception as e:
        logger.error(f"Error adding knowledge documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/retrieve", response_model=List[Dict[str, Any]])
async def retrieve_knowledge(query: str = Body(..., embed=True), filters: Optional[Dict[str, Any]] = Body(None, embed=True)):
    """
    Retrieve knowledge documents relevant to a query.
    """
    try:
        logger.info(f"Retrieving knowledge for query: {query[:100]}...")
        
        # Retrieve documents from the knowledge base
        results = biomech_rag.retrieve(query, filters)
        
        return results
    except Exception as e:
        logger.error(f"Error retrieving knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/templates/retrieve", response_model=List[Dict[str, Any]])
async def retrieve_templates(movement_type: str = Body(..., embed=True)):
    """
    Retrieve posture templates for a specific movement type.
    """
    try:
        logger.info(f"Retrieving templates for movement: {movement_type}")
        
        # Retrieve templates from the knowledge base
        templates = biomech_rag.retrieve_posture_templates(movement_type)
        
        return templates
    except Exception as e:
        logger.error(f"Error retrieving templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback/submit", response_model=Dict[str, Any])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a query to improve the system.
    """
    try:
        logger.info(f"Receiving feedback for query ID: {request.query_id}")
        
        # Format and export the feedback
        feedback_data = {
            "query_id": request.query_id,
            "feedback_text": request.feedback_text,
            "rating": request.rating,
            "improved_parameters": request.improved_parameters,
            "timestamp": str(import_datetime.datetime.now())
        }
        
        biomech_rag.export_feedback(feedback_data)
        
        return {
            "success": True,
            "message": "Feedback submitted successfully"
        }
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "services": {
        "meta_orchestrator": isinstance(meta_orchestrator, MetaOrchestrator),
        "query_processor": isinstance(query_processor, QueryProcessor),
        "biomech_solver": isinstance(biomech_solver, BiomechanicalSolver),
        "glb_interpreter": isinstance(glb_interpreter, GlbInterpreter),
        "biomech_rag": isinstance(biomech_rag, BiomechanicalRAG)
    }}

# Serve frontend in production
if not DEBUG:
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "out"
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
        
        @app.get("/{path:path}", include_in_schema=False)
        async def serve_frontend(path: str):
            frontend_file = frontend_path / path
            if frontend_file.exists() and frontend_file.is_file():
                return FileResponse(str(frontend_file))
            return FileResponse(str(frontend_path / "index.html"))
    else:
        logger.warning(f"Frontend build directory not found at {frontend_path}")

def start_server():
    """Start the orchestration server"""
    logger.info(f"Starting VisualKinetics server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "orchestration.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )

if __name__ == "__main__":
    start_server()

class SpaceComputerOrchestrator:
    def __init__(self):
        # ... existing code ...
        
        # Add pose understanding verification
        self.pose_verifier = PoseUnderstandingVerifier(similarity_threshold=0.7)
        self.verification_enabled = True  # Can be disabled for testing
        self.verification_cache = {}  # Cache verification results
        
    async def initialize(self):
        """Initialize all components"""
        # ... existing code ...
        
        # Initialize pose understanding verifier
        if self.verification_enabled:
            await self.pose_verifier.initialize()
            logger.info("Pose understanding verifier initialized")
    
    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user query with pose understanding verification
        """
        try:
            query_id = str(uuid.uuid4())
            start_time = time.time()
            
            logger.info(f"Processing query {query_id}: {query_data.get('text', '')[:100]}...")
            
            # Extract query components
            query_text = query_data.get('text', '')
            pose_data = query_data.get('pose_data')
            context = query_data.get('context', {})
            
            # Step 1: Verify pose understanding if pose data is provided
            verification_result = None
            if pose_data and self.verification_enabled:
                should_proceed, verification_result = await self._verify_pose_understanding(
                    pose_data, query_text, query_id
                )
                
                if not should_proceed:
                    return {
                        'query_id': query_id,
                        'status': 'failed',
                        'error': 'AI failed to understand the pose data',
                        'verification_result': {
                            'understood': verification_result.understood,
                            'confidence': verification_result.confidence,
                            'similarity_score': verification_result.similarity_score,
                            'error_message': verification_result.error_message
                        },
                        'processing_time': time.time() - start_time
                    }
            
            # Step 2: Proceed with normal query processing
            result = await self._process_verified_query(
                query_id, query_text, pose_data, context, verification_result
            )
            
            # Add verification info to result
            if verification_result:
                result['verification'] = {
                    'understood': verification_result.understood,
                    'confidence': verification_result.confidence,
                    'similarity_score': verification_result.similarity_score,
                    'verification_time': verification_result.verification_time
                }
            
            result['processing_time'] = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query_id': query_id,
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def _verify_pose_understanding(self, pose_data: Dict, query_text: str, query_id: str) -> Tuple[bool, Any]:
        """
        Verify AI understanding of pose data before analysis
        """
        try:
            # Check cache first
            cache_key = self._generate_pose_cache_key(pose_data, query_text)
            if cache_key in self.verification_cache:
                cached_result = self.verification_cache[cache_key]
                logger.info(f"Using cached verification result for query {query_id}")
                return cached_result.understood, cached_result
            
            # Perform verification
            logger.info(f"Verifying pose understanding for query {query_id}")
            result = await self.pose_verifier.verify_with_retry(pose_data, query_text)
            
            # Cache the result
            self.verification_cache[cache_key] = result
            
            # Save verification images for debugging if needed
            if not result.understood:
                try:
                    save_verification_images(
                        result, 
                        f"debug/verification/{query_id}", 
                        f"failed_{query_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save verification images: {e}")
            
            logger.info(f"Pose verification complete for query {query_id}: "
                       f"understood={result.understood}, confidence={result.confidence:.3f}")
            
            return result.understood, result
            
        except Exception as e:
            logger.error(f"Error in pose understanding verification: {e}")
            # Return False to be safe, but don't crash the system
            return False, None
    
    def _generate_pose_cache_key(self, pose_data: Dict, query_text: str) -> str:
        """Generate cache key for pose verification"""
        import hashlib
        
        # Create a simplified representation of pose data
        pose_summary = {}
        for joint, data in pose_data.items():
            if isinstance(data, dict) and 'x' in data and 'y' in data:
                # Round to reduce cache misses from minor variations
                pose_summary[joint] = {
                    'x': round(data['x'], 2),
                    'y': round(data['y'], 2),
                    'confidence': round(data.get('confidence', 1.0), 2)
                }
        
        # Combine pose data and query text
        cache_data = {
            'pose': pose_summary,
            'query': query_text.lower().strip()
        }
        
        # Generate hash
        cache_str = str(sorted(cache_data.items()))
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    async def _process_verified_query(self, query_id: str, query_text: str, 
                                    pose_data: Dict, context: Dict, 
                                    verification_result: Any) -> Dict[str, Any]:
        """
        Process query after pose understanding has been verified
        """
        # Enhanced context with verification confidence
        enhanced_context = context.copy()
        if verification_result:
            enhanced_context['pose_understanding_confidence'] = verification_result.confidence
            enhanced_context['pose_verified'] = verification_result.understood
        
        # Continue with existing query processing logic
        query_complexity = await self.complexity_analyzer.analyze_complexity(query_text)
        intent = await self.intent_classifier.classify_intent(query_text, enhanced_context)
        
        # Route to appropriate processor
        if intent.type == "biomechanical_analysis" and pose_data:
            return await self._process_biomechanical_query(
                query_id, query_text, pose_data, enhanced_context, query_complexity
            )
        elif intent.type == "athlete_comparison" and pose_data:
            return await self._process_athlete_comparison_query(
                query_id, query_text, pose_data, enhanced_context, query_complexity
            )
        else:
            return await self._process_general_query(
                query_id, query_text, enhanced_context, query_complexity
            )
    
    async def _process_biomechanical_query(self, query_id: str, query_text: str, 
                                         pose_data: Dict, context: Dict, 
                                         complexity: Any) -> Dict[str, Any]:
        """Process biomechanical analysis queries with verified pose understanding"""
        try:
            # Perform biomechanical analysis
            analysis_result = await self.biomech_analyzer.analyze_pose(pose_data)
            
            # Generate AI response with high confidence due to verification
            ai_response = await self.biomech_llm.generate_response(
                query_text, 
                pose_data, 
                analysis_result, 
                context
            )
            
            return {
                'query_id': query_id,
                'status': 'success',
                'response': ai_response,
                'analysis': analysis_result,
                'confidence': min(1.0, context.get('pose_understanding_confidence', 0.8) + 0.2),
                'type': 'biomechanical_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in biomechanical query processing: {e}")
            return {
                'query_id': query_id,
                'status': 'error',
                'error': str(e),
                'type': 'biomechanical_analysis'
            }
    
    async def _process_athlete_comparison_query(self, query_id: str, query_text: str, 
                                              pose_data: Dict, context: Dict, 
                                              complexity: Any) -> Dict[str, Any]:
        """Process athlete comparison queries with verified pose understanding"""
        try:
            # Get athlete data if specified
            athlete_id = context.get('athlete_id')
            if not athlete_id:
                # Try to extract athlete from query
                athlete_id = await self._extract_athlete_from_query(query_text)
            
            if athlete_id:
                # Load athlete data
                athlete_data = await self.data_loader.load_athlete_data(athlete_id)
                
                # Perform comparison
                comparison_result = await self.biomech_analyzer.compare_with_athlete(
                    pose_data, athlete_data
                )
                
                # Generate AI response
                ai_response = await self.biomech_llm.generate_comparison_response(
                    query_text, 
                    pose_data, 
                    athlete_data, 
                    comparison_result, 
                    context
                )
                
                return {
                    'query_id': query_id,
                    'status': 'success',
                    'response': ai_response,
                    'comparison': comparison_result,
                    'athlete_data': athlete_data,
                    'confidence': min(1.0, context.get('pose_understanding_confidence', 0.8) + 0.15),
                    'type': 'athlete_comparison'
                }
            else:
                return {
                    'query_id': query_id,
                    'status': 'error',
                    'error': 'No athlete specified for comparison',
                    'type': 'athlete_comparison'
                }
                
        except Exception as e:
            logger.error(f"Error in athlete comparison query processing: {e}")
            return {
                'query_id': query_id,
                'status': 'error',
                'error': str(e),
                'type': 'athlete_comparison'
            }
    
    async def _extract_athlete_from_query(self, query_text: str) -> Optional[str]:
        """Extract athlete name/ID from query text"""
        # This could be enhanced with NER or a more sophisticated approach
        athlete_keywords = {
            'federer': 'ath_federer_001',
            'serena': 'ath_serena_001',
            'bolt': 'ath_bolt_001',
            'messi': 'ath_messi_001',
            # Add more athlete mappings
        }
        
        query_lower = query_text.lower()
        for keyword, athlete_id in athlete_keywords.items():
            if keyword in query_lower:
                return athlete_id
        
        return None
    
    def enable_verification(self, enabled: bool = True):
        """Enable or disable pose understanding verification"""
        self.verification_enabled = enabled
        logger.info(f"Pose understanding verification {'enabled' if enabled else 'disabled'}")
    
    def clear_verification_cache(self):
        """Clear the verification cache"""
        self.verification_cache.clear()
        logger.info("Verification cache cleared")
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get statistics about pose understanding verification"""
        if not self.verification_cache:
            return {'cache_size': 0, 'success_rate': 0.0}
        
        total = len(self.verification_cache)
        successful = sum(1 for result in self.verification_cache.values() if result.understood)
        
        return {
            'cache_size': total,
            'success_rate': successful / total if total > 0 else 0.0,
            'average_confidence': sum(result.confidence for result in self.verification_cache.values()) / total,
            'average_similarity': sum(result.similarity_score for result in self.verification_cache.values()) / total
        }
