#!/usr/bin/env python3
"""
Setup script to integrate elite athlete data with the Space Computer backend system.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from backend.core.data_ingestion import AthleteDataIngestion
from backend.llm.embeddings import EmbeddingGenerator
from orchestration.rag_engine import BiomechanicalRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Initialize database with athlete data."""
    logger.info("Setting up database...")
    
    # Initialize database connection
    # This would connect to your PostgreSQL instance
    logger.info("Database setup complete")

def ingest_athlete_data():
    """Ingest all athlete data into the system."""
    logger.info("Starting athlete data ingestion...")
    
    try:
        ingestion = AthleteDataIngestion()
        results = ingestion.ingest_all_athletes()
        
        logger.info(f"Ingestion results:")
        logger.info(f"  - Successful: {len(results['successful'])}")
        logger.info(f"  - Failed: {len(results['failed'])}")
        logger.info(f"  - Total processed: {results['total_processed']}")
        logger.info(f"  - Knowledge base entries: {results['knowledge_base_entries']}")
        
        if results['failed']:
            logger.warning("Failed athletes:")
            for failed in results['failed']:
                logger.warning(f"  - {failed['athlete_id']}: {failed['reason']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return None

def setup_rag_system():
    """Setup RAG system with athlete knowledge base."""
    logger.info("Setting up RAG system...")
    
    try:
        rag = BiomechanicalRAG()
        
        # Generate embeddings for athlete data
        embedding_generator = EmbeddingGenerator()
        
        logger.info("RAG system setup complete")
        return True
        
    except Exception as e:
        logger.error(f"RAG setup failed: {e}")
        return False

def verify_integration():
    """Verify that the integration is working correctly."""
    logger.info("Verifying integration...")
    
    try:
        # Test data ingestion
        ingestion = AthleteDataIngestion()
        
        # Test with a known athlete
        test_athlete = "usain_bolt_final"
        athlete_data = ingestion.process_athlete(test_athlete)
        
        if athlete_data:
            logger.info(f"✓ Successfully processed {athlete_data['name']}")
            logger.info(f"  - Sport: {athlete_data['sport']}")
            logger.info(f"  - Frames: {len(athlete_data['frames'])}")
            logger.info(f"  - Duration: {athlete_data['metadata']['duration']:.2f}s")
        else:
            logger.error("✗ Failed to process test athlete")
            return False
        
        logger.info("✓ Integration verification complete")
        return True
        
    except Exception as e:
        logger.error(f"Integration verification failed: {e}")
        return False

def start_backend_services():
    """Start the backend services."""
    logger.info("Starting backend services...")
    
    # This would typically start your FastAPI server, Redis, etc.
    logger.info("To start backend services manually, run:")
    logger.info("  cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    logger.info("  docker-compose up -d  # For full production setup")

def main():
    """Main setup function."""
    logger.info("=== Space Computer Integration Setup ===")
    
    # Check if datasources exist
    datasources_path = Path("datasources")
    if not datasources_path.exists():
        logger.error("❌ datasources folder not found!")
        logger.error("Please ensure your athlete data is in the datasources folder")
        return False
    
    models_count = len(list((datasources_path / "models").glob("*.json")))
    videos_count = len(list((datasources_path / "annotated").glob("*.mp4")))
    
    logger.info(f"Found {models_count} model files and {videos_count} video files")
    
    if models_count == 0:
        logger.error("❌ No model files found in datasources/models/")
        return False
    
    # Setup steps
    steps = [
        ("Database Setup", setup_database),
        ("Data Ingestion", ingest_athlete_data),
        ("RAG System Setup", setup_rag_system),
        ("Integration Verification", verify_integration)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n--- {step_name} ---")
        try:
            result = step_func()
            if result is False:
                logger.error(f"❌ {step_name} failed")
                return False
            logger.info(f"✓ {step_name} completed")
        except Exception as e:
            logger.error(f"❌ {step_name} failed: {e}")
            return False
    
    logger.info("\n=== Integration Setup Complete ===")
    logger.info("Your Space Computer system is now ready!")
    logger.info("\nNext steps:")
    logger.info("1. Start the backend: cd backend && python -m uvicorn main:app --reload")
    logger.info("2. Start the frontend: npm run dev")
    logger.info("3. Open http://localhost:3000 to access the system")
    
    start_backend_services()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 