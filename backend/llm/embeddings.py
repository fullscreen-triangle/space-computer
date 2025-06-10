"""
Embeddings module for Spectacular's Retrieval Augmented Generation (RAG).
Implements text embeddings using sentence-transformers for document retrieval.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import faiss

class EmbeddingGenerator:
    """
    Embedding generator for RAG using sentence-transformers.
    Generates embeddings that can be used with FAISS for retrieval.
    """
    
    def __init__(self, device: str = None, use_scibert: bool = False):
        """
        Initialize the embedding generator.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', etc.)
            use_scibert: Whether to use ScienceBERT for academic corpus similarity
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.use_scibert = use_scibert
        self._load_model()
        self.index = None
        self.documents = []
    
    def _load_model(self):
        """Load the sentence-transformers model from Hugging Face."""
        from sentence_transformers import SentenceTransformer
        
        model_name = "allenai/scibert_scivocab_uncased" if self.use_scibert else "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading {model_name} model for embeddings...")
        
        # Load the model (SentenceTransformer wraps HuggingFace models)
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the input text(s).
        
        Args:
            texts: Input text or list of texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize the EmbeddingGenerator first.")
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        return embeddings
    
    def build_index(self, documents: List[str]) -> None:
        """
        Build a FAISS index from a list of documents.
        
        Args:
            documents: List of text documents to index
        """
        self.documents = documents
        
        # Generate embeddings for all documents
        embeddings = self.encode(documents)
        
        # Get embedding dimension
        dimension = embeddings.shape[1]
        
        # Create a FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Built FAISS index with {len(documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dictionaries with document and score
        """
        if self.index is None or not self.documents:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Generate embedding for the query
        query_embedding = self.encode(query)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
        
        # Format results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity score
                    "rank": i
                })
        
        return results
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[Dict]:
        """
        Rerank a list of documents based on their similarity to the query.
        Useful as a second stage after retrieval.
        
        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Number of results to return
            
        Returns:
            List of reranked documents with scores
        """
        if not documents:
            return []
        
        # Generate embeddings for query and documents
        query_embedding = self.encode(query)
        doc_embeddings = self.encode(documents)
        
        # Calculate similarity scores
        scores = []
        for i, doc_emb in enumerate(doc_embeddings):
            # Cosine similarity
            sim = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            scores.append((i, float(sim)))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for i, (idx, score) in enumerate(scores[:top_k]):
            results.append({
                "document": documents[idx],
                "score": score,
                "rank": i
            })
        
        return results 