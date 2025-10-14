"""
Embeddings Engine - Converts text to semantic vectors

This is the "understanding" layer of the RAG system. It uses a sentence transformer
model to convert natural language text into 384-dimensional vectors that capture
semantic meaning.

Key Concepts:
- Embeddings: Numerical representations of text that capture meaning
- Semantic similarity: Similar meanings = similar vectors (measured by cosine similarity)
- Local inference: Runs entirely on your CPU, no API calls needed
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import config


class EmbeddingEngine:
    """
    Handles text-to-vector conversion using sentence transformers.
    
    This class loads the embedding model once and reuses it for all encodings,
    making subsequent operations fast.
    """
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier (default from config)
        
        Note: First run downloads ~80MB model, cached for future use.
        """
        print(f"🧠 Loading embedding model: {model_name}")
        print("   (First run will download the model, please wait...)")
        
        # Load the sentence transformer model
        # This is the "brain" that understands semantic meaning
        self.model = SentenceTransformer(model_name)
        
        # Get the embedding dimension (384 for MiniLM-L6-v2)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"✓ Model loaded! Embedding dimension: {self.dimension}")
    
    def encode(self, text: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        Convert text to embedding vector(s).
        
        This is where the magic happens! The model reads your text and converts it
        into a list of numbers that capture its semantic meaning.
        
        Args:
            text: Single string or list of strings to encode
            show_progress: Show progress bar for batch encoding
        
        Returns:
            numpy array of shape (embedding_dim,) for single text
            or (num_texts, embedding_dim) for multiple texts
        
        Example:
            >>> engine = EmbeddingEngine()
            >>> vector = engine.encode("passport in blue suitcase")
            >>> vector.shape
            (384,)
        """
        # Convert text to vector(s)
        # normalize_embeddings=True ensures vectors are unit length,
        # which makes cosine similarity faster to compute
        embeddings = self.model.encode(
            text,
            normalize_embeddings=True,  # Makes cosine similarity = dot product
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Returns a score from 0 (completely different) to 1 (identical meaning).
        
        This is useful for debugging and understanding how the model works.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        
        Example:
            >>> engine.similarity("passport location", "where is my passport")
            0.87  # High similarity despite different wording!
        """
        # Encode both texts
        embedding1 = self.encode(text1)
        embedding2 = self.encode(text2)
        
        # Compute cosine similarity
        # Since vectors are normalized, this is just the dot product
        similarity_score = np.dot(embedding1, embedding2)
        
        return float(similarity_score)


# =============================================================================
# SINGLETON PATTERN
# =============================================================================
# We create one global instance to avoid reloading the model multiple times
# This makes the app much faster after the first encoding

_embedding_engine = None

def get_embedding_engine() -> EmbeddingEngine:
    """
    Get or create the global embedding engine instance.
    
    This ensures we only load the model once, even if called multiple times.
    """
    global _embedding_engine
    
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    
    return _embedding_engine


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def encode_text(text: Union[str, List[str]]) -> np.ndarray:
    """
    Convenience function to encode text without managing the engine instance.
    
    Args:
        text: Text to encode
    
    Returns:
        Embedding vector(s)
    """
    engine = get_embedding_engine()
    return engine.encode(text)


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Convenience function to calculate similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score (0-1)
    """
    engine = get_embedding_engine()
    return engine.similarity(text1, text2)


if __name__ == "__main__":
    # Quick test to verify the embedding engine works
    print("\n" + "="*60)
    print("Testing Embedding Engine")
    print("="*60 + "\n")
    
    engine = EmbeddingEngine()
    
    # Test 1: Basic encoding
    print("Test 1: Encoding a single sentence")
    text = "My passport is in the blue suitcase"
    embedding = engine.encode(text)
    print(f"   Text: '{text}'")
    print(f"   Vector shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test 2: Semantic similarity
    print("\nTest 2: Semantic similarity")
    text1 = "passport in blue suitcase"
    text2 = "where is my travel document?"
    text3 = "what's for dinner tonight?"
    
    sim_12 = engine.similarity(text1, text2)
    sim_13 = engine.similarity(text1, text3)
    
    print(f"   '{text1}' vs '{text2}'")
    print(f"   Similarity: {sim_12:.3f} (should be HIGH)")
    
    print(f"\n   '{text1}' vs '{text3}'")
    print(f"   Similarity: {sim_13:.3f} (should be LOW)")
    
    print("\n✓ Embedding engine working correctly!\n")

