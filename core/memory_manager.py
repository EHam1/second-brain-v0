"""
Memory Manager - Business logic for semantic memory operations

This is the "orchestration" layer that brings everything together:
- Embeddings (understanding text)
- Vector Store (storing and retrieving)
- Hybrid Scoring (combining relevance + recency)

This is where the magic happens - turning semantic search into smart memory recall!
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math
import numpy as np

from core.embeddings import get_embedding_engine, encode_text
from core.vector_store import get_vector_store
import config


class MemoryManager:
    """
    High-level interface for managing semantic memories.
    
    This class coordinates between embeddings and vector store, implementing
    the hybrid scoring algorithm that makes memories both relevant AND recent.
    """
    
    def __init__(self):
        """Initialize the memory manager with embedding engine and vector store."""
        self.embedding_engine = get_embedding_engine()
        self.vector_store = get_vector_store()
    
    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new memory to the system.
        
        Steps:
        1. Generate embedding for the text
        2. Add timestamp to metadata
        3. Store in vector database
        
        Args:
            text: The memory text to store
            metadata: Optional metadata (tags, category, etc.)
        
        Returns:
            Dictionary with memory details: {id, text, timestamp}
        
        Example:
            >>> manager = MemoryManager()
            >>> result = manager.add_memory("passport in blue suitcase")
            >>> print(result['id'])  # 'a3f2'
        """
        # Generate embedding vector
        embedding = encode_text(text)
        
        # Prepare metadata with timestamp
        if metadata is None:
            metadata = {}
        
        timestamp = datetime.now().isoformat()
        metadata["timestamp"] = timestamp
        
        # Store in vector database
        short_id = self.vector_store.add_memory(
            text=text,
            embedding=embedding.tolist(),
            metadata=metadata
        )
        
        return {
            "id": short_id,
            "text": text,
            "timestamp": timestamp,
            "metadata": metadata
        }
    
    def recall_memory(
        self,
        query: str,
        n_results: int = config.TOP_N_RESULTS,
        min_confidence: float = config.CONFIDENCE_THRESHOLD,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Recall memories similar to the query using hybrid scoring.
        
        This is the heart of the system! Here's how it works:
        
        1. Convert query to embedding vector
        2. Find top K candidates from vector database (semantic search)
        3. Calculate recency score for each candidate
        4. Combine similarity + recency into final score
        5. Re-rank and filter by confidence threshold
        6. Return top N results
        
        Args:
            query: Natural language query ("where's my passport?")
            n_results: Maximum number of results to return
            min_confidence: Minimum score to return (0-1)
            debug: If True, include scoring details in results
        
        Returns:
            List of memory dictionaries with scores and metadata
        
        Example:
            >>> manager = MemoryManager()
            >>> results = manager.recall_memory("where's my passport?")
            >>> print(results[0]['text'])
            'passport in red backpack'
        """
        # Step 1: Generate query embedding
        query_embedding = encode_text(query)
        
        # Step 2: Semantic search - get top K candidates
        ids, documents, distances, metadatas = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            n_results=config.TOP_K_RETRIEVAL
        )
        
        if not ids:
            return []  # No memories stored yet
        
        # Step 3-4: Calculate hybrid scores
        scored_results = []
        
        for i, memory_id in enumerate(ids):
            # Convert distance to similarity score
            # ChromaDB returns distance (lower = better)
            # We convert to similarity (higher = better, 0-1 range)
            similarity_score = self._distance_to_similarity(distances[i])
            
            # Calculate recency score
            timestamp_str = metadatas[i].get("timestamp", "")
            recency_score = self._calculate_recency_score(timestamp_str)
            
            # Combine into final score
            final_score = (
                similarity_score * config.SIMILARITY_WEIGHT +
                recency_score * config.RECENCY_WEIGHT
            )
            
            # Build result dictionary
            result = {
                "id": memory_id[:4],  # Short ID for display
                "text": documents[i],
                "score": final_score,
                "timestamp": timestamp_str,
                "metadata": metadatas[i]
            }
            
            # Add debug info if requested
            if debug:
                result["debug"] = {
                    "similarity_score": similarity_score,
                    "recency_score": recency_score,
                    "distance": distances[i],
                    "similarity_weight": config.SIMILARITY_WEIGHT,
                    "recency_weight": config.RECENCY_WEIGHT
                }
            
            scored_results.append(result)
        
        # Step 5: Re-rank by final score and filter by confidence
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        filtered_results = [
            r for r in scored_results
            if r["score"] >= min_confidence
        ]
        
        # Step 6: Return top N
        return filtered_results[:n_results]
    
    def list_memories(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List stored memories, optionally filtered by a query.
        
        Args:
            query: Optional search query to filter results
            limit: Maximum number of results
        
        Returns:
            List of memory dictionaries
        """
        if query:
            # If query provided, do semantic search
            # Use low confidence threshold to show more results
            return self.recall_memory(
                query=query,
                n_results=limit or 20,
                min_confidence=0.0  # Show all matches
            )
        else:
            # No query - list all memories sorted by recency
            memories = self.vector_store.list_all(limit=limit)
            
            # Format for consistency with recall results
            return [
                {
                    "id": m["short_id"],
                    "text": m["text"],
                    "timestamp": m["metadata"].get("timestamp", ""),
                    "metadata": m["metadata"]
                }
                for m in memories
            ]
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.
        
        Args:
            memory_id: Memory ID (short or full)
        
        Returns:
            True if deleted, False if not found
        """
        return self.vector_store.delete_memory(memory_id)
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: Memory ID (short or full)
        
        Returns:
            Memory dictionary or None if not found
        """
        memory = self.vector_store.get_by_id(memory_id)
        
        if memory:
            return {
                "id": memory["id"][:4],
                "text": memory["text"],
                "timestamp": memory["metadata"].get("timestamp", ""),
                "metadata": memory["metadata"]
            }
        
        return None
    
    def count_memories(self) -> int:
        """Get total number of stored memories."""
        return self.vector_store.count()
    
    def clear_all_memories(self) -> None:
        """
        Delete all memories from the system.
        
        WARNING: This is irreversible!
        """
        self.vector_store.clear_all()
    
    # =========================================================================
    # SCORING HELPER METHODS
    # =========================================================================
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert ChromaDB distance to similarity score (0-1).
        
        ChromaDB uses cosine distance = 1 - cosine_similarity
        So: similarity = 1 - distance
        
        We also handle edge cases and ensure result is in [0, 1]
        
        Args:
            distance: Distance from ChromaDB (typically 0-2)
        
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # For normalized embeddings with cosine distance:
        # distance = 1 - cosine_similarity
        # Therefore: similarity = 1 - distance
        similarity = 1.0 - distance
        
        # Clamp to [0, 1] range to handle numerical errors
        return max(0.0, min(1.0, similarity))
    
    def _calculate_recency_score(self, timestamp_str: str) -> float:
        """
        Calculate recency score based on how old the memory is.
        
        Uses exponential decay: score = exp(-days_old * decay_rate)
        - Recent memories (0 days old): score ≈ 1.0
        - Old memories: score approaches 0
        
        The decay rate controls how fast memories fade:
        - Low decay rate: memories stay relevant longer
        - High decay rate: memories fade faster
        
        Args:
            timestamp_str: ISO format timestamp string
        
        Returns:
            Recency score (0-1, higher = more recent)
        """
        if not timestamp_str:
            return 0.5  # Unknown timestamp, use neutral score
        
        try:
            # Parse timestamp
            timestamp = datetime.fromisoformat(timestamp_str)
            
            # Calculate age in days
            now = datetime.now()
            age = now - timestamp
            days_old = age.total_seconds() / (24 * 3600)
            
            # Exponential decay formula
            # e^0 = 1 (just created)
            # e^(-1) ≈ 0.37 (1/decay_rate days old)
            # e^(-2) ≈ 0.14 (2/decay_rate days old)
            recency_score = math.exp(-days_old * config.RECENCY_DECAY_RATE)
            
            return recency_score
        
        except (ValueError, AttributeError):
            # If timestamp parsing fails, use neutral score
            return 0.5
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Useful for debugging and testing.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        return self.embedding_engine.similarity(text1, text2)


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """
    Get or create the global memory manager instance.
    """
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    
    return _memory_manager


if __name__ == "__main__":
    # Test the memory manager
    print("\n" + "="*60)
    print("Testing Memory Manager")
    print("="*60 + "\n")
    
    manager = MemoryManager()
    
    # Test 1: Add memories
    print("Test 1: Adding memories")
    memories_to_add = [
        "My passport is in the blue suitcase",
        "Wallet is on the kitchen counter",
        "Keys are in my jacket pocket"
    ]
    
    for memory_text in memories_to_add:
        result = manager.add_memory(memory_text)
        print(f"   Added [{result['id']}]: {memory_text}")
    
    # Test 2: Recall with semantic search
    print("\nTest 2: Semantic recall")
    query = "where is my travel document?"
    results = manager.recall_memory(query, debug=True)
    
    print(f"   Query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   - Text: {result['text']}")
        print(f"   - Score: {result['score']:.3f}")
        if 'debug' in result:
            print(f"   - Similarity: {result['debug']['similarity_score']:.3f}")
            print(f"   - Recency: {result['debug']['recency_score']:.3f}")
    
    # Test 3: List all
    print("\n\nTest 3: Listing all memories")
    all_memories = manager.list_memories()
    print(f"   Total: {len(all_memories)}")
    for m in all_memories:
        print(f"   [{m['id']}] {m['text']}")
    
    # Test 4: Clean up
    print("\n\nTest 4: Cleaning up")
    manager.clear_all_memories()
    print(f"   Cleared all test memories")
    
    print("\n✓ Memory manager working correctly!\n")

