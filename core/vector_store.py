"""
Vector Store - Persistent storage for semantic memories

This is the "memory" layer of the RAG system. It uses ChromaDB to store text
along with their embeddings (vectors) and enables fast similarity search.

Key Concepts:
- Vector database: Specialized DB for storing and searching high-dimensional vectors
- Similarity search: Find vectors closest to a query vector (nearest neighbors)
- Persistence: Data saved to disk, survives between app restarts
- Metadata: Additional info stored with each memory (timestamp, tags, etc.)
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import config


class VectorStore:
    """
    Wrapper around ChromaDB for storing and retrieving semantic memories.
    
    ChromaDB handles the heavy lifting of:
    - Storing vectors efficiently
    - Fast similarity search (approximate nearest neighbors)
    - Persistence to disk
    """
    
    def __init__(
        self,
        persist_directory: str = str(config.DATA_DIR),
        collection_name: str = config.COLLECTION_NAME
    ):
        """
        Initialize the vector store with persistent storage.
        
        Args:
            persist_directory: Where to store the database on disk
            collection_name: Name of the collection (like a table name)
        """
        print(f"📦 Initializing vector store at: {persist_directory}")
        
        # Create ChromaDB client with persistent storage
        # This means your memories are saved to disk and persist between runs
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable usage tracking
                allow_reset=True  # Allow clearing the database
            )
        )
        
        # Get or create the collection
        # A collection is like a table in a traditional database
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Semantic memory storage"}
        )
        
        print(f"✓ Vector store ready! Collection: '{collection_name}'")
        print(f"   Existing memories: {self.collection.count()}")
    
    def add_memory(
        self,
        text: str,
        embedding: List[float],
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new memory to the store.
        
        Args:
            text: The original text of the memory
            embedding: Vector representation of the text (from embeddings.py)
            memory_id: Optional custom ID (auto-generated if not provided)
            metadata: Optional metadata (timestamp, tags, etc.)
        
        Returns:
            The memory ID (4-char short ID for display)
        
        Example:
            >>> store = VectorStore()
            >>> embedding = encode_text("passport in blue suitcase")
            >>> memory_id = store.add_memory(
            ...     text="passport in blue suitcase",
            ...     embedding=embedding.tolist(),
            ...     metadata={"timestamp": "2025-10-13T20:32:00"}
            ... )
        """
        # Generate a unique ID if not provided
        if memory_id is None:
            memory_id = str(uuid.uuid4())
        
        # Add timestamp to metadata if not present
        if metadata is None:
            metadata = {}
        
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Add to ChromaDB
        # ChromaDB stores: documents (text), embeddings (vectors), metadata, and IDs
        self.collection.add(
            ids=[memory_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        
        # Return short ID (first 4 chars) for user-friendly display
        short_id = memory_id[:4]
        return short_id
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = config.TOP_K_RETRIEVAL,
        where: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[str], List[float], List[Dict[str, Any]]]:
        """
        Search for similar memories using vector similarity.
        
        This is the core of semantic search! ChromaDB finds memories whose
        embeddings are closest to the query embedding (using cosine similarity).
        
        Args:
            query_embedding: Vector representation of the search query
            n_results: How many results to return
            where: Optional metadata filters (e.g., {"category": "travel"})
        
        Returns:
            Tuple of (ids, documents, distances, metadatas)
            - ids: Memory IDs (full UUIDs)
            - documents: Original text of memories
            - distances: Similarity distances (lower = more similar)
            - metadatas: Associated metadata for each memory
        
        Note: ChromaDB returns "distance" where lower is better.
        We'll convert this to similarity score (0-1) in memory_manager.py
        """
        # Query ChromaDB for nearest neighbors
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "distances", "metadatas"]
        )
        
        # ChromaDB returns nested lists, flatten them
        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        return ids, documents, distances, metadatas
    
    def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by its ID.
        
        Useful for displaying memory details or deleting specific memories.
        
        Args:
            memory_id: Full UUID or short ID (will search for prefix match)
        
        Returns:
            Dictionary with {id, text, metadata} or None if not found
        """
        # If short ID provided, find the full ID
        if len(memory_id) == 4:
            all_memories = self.collection.get()
            matching_ids = [id for id in all_memories["ids"] if id.startswith(memory_id)]
            
            if not matching_ids:
                return None
            
            memory_id = matching_ids[0]  # Use the first match
        
        # Get the memory
        result = self.collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"]
        )
        
        if not result["ids"]:
            return None
        
        return {
            "id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0]
        }
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: Full UUID or short ID
        
        Returns:
            True if deleted, False if not found
        """
        # Handle short IDs
        if len(memory_id) == 4:
            all_memories = self.collection.get()
            matching_ids = [id for id in all_memories["ids"] if id.startswith(memory_id)]
            
            if not matching_ids:
                return False
            
            memory_id = matching_ids[0]
        
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False
    
    def list_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all memories in the store.
        
        Args:
            limit: Optional limit on number of results
        
        Returns:
            List of dictionaries with {id, text, metadata}
        """
        result = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        memories = []
        for i, memory_id in enumerate(result["ids"]):
            memories.append({
                "id": memory_id,
                "short_id": memory_id[:4],
                "text": result["documents"][i],
                "metadata": result["metadatas"][i]
            })
        
        # Sort by timestamp (most recent first)
        memories.sort(
            key=lambda x: x["metadata"].get("timestamp", ""),
            reverse=True
        )
        
        return memories
    
    def count(self) -> int:
        """Get the total number of memories stored."""
        return self.collection.count()
    
    def clear_all(self) -> None:
        """
        Delete all memories from the store.
        
        WARNING: This is irreversible!
        """
        # Get all IDs
        all_memories = self.collection.get()
        
        if all_memories["ids"]:
            self.collection.delete(ids=all_memories["ids"])
    
    def reset(self) -> None:
        """
        Completely reset the vector store (delete collection and recreate).
        
        WARNING: This is irreversible!
        """
        # Delete the collection
        self.client.delete_collection(name=self.collection.name)
        
        # Recreate it
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"description": "Semantic memory storage"}
        )


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_vector_store = None

def get_vector_store() -> VectorStore:
    """
    Get or create the global vector store instance.
    
    This ensures we only initialize the database once.
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = VectorStore()
    
    return _vector_store


if __name__ == "__main__":
    # Quick test to verify the vector store works
    print("\n" + "="*60)
    print("Testing Vector Store")
    print("="*60 + "\n")
    
    from embeddings import encode_text
    
    store = VectorStore()
    
    # Test 1: Add a memory
    print("Test 1: Adding a memory")
    text = "My passport is in the blue suitcase"
    embedding = encode_text(text)
    memory_id = store.add_memory(
        text=text,
        embedding=embedding.tolist(),
        metadata={"test": True}
    )
    print(f"   Added: '{text}'")
    print(f"   ID: {memory_id}")
    
    # Test 2: Search for similar memory
    print("\nTest 2: Searching for similar memory")
    query = "where is my travel document?"
    query_embedding = encode_text(query)
    ids, docs, distances, metas = store.search(query_embedding.tolist(), n_results=1)
    
    print(f"   Query: '{query}'")
    print(f"   Found: '{docs[0]}'")
    print(f"   Distance: {distances[0]:.3f} (lower = more similar)")
    
    # Test 3: List all
    print("\nTest 3: Listing all memories")
    all_memories = store.list_all()
    print(f"   Total memories: {len(all_memories)}")
    
    # Test 4: Clean up
    print("\nTest 4: Cleaning up")
    store.delete_memory(memory_id)
    print(f"   Deleted test memory")
    
    print("\n✓ Vector store working correctly!\n")

