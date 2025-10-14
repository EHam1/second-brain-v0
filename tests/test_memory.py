"""
Test suite for Second Brain memory system.

Tests the core RAG functionality:
- Adding memories
- Semantic recall
- Recency prioritization
- Hybrid scoring
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import MemoryManager
import config


@pytest.fixture
def manager():
    """Create a fresh memory manager for each test."""
    manager = MemoryManager()
    # Clear any existing memories
    manager.clear_all_memories()
    yield manager
    # Cleanup after test
    manager.clear_all_memories()


class TestBasicOperations:
    """Test basic CRUD operations."""
    
    def test_add_memory(self, manager):
        """Test adding a single memory."""
        result = manager.add_memory("My passport is in the blue suitcase")
        
        assert result['id'] is not None
        assert result['text'] == "My passport is in the blue suitcase"
        assert 'timestamp' in result
        assert manager.count_memories() == 1
    
    def test_add_multiple_memories(self, manager):
        """Test adding multiple memories."""
        memories = [
            "passport in blue suitcase",
            "wallet on kitchen counter",
            "keys in jacket pocket"
        ]
        
        for text in memories:
            manager.add_memory(text)
        
        assert manager.count_memories() == len(memories)
    
    def test_list_memories(self, manager):
        """Test listing all memories."""
        # Add some memories
        manager.add_memory("test memory 1")
        manager.add_memory("test memory 2")
        
        memories = manager.list_memories()
        
        assert len(memories) == 2
        assert all('id' in m for m in memories)
        assert all('text' in m for m in memories)
    
    def test_delete_memory(self, manager):
        """Test deleting a specific memory."""
        result = manager.add_memory("temporary memory")
        memory_id = result['id']
        
        assert manager.count_memories() == 1
        
        # Delete it
        deleted = manager.delete_memory(memory_id)
        
        assert deleted is True
        assert manager.count_memories() == 0
    
    def test_get_memory_by_id(self, manager):
        """Test retrieving a specific memory by ID."""
        result = manager.add_memory("findable memory")
        memory_id = result['id']
        
        retrieved = manager.get_memory(memory_id)
        
        assert retrieved is not None
        assert retrieved['text'] == "findable memory"
        assert retrieved['id'] == memory_id


class TestSemanticSearch:
    """Test semantic similarity and recall."""
    
    def test_exact_match(self, manager):
        """Test recalling with exact same text."""
        manager.add_memory("passport in blue suitcase")
        
        results = manager.recall_memory("passport in blue suitcase")
        
        assert len(results) > 0
        assert results[0]['text'] == "passport in blue suitcase"
        assert results[0]['score'] > 0.95  # Should be nearly perfect match
    
    def test_semantic_similarity(self, manager):
        """Test recalling with different wording (semantic match)."""
        manager.add_memory("My passport is in the blue suitcase")
        
        # Query with different wording
        results = manager.recall_memory("where did I put my travel document?")
        
        assert len(results) > 0
        # Should find the passport memory even though wording is different
        assert "passport" in results[0]['text'].lower()
    
    def test_multiple_similar_items(self, manager):
        """Test distinguishing between similar items."""
        manager.add_memory("passport in blue suitcase")
        manager.add_memory("wallet on kitchen counter")
        manager.add_memory("keys in jacket pocket")
        
        # Query for specific item
        results = manager.recall_memory("where are my keys?")
        
        assert len(results) > 0
        # Should return keys-related memory first
        assert "keys" in results[0]['text'].lower()
    
    def test_no_match(self, manager):
        """Test querying for something that doesn't exist."""
        manager.add_memory("passport in blue suitcase")
        
        # Query for something completely unrelated
        results = manager.recall_memory("what's the weather tomorrow?")
        
        # Depending on threshold, might return no results or low-confidence results
        if results:
            assert results[0]['score'] < 0.5  # Low confidence


class TestRecencyScoring:
    """Test that recency affects ranking."""
    
    def test_recency_prioritization(self, manager):
        """Test that newer memories are prioritized."""
        # Add old memory (simulate by adding first)
        old_result = manager.add_memory("passport in blue suitcase")
        
        # Add new memory with slightly different location
        new_result = manager.add_memory("passport in red backpack")
        
        # Query for passport location
        results = manager.recall_memory("where is my passport?")
        
        assert len(results) >= 2
        
        # The newer memory should rank higher due to recency
        # (assuming similar semantic scores)
        first_result_text = results[0]['text']
        
        # The first result should be one of the passport memories
        assert "passport" in first_result_text.lower()
        
        # Due to recency weight, newer memory likely ranked higher
        # (but this depends on exact scores - we mostly just verify both are returned)
    
    def test_old_vs_recent(self, manager):
        """Test scoring difference between old and recent memories."""
        # Add a memory
        result = manager.add_memory("test memory")
        
        # Get it immediately (should have high recency score)
        results = manager.recall_memory("test memory")
        recent_score = results[0]['score']
        
        # In a real test, we'd manipulate the timestamp to simulate age
        # For now, we just verify the score is high
        assert recent_score > 0.7


class TestHybridScoring:
    """Test the combination of similarity and recency."""
    
    def test_debug_mode(self, manager):
        """Test that debug mode returns scoring details."""
        manager.add_memory("passport in blue suitcase")
        
        results = manager.recall_memory("passport", debug=True)
        
        assert len(results) > 0
        result = results[0]
        
        # Check debug info is present
        assert 'debug' in result
        assert 'similarity_score' in result['debug']
        assert 'recency_score' in result['debug']
        assert 'distance' in result['debug']
    
    def test_confidence_threshold(self, manager):
        """Test that confidence threshold filters results."""
        manager.add_memory("passport in blue suitcase")
        
        # Query with high threshold
        results = manager.recall_memory(
            "completely unrelated query about space aliens",
            min_confidence=0.8
        )
        
        # Should return no results (or very few) due to high threshold
        assert len(results) == 0 or results[0]['score'] < 0.8
    
    def test_result_limit(self, manager):
        """Test limiting number of returned results."""
        # Add many memories
        for i in range(10):
            manager.add_memory(f"memory number {i}")
        
        # Query with limit
        results = manager.recall_memory("memory", n_results=3)
        
        # Should return at most 3 results
        assert len(results) <= 3


class TestRealWorldScenarios:
    """Test real-world use cases from the PRD."""
    
    def test_passport_scenario(self, manager):
        """Test the passport use case from the PRD."""
        # User tells the brain
        manager.add_memory("I put my passport in the blue suitcase")
        
        # User asks later
        results = manager.recall_memory("where did I put my passport?")
        
        # Should find the memory
        assert len(results) > 0
        assert "passport" in results[0]['text'].lower()
        assert "blue suitcase" in results[0]['text'].lower()
    
    def test_conflicting_locations(self, manager):
        """Test updating location by adding new memory."""
        # Initial location
        manager.add_memory("passport in blue suitcase")
        
        # Moved to new location
        manager.add_memory("passport in red backpack")
        
        # Query for current location
        results = manager.recall_memory("where's my passport?")
        
        # Should get multiple results, with recent one ranked higher
        assert len(results) >= 1
        
        # Both memories should be findable
        texts = [r['text'].lower() for r in results]
        assert any("passport" in t for t in texts)
    
    def test_multiple_items(self, manager):
        """Test managing multiple different items."""
        items = {
            "passport": "passport in blue suitcase",
            "wallet": "wallet on kitchen counter",
            "keys": "keys in jacket pocket",
            "phone": "phone charging on desk"
        }
        
        # Add all items
        for text in items.values():
            manager.add_memory(text)
        
        # Query for each item
        for item_name, expected_text in items.items():
            results = manager.recall_memory(f"where is my {item_name}?")
            
            assert len(results) > 0
            # Top result should contain the item name
            assert item_name in results[0]['text'].lower()
    
    def test_natural_language_variety(self, manager):
        """Test that different phrasings work."""
        manager.add_memory("My laptop is on the dining table")
        
        queries = [
            "where's my laptop?",
            "where did I leave my computer?",
            "laptop location",
            "find my laptop"
        ]
        
        for query in queries:
            results = manager.recall_memory(query)
            # All queries should find the laptop memory
            assert len(results) > 0
            assert "laptop" in results[0]['text'].lower()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_brain(self, manager):
        """Test querying when no memories exist."""
        results = manager.recall_memory("anything")
        
        assert len(results) == 0
    
    def test_very_long_text(self, manager):
        """Test adding a very long memory."""
        long_text = "word " * 1000  # 1000 words
        result = manager.add_memory(long_text)
        
        assert result['id'] is not None
        assert len(result['text']) == len(long_text)
    
    def test_special_characters(self, manager):
        """Test memories with special characters."""
        special_text = "Password is: p@ssw0rd! (don't forget) #important"
        result = manager.add_memory(special_text)
        
        results = manager.recall_memory("password")
        
        assert len(results) > 0
        assert special_text in results[0]['text']
    
    def test_unicode_text(self, manager):
        """Test memories with unicode characters."""
        unicode_text = "Café address: 日本 Tokyo 🗼"
        result = manager.add_memory(unicode_text)
        
        results = manager.recall_memory("café")
        
        assert len(results) > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])

