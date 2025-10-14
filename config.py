"""
Configuration file for Second Brain

All tunable parameters live here. Experiment with these to optimize for your use case!
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Root directory of the project
ROOT_DIR = Path(__file__).parent

# Where ChromaDB stores vector data (persists between runs)
DATA_DIR = ROOT_DIR / "data" / "chroma"

# ChromaDB collection name (like a table in a database)
COLLECTION_NAME = "memories"

# =============================================================================
# EMBEDDING MODEL
# =============================================================================

# Sentence transformer model for generating embeddings
# This model converts text into 384-dimensional vectors
# - Fast: Runs on CPU, ~50ms per encoding
# - Small: ~80MB download on first run
# - Accurate: Good balance for semantic search
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =============================================================================
# HYBRID SCORING PARAMETERS
# =============================================================================

# How much to weigh semantic similarity vs recency (must sum to 1.0)
SIMILARITY_WEIGHT = 0.7  # 70% based on how similar the query is to the memory
RECENCY_WEIGHT = 0.3     # 30% based on how recent the memory is

# Recency decay rate (higher = old memories fade faster)
# Formula: recency_score = exp(-days_old * decay_rate)
# - 0.05: Gentle decay (memories from 2 weeks ago still relevant)
# - 0.1: Moderate decay (memories from 1 week ago start fading)
# - 0.2: Fast decay (memories from 3-4 days ago fade significantly)
RECENCY_DECAY_RATE = 0.1

# =============================================================================
# SEARCH PARAMETERS
# =============================================================================

# How many candidates to retrieve from vector DB before re-ranking
# Higher = more thorough but slower
TOP_K_RETRIEVAL = 10

# How many final results to show to the user
TOP_N_RESULTS = 3

# Minimum confidence score to return a result (0-1)
# Lower = more results but potentially less relevant
# Higher = fewer results but more confident matches
CONFIDENCE_THRESHOLD = 0.3

# =============================================================================
# CLI SETTINGS
# =============================================================================

# Show confirmation preview when adding memories
CONFIRM_BEFORE_SAVE = True

# Maximum characters to display in previews
PREVIEW_MAX_LENGTH = 200

# =============================================================================
# VALIDATION
# =============================================================================

# Ensure weights sum to 1.0
assert abs(SIMILARITY_WEIGHT + RECENCY_WEIGHT - 1.0) < 0.001, \
    "SIMILARITY_WEIGHT and RECENCY_WEIGHT must sum to 1.0"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

