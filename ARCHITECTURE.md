# Second Brain Architecture

## 🎯 Purpose
This document explains how your semantic memory system works under the hood. Perfect for learning RAG (Retrieval Augmented Generation) concepts with a real, practical application.

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                │
│                     (CLI Interface)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   MEMORY MANAGER                            │
│              (Business Logic Layer)                         │
│  • Add memory with review step                              │
│  • Recall with hybrid scoring                               │
│  • List/Delete operations                                   │
└────────────┬───────────────────────────┬────────────────────┘
             │                           │
             ▼                           ▼
┌────────────────────────┐   ┌──────────────────────────────┐
│   EMBEDDINGS ENGINE    │   │    VECTOR STORE (ChromaDB)   │
│  sentence-transformers │   │   Persistent Local Storage   │
│   (MiniLM-L6-v2)       │   │                              │
│                        │   │  Stores:                     │
│  Converts text →       │   │  • Text                      │
│  384-dim vector        │   │  • Embeddings (vectors)      │
│                        │   │  • Metadata (timestamp, etc) │
└────────────────────────┘   └──────────────────────────────┘
```

---

## 📚 RAG Concepts Explained

### What is RAG?
**Retrieval Augmented Generation** = Retrieve relevant info + Generate response

For this "Lite" version, we're doing **RAG without the Generation**:
- ✅ **Retrieval**: Find semantically similar memories
- ❌ **Generation**: No LLM (future version)

### The Three Core Components

#### 1️⃣ **Embeddings** (The "Understanding" Layer)
Converts text into mathematical vectors that capture meaning.

```python
# Example (simplified):
"passport in blue suitcase"  →  [0.23, -0.15, 0.87, ...]  (384 numbers)
"where is my passport?"      →  [0.21, -0.13, 0.89, ...]  (similar!)
"what's for dinner?"         →  [-0.45, 0.67, -0.12, ...] (different)
```

**Why?** Computers can measure similarity between vectors using math (cosine similarity).

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Fast (runs on CPU)
- Small (384 dimensions)
- Accurate enough for our use case
- Fully local (no API calls)

#### 2️⃣ **Vector Store** (The "Memory" Layer)
Stores memories with their embeddings for fast similarity search.

```python
# When you add: "passport in blue suitcase"
{
    "id": "a3f2",
    "text": "passport in blue suitcase",
    "embedding": [0.23, -0.15, 0.87, ...],  # 384 numbers
    "timestamp": "2025-10-13T20:32:00",
    "metadata": {}
}
```

**ChromaDB** handles:
- Storing vectors on disk (`./data/chroma/`)
- Fast similarity search (finds nearest vectors)
- Persistence between runs

#### 3️⃣ **Hybrid Scoring** (The "Smart Recall" Layer)
Combines semantic similarity + recency to find the best answer.

```python
# For each memory, calculate:
similarity_score = cosine_similarity(query_vector, memory_vector)  # 0-1
recency_score = exp(-days_old * decay_rate)  # Newer = higher

final_score = (similarity_score * 0.7) + (recency_score * 0.3)
```

**Example:**
```
Query: "where's my passport?"

Memory 1: "passport in blue suitcase" (30 days ago)
  - Similarity: 0.95 (very relevant)
  - Recency: 0.30 (old)
  - Final: 0.95*0.7 + 0.30*0.3 = 0.755

Memory 2: "passport in red backpack" (1 day ago)
  - Similarity: 0.92 (very relevant)  
  - Recency: 0.95 (recent)
  - Final: 0.92*0.7 + 0.95*0.3 = 0.929 ✅ WINNER
```

---

## 🗂️ Project Structure

```
second-brain-v0/
├── config.py                 # All tunable parameters in one place
│                            # (similarity weights, decay rates, thresholds)
│
├── core/                     # Core RAG components
│   ├── embeddings.py         # Embedding model initialization & encoding
│   ├── vector_store.py       # ChromaDB wrapper (add, search, delete)
│   └── memory_manager.py     # Business logic (hybrid scoring, CRUD ops)
│
├── cli/                      # User interface
│   └── brain.py             # Command-line interface
│
├── tests/                    # Test scenarios
│   └── test_memory.py       # Passport example, conflict resolution, etc.
│
├── data/                     # Auto-created at runtime
│   └── chroma/              # Vector database storage (gitignored)
│
├── requirements.txt          # Python dependencies
├── README.md                 # User guide (how to install & use)
└── ARCHITECTURE.md          # This file!
```

---

## 🔄 Data Flow: Adding a Memory

```
1. User Input
   └─> "passport in blue suitcase"

2. Preview & Confirm (CLI)
   └─> Show text, timestamp, ID
   └─> Wait for confirmation

3. Generate Embedding
   └─> embeddings.py: text → 384-dim vector

4. Store in Vector DB
   └─> vector_store.py: save to ChromaDB
   └─> Persisted to ./data/chroma/

5. Confirmation
   └─> "✓ Memory saved! [ID: a3f2]"
```

---

## 🔍 Data Flow: Recalling a Memory

```
1. User Query
   └─> "where's my passport?"

2. Generate Query Embedding
   └─> embeddings.py: query → 384-dim vector

3. Vector Similarity Search
   └─> vector_store.py: find top K similar vectors
   └─> ChromaDB returns top 10 matches

4. Hybrid Scoring
   └─> memory_manager.py: 
       - Calculate recency score for each
       - Combine with similarity score
       - Re-rank results

5. Return Top 3
   └─> Filter by confidence threshold (default: 0.3)
   └─> Format response with timestamps
   └─> "You said: 'passport in red backpack' on Oct 13, 8:15 PM"
```

---

## ⚙️ Configuration & Tuning

All tunable parameters live in `config.py`:

```python
# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Hybrid scoring weights
SIMILARITY_WEIGHT = 0.7  # How much semantic similarity matters
RECENCY_WEIGHT = 0.3     # How much recency matters

# Recency decay
RECENCY_DECAY_RATE = 0.1  # How fast old memories fade (higher = faster)

# Search parameters
TOP_K_RETRIEVAL = 10      # How many candidates to retrieve from vector DB
TOP_N_RESULTS = 3         # How many to show to user
CONFIDENCE_THRESHOLD = 0.3  # Minimum score to return

# Storage
DATA_DIR = "./data/chroma"
COLLECTION_NAME = "memories"
```

**What happens if you change these?**

- ↑ `RECENCY_WEIGHT` → More emphasis on recent memories
- ↑ `RECENCY_DECAY_RATE` → Old memories fade faster
- ↑ `CONFIDENCE_THRESHOLD` → Fewer, more confident results
- ↑ `TOP_N_RESULTS` → More results returned

You can experiment to find what works for your use case!

---

## 🎯 Design Decisions

### Why No "Update" Command?
**Problem**: You move your passport from blue suitcase → red backpack.

**Bad Solution**: Find and update the old memory.
- Complex: Which memory to update?
- Error-prone: What if it updates the wrong one?

**Good Solution**: Just add a new memory.
- Recency scoring naturally prioritizes the latest
- Historical record preserved (you *did* put it in blue suitcase once)
- Simple: no ambiguity

### Why Short IDs?
Full UUIDs: `550e8400-e29b-41d4-a716-446655440000` (ugly!)
Short IDs: `a3f2` (human-friendly)

We generate UUIDs internally but display only first 4 chars. Good enough for uniqueness in personal use case (16^4 = 65,536 possibilities).

### Why ChromaDB?
- ✅ Simple Python API
- ✅ Local persistence built-in
- ✅ No server needed (embedded mode)
- ✅ Fast enough for 10k+ memories
- ✅ Good documentation for learners

Alternatives considered:
- **FAISS**: Faster, but no built-in persistence (more setup)
- **Qdrant**: Great but overkill for local-only use case
- **Pinecone/Weaviate**: Cloud-based (defeats our "fully local" goal)

---

## 🧪 Testing Strategy

We'll test real-world scenarios:

### Test Cases

1. **Basic Add & Recall**
   - Add: "passport in blue suitcase"
   - Recall: "where's my passport?" → should match

2. **Semantic Similarity**
   - Add: "I put my passport in the blue suitcase"
   - Recall: "where did I leave my travel document?" → should still match

3. **Recency Prioritization**
   - Add: "passport in blue suitcase" (simulate old timestamp)
   - Add: "passport in red backpack" (now)
   - Recall: "where's my passport?" → should return red backpack

4. **No Match Handling**
   - Query: "where's my spaceship?" (never mentioned)
   - Should return: "No confident matches found"

5. **Multiple Items**
   - Add 10 different items in different locations
   - Recall each one → verify correct retrieval

6. **Confidence Threshold**
   - Query something vaguely related but not exact
   - Verify it respects threshold setting

---

## 🚀 Future Enhancements

### Phase 2: Conversational Interface
Add LLM layer for natural language responses:
```
Current: "You said: 'passport in red backpack' on Oct 13"
Future:  "Your passport is in the red backpack. You mentioned this yesterday evening."
```

### Phase 3: Categories/Tags
```bash
brain add "passport in backpack" --category travel
brain recall "passport" --category travel  # Only search travel memories
```

### Phase 4: Voice Interface
- Speech-to-text input (Whisper)
- Text-to-speech output

### Phase 5: Metadata Enrichment
- Auto-capture location (GPS)
- File attachments
- Photos

---

## 📖 Learning Resources

If you want to dive deeper into the concepts:

**Embeddings:**
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Word Embeddings Explained](https://jalammar.github.io/illustrated-word2vec/)

**Vector Databases:**
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Understanding Vector Databases](https://www.pinecone.io/learn/vector-database/)

**RAG:**
- [What is RAG?](https://www.anthropic.com/index/retrieval-augmented-generation)
- [Building RAG Applications](https://python.langchain.com/docs/use_cases/question_answering/)

---

## 🐛 Debugging Tips

### Memory not recalling correctly?
1. Check similarity scores: `brain recall --debug "query"`
2. Adjust `SIMILARITY_WEIGHT` vs `RECENCY_WEIGHT` in config
3. Lower `CONFIDENCE_THRESHOLD` to see more marginal matches

### Embeddings slow?
- First run downloads the model (~80MB)
- Subsequent runs use cached model (fast)
- CPU inference should be <100ms per encoding

### ChromaDB errors?
- Check `./data/chroma/` permissions
- Delete and reinitialize if corrupted: `rm -rf ./data/chroma/`

---

## 🤝 Contributing

This is a learning project! Feel free to:
- Experiment with different embedding models
- Try different scoring algorithms
- Add new commands
- Improve the CLI interface

Every file is heavily commented to explain *why* not just *what*.

---

**Questions?** Check the inline code comments or ask in issues!

