# Second Brain v0

Your personal semantic memory assistant. Remember anything, recall it naturally.

> "Where did I put my passport?"  
> → "You said: 'passport in the blue suitcase' on Oct 13, 8:02 PM"

## Features

- **Semantic search**: Find memories even when worded differently
- **Recency-aware**: Newer memories automatically prioritized  
- **100% local & private**: No API calls, no internet required
- **Fast**: Sub-second query responses
- **Simple**: Clean CLI interface, no complexity

## What You'll Learn

This project is designed as a learning tool for understanding RAG (Retrieval Augmented Generation):

- **Embeddings**: How text becomes semantic vectors
- **Vector databases**: Efficient similarity search
- **Hybrid scoring**: Combining multiple ranking signals
- **Local AI**: Running ML models without APIs

Every file is heavily commented to explain not just *what* it does, but *why*.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- ~500MB disk space (for embedding model)

### Installation

```bash
# Clone the repository
git clone https://github.com/EHam1/external-brain-v0.git
cd external-brain-v0

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: First run will download the embedding model (~80MB). This is cached for future use.

### Basic Usage

```bash
# Add a memory
python -m cli.brain add "My passport is in the blue suitcase"

# Recall a memory
python -m cli.brain recall "where's my passport?"

# List all memories
python -m cli.brain list

# Show statistics
python -m cli.brain stats
```

## Usage Guide

### Adding Memories

```bash
# Simple addition
python -m cli.brain add "wallet on kitchen counter"

# Skip confirmation prompt
python -m cli.brain add "keys in jacket pocket" --no-confirm
```

### Recalling Memories

```bash
# Natural language query
python -m cli.brain recall "where did I put my wallet?"

# Show more results
python -m cli.brain recall "passport" --limit 5

# Lower confidence threshold for more results
python -m cli.brain recall "passport" --threshold 0.2

# Debug mode (show scoring details)
python -m cli.brain recall "passport" --debug
```

### Managing Memories

```bash
# List all memories
python -m cli.brain list

# Search memories
python -m cli.brain list "passport"

# Limit results
python -m cli.brain list --limit 10

# Delete specific memory
python -m cli.brain delete a3f2

# Clear all memories (careful!)
python -m cli.brain clear
```

## Real-World Examples

### Moving House

```bash
python -m cli.brain add "winter clothes in box marked 'bedroom'"
python -m cli.brain add "kitchen utensils in box marked 'K-1'"
python -m cli.brain add "important documents in blue folder on desk"

# Later...
python -m cli.brain recall "where are my winter coats?"
```

### Daily Items

```bash
python -m cli.brain add "spare car key in kitchen drawer"
python -m cli.brain add "passport in safe, combination is birthday"
python -m cli.brain add "phone charger left at office, desk drawer"

# Next day...
python -m cli.brain recall "where's my phone charger?"
```

### Reminders & To-Dos

```bash
python -m cli.brain add "need to call dentist on Monday"
python -m cli.brain add "Sarah's birthday is March 15"
python -m cli.brain add "meeting notes in Google Doc titled 'Q4 Planning'"

# Later...
python -m cli.brain recall "when is Sarah's birthday?"
```

### Updating Locations

No need to delete or update! Just add new memories - recency handles it:

```bash
# Monday
python -m cli.brain add "laptop on dining table"

# Tuesday (you moved it)
python -m cli.brain add "laptop in bedroom on desk"

# Query returns the most recent location first
python -m cli.brain recall "where's my laptop?"
# → "laptop in bedroom on desk" (from Tuesday)
```

## Configuration

Edit `config.py` to tune the system:

### Hybrid Scoring Weights

```python
SIMILARITY_WEIGHT = 0.7  # How much semantic similarity matters (0-1)
RECENCY_WEIGHT = 0.3     # How much recency matters (0-1)
```

- **Increase SIMILARITY_WEIGHT**: More accurate matches, ignore time
- **Increase RECENCY_WEIGHT**: Prioritize recent memories more

### Recency Decay Rate

```python
RECENCY_DECAY_RATE = 0.1  # How fast old memories fade
```

- **Lower (0.05)**: Memories stay relevant for weeks
- **Higher (0.2)**: Memories fade after a few days

### Search Parameters

```python
CONFIDENCE_THRESHOLD = 0.3  # Minimum score to return (0-1)
TOP_N_RESULTS = 3          # How many results to show
TOP_K_RETRIEVAL = 10       # How many candidates to consider
```

### Embedding Model

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

To try a different model, change this to any [sentence-transformers model](https://www.sbert.net/docs/pretrained_models.html).

## Testing

Run the test suite to verify everything works:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html

# Run specific test
pytest tests/test_memory.py::TestSemanticSearch -v
```

## Project Structure

```
second-brain-v0/
├── config.py              # All tunable parameters
├── core/                  # Core RAG components
│   ├── embeddings.py      # Text → vector conversion
│   ├── vector_store.py    # ChromaDB storage/retrieval
│   └── memory_manager.py  # Business logic & hybrid scoring
├── cli/                   # User interface
│   └── brain.py          # CLI commands
├── tests/                 # Test suite
│   └── test_memory.py    # Comprehensive tests
├── data/                  # Auto-created
│   └── chroma/           # Vector database (gitignored)
├── ARCHITECTURE.md        # Deep dive into how it works
└── README.md             # This file!
```

## Troubleshooting

### Model Download Fails

If the embedding model download fails:

```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python -m cli.brain add "test"
```

### ChromaDB Errors

If you get database errors:

```bash
# Reset the database
rm -rf data/chroma/
python -m cli.brain add "test"  # Reinitialize
```

### Low-Quality Results

If searches return poor results:

1. **Lower the confidence threshold**:
   ```bash
   python -m cli.brain recall "query" --threshold 0.1
   ```

2. **Check your memories**:
   ```bash
   python -m cli.brain list
   ```

3. **Try debug mode**:
   ```bash
   python -m cli.brain recall "query" --debug
   ```

4. **Adjust config.py**:
   - Increase `SIMILARITY_WEIGHT` for more relevance
   - Decrease `RECENCY_DECAY_RATE` to keep old memories relevant

## Roadmap

### v0.1 - Current (Lite) ✅
- ✅ Local semantic memory storage
- ✅ Hybrid scoring (similarity + recency)
- ✅ CLI interface
- ✅ Basic CRUD operations

### v0.2 - API & Interface (Coming Soon)
- [ ] FastAPI REST endpoints
- [ ] Simple web UI
- [ ] Export/import memories
- [ ] Categories/tags support

### v1.0 - Conversational (Future)
- [ ] LLM integration for natural responses
- [ ] Conversation history
- [ ] Multi-turn queries

### v2.0 - Advanced (Future)
- [ ] Voice input/output
- [ ] Image attachments
- [ ] Automatic summarization
- [ ] Proactive reminders
- [ ] Mobile app

## Contributing

This is a learning project! Contributions welcome:

- Report bugs via issues
- Suggest features
- Submit PRs for improvements
- Improve documentation

## Learn More

- **Architecture Deep Dive**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/)
- **Vector DB**: [ChromaDB Docs](https://docs.trychroma.com/)
- **RAG Concepts**: [What is RAG?](https://www.anthropic.com/index/retrieval-augmented-generation)

## License

MIT License - feel free to use, modify, and learn from this project!

## Acknowledgments

Built with:
- [sentence-transformers](https://www.sbert.net/) - Semantic embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Click](https://click.palletsprojects.com/) - CLI framework

---

**Made for learning and productivity**

Questions? Issues? Check out the [Architecture docs](ARCHITECTURE.md) or open an issue!

