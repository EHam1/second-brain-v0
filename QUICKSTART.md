# Quick Start Guide

Get up and running with Second Brain in 5 minutes!

## 1. Setup (One Time)

```bash
# Navigate to the project
cd ~/Repositories/external-brain-v0

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

**First run note**: The embedding model (~80MB) will download automatically. This only happens once!

## 2. Try It Out

```bash
# Add your first memory
./brain add "My passport is in the blue suitcase"

# Add a few more
./brain add "Wallet on the kitchen counter"
./brain add "Keys in my jacket pocket"
./brain add "Phone charging on my desk"

# Now recall them with natural language
./brain recall "where's my passport?"
./brain recall "where did I put my wallet?"

# List all memories
./brain list

# Search for specific memories
./brain list "keys"

# See statistics
./brain stats
```

## 3. Real Usage Pattern

### Morning Routine
```bash
# Before leaving the house
./brain add "left spare key with neighbor Bob"
./brain add "turned off stove"
```

### During the Day
```bash
# Quick notes
./brain add "parked in lot B, level 3, space 47"
./brain add "doctor appointment next Tuesday at 2pm"
./brain add "need to call Sarah about dinner plans"
```

### When You Need to Remember
```bash
# Natural language queries
./brain recall "where did I park?"
./brain recall "when is my doctor appointment?"
./brain recall "who was I supposed to call?"
```

## 4. Power User Tips

### Debug Mode
See exactly how the scoring works:
```bash
./brain recall "passport" --debug
```

### Adjust Sensitivity
Find more matches with lower threshold:
```bash
./brain recall "passport" --threshold 0.1
```

### More Results
Show top 5 instead of 3:
```bash
./brain recall "passport" --limit 5
```

### Skip Confirmation
Add memories faster:
```bash
./brain add "quick memory" --no-confirm
```

## 5. Customization

Edit `config.py` to tune the system:

```python
# Make recency matter more (prioritize recent memories)
SIMILARITY_WEIGHT = 0.5
RECENCY_WEIGHT = 0.5

# Make old memories fade faster
RECENCY_DECAY_RATE = 0.2

# Be more selective with results
CONFIDENCE_THRESHOLD = 0.5
```

## 6. Testing

Verify everything works:
```bash
# Run the test suite
pytest tests/ -v

# Or test individual components
python core/embeddings.py
python core/vector_store.py
python core/memory_manager.py
```

## Example Session

```bash
$ ./brain add "passport in blue suitcase"
✓ Memory saved! [ID: a3f2]

$ ./brain add "wallet on kitchen counter"
✓ Memory saved! [ID: b8e1]

$ ./brain recall "where's my travel document?"
🔍 Found 1 matching memories:

1. My passport is in the blue suitcase
   [a3f2] · just now · Score: 0.89

$ ./brain stats
📊 Second Brain Statistics

Total memories: 2
Latest memory: just now

Storage location: /Users/you/Repositories/second-brain-v0/data/chroma
Embedding model: all-MiniLM-L6-v2
```

## Troubleshooting

### "Module not found" error
Make sure you activated the virtual environment:
```bash
source venv/bin/activate  # Run this first!
```

### "Command not found: ./brain"
Make sure you're in the project directory:
```bash
cd ~/Repositories/external-brain-v0
```

Or use the full command:
```bash
python -m cli.brain add "your memory"
```

### Model download is slow
Be patient on first run (80MB download). Subsequent runs are instant.

### Searches return nothing
Try lowering the threshold:
```bash
./brain recall "query" --threshold 0.1
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand how it works
- Check [README.md](README.md) for complete documentation
- Run `pytest` to see all the test scenarios
- Experiment with `config.py` settings
- Start using it daily!

## Pro Tips

1. **Be descriptive**: "passport in blue suitcase" > "passport there"
2. **Use natural language**: The system understands context
3. **Don't worry about updates**: Just add new memories, recency handles it
4. **Check stats regularly**: `./brain stats` shows what's stored
5. **Backup your memories**: The `data/` folder contains everything

---

**You're all set! Start building your second brain!**

