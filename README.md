# VegaMini

> **VegaMini** is a research prototype for adaptive learning and memory consolidation using swarm-based candidate generation, consensus voting, and persistent memory. For technical details, see ARCHITECTURE.md.

## Features
- Swarm-based candidate generation and voting
- Persistent memory with FAISS and SQLite
- Sleep cycle for replay, dreaming, and pruning
- Modular, extensible codebase

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize the database and FAISS index
```bash
python init_db.py
```

### 3. Bootstrap the quality model (optional, for cold start)
```bash
python train_quality.py --bootstrap 1000
```

### 4. Run the day loop (main inference and learning)
```bash
python run.py --task arc --file sample_tasks.json
```

### 5. Run the sleep cycle (consolidation, dreaming, pruning)
```bash
python sleep.py
```

## Project Structure

```
vega_mini/
├── controller/      # Transformer and flow logic
├── memory/          # Lighthouse memory and management
├── sleep/           # Dream, nightmare, and consolidation modules
├── eval/            # Quality model
├── run.py           # Main day loop
├── sleep.py         # Night loop
```

Other files:
- init_db.py — Initialize database and FAISS index
- train_quality.py — Train or bootstrap the quality model
- requirements.txt — Python dependencies
- ARCHITECTURE.md — Full technical and algorithmic details

## Citation
If you use or build on VegaMini, please cite the repository and reference the architecture document.

## License
MIT License (see LICENSE file)
