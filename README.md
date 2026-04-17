# VegaMini

> **VegaMini** is a research prototype for adaptive learning and memory consolidation using swarm-based candidate generation, consensus voting, and persistent memory.

## Features
- **Swarm Intelligence**: Parallel candidate generation using anchored flow matching.
- **Consensus Voting**: Single Transferable Vote (STV) to identify stable latent clusters.
- **Persistent Memory**: "Lighthouse" system combining FAISS (vector search) and SQLite (metadata).
- **Sleep Cycles**: Automated consolidation through replay, dreaming (generative), and nightmares (adversarial).
- **Observability**: Structured JSONL/CSV logging and interactive Plotly dashboards.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize the database
```bash
python init_db.py
```

### 3. Run a query
```bash
python run.py --query "What is the pattern: 2, 4, 6, _?" --task patterns
```

### 4. Run the sleep cycle
```bash
python -m vega_mini.sleep
```

## Module Overview for Contributors

- **`vega_mini/controller/`**: Core neural logic. `flow.py` implements the anchored ODE solver.
- **`vega_mini/memory/`**: Memory persistence. `punk.py` handles the FAISS/SQLite integration.
- **`vega_mini/sleep/`**: Offline optimization. Modules for replay (`consolidate.py`), perturbation (`dream.py`), and adversarial testing (`nightmare.py`).
- **`vega_mini/logging/`**: Structured observability foundation.
- **`vega_mini/vis/`**: Dashboard and trajectory plotting tools.

## Example: Accessing Logs and Visualizations

After running the system, check the following directories:
- `logs/{session_id}/events.jsonl`: Full trace of every system event.
- `logs/{session_id}/metrics.csv`: Time-series of quality and voting margins.
- `visualizations/trajectories_{task}_{size}.html`: Interactive PCA plot of latent paths.

## Contributing
We welcome contributions to the flow solver, quality model, and visualization suite. Please ensure all new modules include docstrings and update the `ARCHITECTURE.md` if core contracts change.
