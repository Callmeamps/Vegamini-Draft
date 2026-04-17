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

### 2. Initialize the database and bootstrap quality model
```bash
python init_db.py
python train_quality.py --bootstrap 1000
```

### 3. Run Generation 0: Full Factorial DOE
```bash
python build_doe.py
```
This will generate a broad sweep of configs, run all, and output `portfolio.json` and `doe_portfolio.csv` with the top 5 variants.

### 4. Evolve Portfolio (Generation 1+)
```bash
python evolve_portfolio.py
```
This mutates and tests around the current best variants. Repeat until quality plateaus.

### 5. Run a specific config (production)
```bash
python run.py --config configs/cell_<best_id>.json --task arc --file train.json
```

### 6. Run the sleep cycle (consolidation, pruning)
```bash
python -m vega_mini.sleep --config configs/cell_<best_id>.json --prune
```

---

## DOE Playbook Workflow (v1.0)

**Goal:** Escape quality plateaus by running a full factorial DOE, iterating, and retaining a diverse portfolio of top configs.

**Key scripts:**
- `build_doe.py`: Generation 0, broad sweep, outputs `portfolio.json` and `doe_portfolio.csv`
- `evolve_portfolio.py`: Generation 1+, hill climbs around winners
- `run.py`: Runs a specific config
- `sleep/consolidate.py`: Prunes storage, keeps only portfolio and top lighthouses

**Outputs:**
- `doe_portfolio.csv`: Table of top configs and stats
- `portfolio.json`: List of cell_ids to keep alive

**Typical workflow:**
1. Initialize DB and quality model
2. Run `build_doe.py` (broad sweep)
3. Run `evolve_portfolio.py` (refine winners, repeat as needed)
4. Use best config for production
5. Prune storage with sleep cycle

See `DOE_PLAYBOOK.md` for full rationale and details.

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
- `doe_portfolio.csv`: Table of top configs from DOE.
- `portfolio.json`: List of cell_ids to keep alive.
- `visualizations/trajectories_{task}_{size}.html`: Interactive PCA plot of latent paths.

## Contributing
We welcome contributions to the flow solver, quality model, and visualization suite. Please ensure all new modules include docstrings and update the `ARCHITECTURE.md` if core contracts change.
