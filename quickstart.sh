#!/bin/bash
# VegaMini Quickstart Script (DOE Playbook Edition)
set -e

# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize the database and FAISS index
python init_db.py

# 3. Bootstrap the quality model (for cold start)
python train_quality.py --bootstrap 1000

# 4. Run Generation 0: Full Factorial DOE
python build_doe.py
# Outputs: configs, logs, doe_portfolio.csv, portfolio.json

# 5. Evolve portfolio (Generation 1+)
python evolve_portfolio.py
# Repeat as needed to refine winners

# 6. (Optional) Run a specific config for production
# python run.py --config configs/cell_<best_id>.json --task arc --file train.json

# 7. Run the sleep cycle (consolidation, pruning)
python -m vega_mini.sleep --config configs/cell_<best_id>.json --prune

echo "VegaMini DOE quickstart complete!"
