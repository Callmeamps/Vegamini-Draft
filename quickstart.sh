#!/bin/bash
# VegaMini Quickstart Script
set -e

# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize the database and FAISS index
python init_db.py

# 3. Bootstrap the quality model (optional, for cold start)
python train_quality.py --bootstrap 1000

# 4. Run the day loop (main inference and learning)
python run.py --task arc --file sample_tasks.json

# 5. Run the sleep cycle (consolidation, dreaming, pruning)
python sleep.py

echo "VegaMini quickstart complete!"
