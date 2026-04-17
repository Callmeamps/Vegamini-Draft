#!/bin/bash

echo "🌟 VegaMini v0.1 Setup and Run"

# Step 1: Initialize database
echo "Step 1: Initializing database..."
python init_db.py

# Step 2: Bootstrap quality model
echo "Step 2: Bootstrapping quality model..."
python train_quality.py --bootstrap 1000

# Step 3: Create sample task data
echo "Step 3: Creating sample task data..."
cat > sample_tasks.json << 'EOF'
[
    {"input": "What is 2 + 2?"},
    {"input": "Solve the puzzle: [1, 2, _, 4]"},
    {"input": "Complete the pattern: A B A B _"},
    {"input": "What comes next: 1, 1, 2, 3, 5, _"},
    {"input": "If all roses are red and this is a rose, what color is it?"}
]
EOF

# Step 4: Run day loop
echo "Step 4: Running day loop..."
python run.py --task arc --file sample_tasks.json --num_steps 20 --workers 16

# Step 5: Run sleep cycle
echo "Step 5: Running sleep cycle..."
python sleep.py

echo "✓ VegaMini v0.1 complete pipeline executed"
echo "Check vega_mini/data/ for lighthouse database and model files"