"""
Central configuration for VegaMini project.
"""
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'vega_mini', 'data')
LIGHTHOUSE_DB_PATH = os.path.join(DATA_DIR, 'lighthouses.db')
FAISS_INDEX_PATH = os.path.join(DATA_DIR, 'lighthouse_index.faiss')
QUALITY_MODEL_PATH = os.path.join(DATA_DIR, 'quality_model.pt')

# Model and flow parameters
MODEL_DIM = 1024
SWARM_SIZE_MIN = 4
SWARM_SIZE_MAX = 32
QUALITY_WRITE_THRESHOLD = 0.8
QUALITY_REINFORCE_THRESHOLD = 0.65
STV_MARGIN_MIN = 0.5
FLOW_STEPS = 6

# Lighthouse parameters
LIGHTHOUSE_BRIGHTNESS_INIT = 1.0
LIGHTHOUSE_BRIGHTNESS_MIN = 0.05
LIGHTHOUSE_DECAY_LAMBDA = 0.95
LIGHTHOUSE_MERGE_DELTA = 0.1

# Quality model
QUALITY_BUFFER_SIZE = 10000
QUALITY_HIDDEN_DIM = 256
QUALITY_EMBED_DIM = 64
QUALITY_VOCAB_SIZE = 10000

# Sleep cycle
SLEEP_REPLAY_K = 2000
SLEEP_DREAMS = 500
SLEEP_NIGHTMARES = 200

# Misc
SEED = 42
