import torch

# Name or path of the HF model to load
MODEL_NAME = "timaeus/tetrahedron-3m-og"

# Data parameters
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
