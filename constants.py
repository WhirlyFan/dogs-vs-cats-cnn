import os
from pathlib import Path

BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()
NUM_EPOCHS = 5
MODEL_PATH = Path("models")
MODEL_NAME = "model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
