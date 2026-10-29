import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"data"
RAW = DATA/"raw"
# kaggle mounts the code read only and the tests use small copies so these folders can be moved
PROMPTS = Path(os.environ.get("LBS_PROMPTS",DATA/"prompts"))
RESULTS = Path(os.environ.get("LBS_RESULTS",ROOT/"results"))
MODELS = Path(os.environ.get("LBS_MODELS",ROOT/"cache"/"models"))
GEN = RESULTS/"gen"
SCORES = RESULTS/"scores"
