"""Project-wide paths, seed, and hyperparameters.

Every value that affects results lives here so runs are reproducible.
The reason behind each non-obvious value is recorded in docs/decisions.md.
"""
from pathlib import Path

import torch

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
# The Kaggle zip extracts to car_data/car_data/{train,test}
IMAGE_ROOT = ARCHIVE / "car_data" / "car_data"
ANNO_TRAIN = ARCHIVE / "anno_train.csv"
ANNO_TEST = ARCHIVE / "anno_test.csv"
NAMES = ARCHIVE / "names.csv"

SPLITS_DIR = ROOT / "splits"
OUTPUTS_DIR = ROOT / "outputs"

# ---------------------------------------------------------------- split
SEED = 42
VAL_FRACTION = 0.15  # stratified per class, taken from the official train set
NUM_CLASSES = 196

# Makes whose name is two words; every other make is the first word.
MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")

# ---------------------------------------------------------------- images
IMG_SIZE = 224          # ImageNet pretraining resolution
BBOX_MARGIN = 0.10      # crop margin as a fraction of box width/height, each side
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Train augmentation
RRC_SCALE = (0.5, 1.0)  # RandomResizedCrop area range
RRC_RATIO = (3 / 4, 4 / 3)
COLOR_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)

# ---------------------------------------------------------------- training
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 6            # early stopping on val top-1
WARMUP_EPOCHS = 1
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 0.05     # AdamW decay is multiplied by the LR, so it must be much larger than for SGD
NUM_WORKERS = 8


def get_device() -> torch.device:
    """cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_of(class_name: str) -> str:
    for make in MULTI_WORD_MAKES:
        if class_name.startswith(make + " "):
            return make
    return class_name.split(" ")[0]
