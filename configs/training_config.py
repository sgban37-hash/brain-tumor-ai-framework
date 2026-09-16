from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# DATASET
# ============================================================

MSD_DIR = DATA_DIR / "Task01_BrainTumour"

TRAIN_MANIFEST = OUTPUT_DIR / "msd_split_manifest.csv"


# ============================================================
# MRI CONFIGURATION
# ============================================================

# Channel order:
# 0 = FLAIR
# 1 = T1
# 2 = T1c
# 3 = T2

IN_CHANNELS = 4

# Target channels:
# 0 = Whole Tumor (WT)
# 1 = Tumor Core (TC)
# 2 = Enhancing Tumor (ET)

OUT_CHANNELS = 3


# ============================================================
# PATCH CONFIGURATION
# ============================================================

PATCH_SIZE = (96, 96, 96)

BATCH_SIZE = 1


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "SegResNet"

INIT_FILTERS = 16

BLOCKS_DOWN = (1, 2, 2, 4)

BLOCKS_UP = (1, 1, 1)

DROPOUT = 0.2


# ============================================================
# TRAINING
# ============================================================

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

MAX_EPOCHS = 10


# ============================================================
# PERFORMANCE
# ============================================================

USE_AMP = True

NUM_WORKERS = 2

PIN_MEMORY = True

PERSISTENT_WORKERS = True


# ============================================================
# VALIDATION
# ============================================================

VALIDATION_FREQUENCY = 5

SLIDING_WINDOW_OVERLAP = 0.25

SLIDING_WINDOW_BATCH_SIZE = 1


# ============================================================
# CHECKPOINTING
# ============================================================

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

BEST_MODEL_NAME = "best_model.pth"

LAST_MODEL_NAME = "last_model.pth"


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42