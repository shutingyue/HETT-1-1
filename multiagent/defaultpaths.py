from pathlib import Path


PROJECT_ROOT = Path("..")

WEIGHTS_DIR = PROJECT_ROOT/"weights"
GOAL_PREDICTOR_CHECKPOINT_DIR = PROJECT_ROOT/"checkpoints/multi"

CITYREFER_DATA_DIR = PROJECT_ROOT/"data/cityrefer"
OBJECTS_PATH = CITYREFER_DATA_DIR/"objects.json"
PROCESSED_DECRIPTIONS_PATH = CITYREFER_DATA_DIR/"processed_descriptions.json"
MTURK_TRAJECTORY_DIR = PROJECT_ROOT/"data/processed_citynav"

ORTHO_IMAGE_DIR = PROJECT_ROOT/"data/rgbd"
SUBBLOCKS_DIR = PROJECT_ROOT/"data/subblocks"
