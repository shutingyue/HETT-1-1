from pathlib import Path

# 你的项目根目录
PROJECT_ROOT = Path("/mnt/HDD/data/YST/HETT/HETT")
# 你的数据集根目录
DATA_BASE = PROJECT_ROOT / "datasets"

# 1. 权重与 Checkpoint 路径
WEIGHTS_DIR = DATA_BASE / "darknet"  # 对应截图中的 darknet 目录
GOAL_PREDICTOR_CHECKPOINT_DIR = DATA_BASE / "checkpoint" # 对应截图中的 checkpoint 目录

# 2. CityRefer 标注数据路径
# 对应截图中的 refined_citynav/cityrefer
CITYREFER_DATA_DIR = DATA_BASE / "refined_citynav" / "cityrefer"
OBJECTS_PATH = CITYREFER_DATA_DIR / "objects.json"
PROCESSED_DECRIPTIONS_PATH = CITYREFER_DATA_DIR / "processed_descriptions.json"

# 3. 轨迹与图像数据路径
# 对应截图中的 refined_citynav/processed_citynav
MTURK_TRAJECTORY_DIR = DATA_BASE / "refined_citynav" / "processed_citynav"
# 根据截图，refined_citynav 下没看到 rgbd 和 subblocks，
# 如果它们在别处，请修改下方路径；如果还没解压，可以先保持这样
ORTHO_IMAGE_DIR = DATA_BASE / "refined_citynav" / "rgbd"
SUBBLOCKS_DIR = DATA_BASE / "refined_citynav" / "subblocks"