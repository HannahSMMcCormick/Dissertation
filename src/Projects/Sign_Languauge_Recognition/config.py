from pathlib import Path
import os


Project_ROOT = Path(__file__).resolve().parents[3]

#DATA FOLDERS
DATA_PATH = Project_ROOT / "Data"

MEDIAPIPE_PATH = DATA_PATH / "Mediapipe"
hand_landmarker_task_PATH = MEDIAPIPE_PATH / "hand_landmarker.task"

EXTERNAL_PATH = DATA_PATH / "External"

INTERIM_PATH = DATA_PATH / "Interim"
