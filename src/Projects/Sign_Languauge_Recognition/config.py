import os

# Path to this file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the Data folder (two levels up)
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "..", "Data"))


# -----------------------------
# EXTERNAL VIDEO FOLDERS
# -----------------------------
BSL_EXTERNAL_PATH = os.path.join(DATA_ROOT, "External", "BSL")
DGS_EXTERNAL_PATH = os.path.join(DATA_ROOT, "External", "DGS")
LSF_EXTERNAL_PATH = os.path.join(DATA_ROOT, "External", "LSF")
GSL_EXTERNAL_PATH = os.path.join(DATA_ROOT, "External", "GSL")

# -----------------------------
# INTERIM JSON OUTPUT FOLDERS
# -----------------------------
BSL_INTERIM_PATH = os.path.join(DATA_ROOT, "Interim", "BSL")
DGS_INTERIM_PATH = os.path.join(DATA_ROOT, "Interim", "DGS")
LSF_INTERIM_PATH = os.path.join(DATA_ROOT, "Interim", "LSF")
GSL_INTERIM_PATH = os.path.join(DATA_ROOT, "Interim", "GSL")

# -----------------------------
# INTERIM ANNOTATED VIDEO FOLDERS
# -----------------------------
INTERIM_BSL_ANNOTATED_PATH = os.path.join(DATA_ROOT, "Interim", "BSL_ANNOTATED")
INTERIM_DGS_ANNOTATED_PATH = os.path.join(DATA_ROOT, "Interim", "DGS_ANNOTATED")
INTERIM_LSF_ANNOTATED_PATH = os.path.join(DATA_ROOT, "Interim", "LSF_ANNOTATED")
INTERIM_GSL_ANNOTATED_PATH = os.path.join(DATA_ROOT, "Interim", "GSL_ANNOTATED")

# -----------------------------
# DGS CROPPED VIDEO FOLDER
# -----------------------------
INTERIM_DGS_CROPPED_PATH = os.path.join(DATA_ROOT, "Interim", "DGS_CROPPED")

# -----------------------------
# MODEL ASSET PATHS
# -----------------------------
hand_landmarker_task_PATH = os.path.join(DATA_ROOT, "Mediapipe", "hand_landmarker.task")
Face_Landmarker_task_PATH = os.path.join(DATA_ROOT, "Mediapipe", "face_landmarker_v2_with_blendshapes.task")
pose_landmarker_task_PATH = os.path.join(DATA_ROOT, "Mediapipe", "pose_landmarker.task")

# -----------------------------
# ENSURE DIRECTORIES EXIST
# -----------------------------
ALL_DIRS = [
    BSL_EXTERNAL_PATH, DGS_EXTERNAL_PATH, LSF_EXTERNAL_PATH, GSL_EXTERNAL_PATH,
    BSL_INTERIM_PATH, DGS_INTERIM_PATH, LSF_INTERIM_PATH, GSL_INTERIM_PATH,
    INTERIM_BSL_ANNOTATED_PATH, INTERIM_DGS_ANNOTATED_PATH,
    INTERIM_LSF_ANNOTATED_PATH, INTERIM_GSL_ANNOTATED_PATH,
    INTERIM_DGS_CROPPED_PATH
]

for d in ALL_DIRS:
    os.makedirs(d, exist_ok=True)
