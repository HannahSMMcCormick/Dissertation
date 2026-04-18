import os
import json
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles


from config import (
    hand_landmarker_task_PATH,
    Face_Landmarker_task_PATH,
    pose_landmarker_task_PATH,
)

HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

POSE_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


def HandObject():
    base_options = python.BaseOptions(model_asset_path=str(hand_landmarker_task_PATH))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)


def FaceObject():
    base_options = python.BaseOptions(model_asset_path=str(Face_Landmarker_task_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def poseObject():
    base_options = python.BaseOptions(model_asset_path=str(pose_landmarker_task_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
    )
    return vision.PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, hand_result, face_result, pose_result):
    annotated = np.copy(rgb_image)

    # HANDS
    if hand_result and getattr(hand_result, "hand_landmarks", None):
        for hand_landmarks in hand_result.hand_landmarks:
            drawing_utils.draw_landmarks(
                annotated,
                hand_landmarks,
                connections=None,
                landmark_drawing_spec=None,
                connection_drawing_spec=None,
            )

    # FACE
    if face_result and getattr(face_result, "face_landmarks", None):
        for face_landmarks in face_result.face_landmarks:
            drawing_utils.draw_landmarks(
                annotated,
                face_landmarks,
                connections=None,
                landmark_drawing_spec=None,
                connection_drawing_spec=None,
            )

    # POSE
    if pose_result and getattr(pose_result, "pose_landmarks", None):
        for pose_landmarks in pose_result.pose_landmarks:
            drawing_utils.draw_landmarks(
                annotated,
                pose_landmarks,
                connections=None,
                landmark_drawing_spec=None,
                connection_drawing_spec=None,
            )

    return annotated


def Extract_points(hand_detection_result, face_detection_result, pose_detection_result, frame_idx, fps):
    frame_record = {
        "frame_index": frame_idx,
        "time_sec": frame_idx / fps,
        "hands": [],
        "face": [],
        "pose": [],
    }

    # Hands
    hand_landmarks_list = getattr(hand_detection_result, "hand_landmarks", []) or []
    handedness_list = getattr(hand_detection_result, "handedness", []) or []

    for h_i, hand_landmarks in enumerate(hand_landmarks_list):
        label = None
        score = None
        if h_i < len(handedness_list) and handedness_list[h_i]:
            category = handedness_list[h_i][0]
            label = category.category_name
            score = float(category.score)

        points = []
        for lm_i, lm in enumerate(hand_landmarks):
            points.append(
                {
                    "id": lm_i,
                    "name": HAND_LANDMARK_NAMES[lm_i] if lm_i < len(HAND_LANDMARK_NAMES) else str(lm_i),
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                }
            )

        frame_record["hands"].append(
            {
                "hand_index": h_i,
                "handedness": label,
                "handedness_score": score,
                "landmarks": points,
            }
        )

    # Face
    face_landmarks_list = getattr(face_detection_result, "face_landmarks", []) or []
    for face in face_landmarks_list:
        face_points = []
        for f_i, lm in enumerate(face):
            face_points.append(
                {
                    "id": f_i,
                    "name": f"FACE_{f_i}",
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                }
            )
        frame_record["face"].append({"landmarks": face_points})

    # Pose
    pose_landmarks_list = getattr(pose_detection_result, "pose_landmarks", []) or []
    for pose in pose_landmarks_list:
        pose_points = []
        for p_i, lm in enumerate(pose):
            name = POSE_LANDMARK_NAMES[p_i] if p_i < len(POSE_LANDMARK_NAMES) else str(p_i)
            pose_points.append(
                {
                    "id": p_i,
                    "name": name,
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(getattr(lm, "visibility", 0.0)),
                }
            )
        frame_record["pose"].append({"landmarks": pose_points})

    return frame_record


def extract_video(
    input_video_path: str,
    json_output_path: str,
    language: str,
    annotated_output_path: Optional[str] = None,
) -> bool:
    if not os.path.isfile(input_video_path):
        print(f"[extract_video] Missing file: {input_video_path}")
        return False

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[extract_video] Cannot open: {input_video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if annotated_output_path is not None:
        os.makedirs(os.path.dirname(annotated_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_output_path, fourcc, fps, (width, height))

    json_output = {
        "video_path": input_video_path,
        "fps": fps,
        "language": language,
        "frames": [],
    }

    hand_detector = HandObject()
    face_detector = FaceObject()
    pose_detector = poseObject()

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hand_result = hand_detector.detect(mp_image)
        face_result = face_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)

        if writer is not None:
            annotated_rgb = draw_landmarks_on_image(frame_rgb, hand_result, face_result, pose_result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)

        json_output["frames"].append(
            Extract_points(hand_result, face_result, pose_result, frame_idx, fps)
        )
        frame_idx += 1

    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    print(f"[extract_video] Saved JSON: {json_output_path}")

    cap.release()
    if writer is not None:
        writer.release()

    return True
