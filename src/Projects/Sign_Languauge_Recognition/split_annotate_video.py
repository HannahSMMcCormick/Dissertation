import json
import torch
import numpy as np
import os

from extract_video import extract_video

# These imports will work once you create ml_core.py
from ml_core import (
    Seq2Seq,
    convert_frames_to_sequence,
    beam_search,
    id_to_token,
    sos_id,
    eos_id,
    num_features,
    vocab_size,
    device,
)


# ---------------------------------------------------------
# HYBRID SEGMENTATION
# ---------------------------------------------------------

def frame_velocity(seq):
    """Compute per-frame motion magnitude."""
    diffs = np.linalg.norm(seq[1:] - seq[:-1], axis=1)
    return diffs


def handshape_change(seq, hand_indices):
    """Compute motion only for hand landmarks."""
    hand = seq[:, hand_indices]
    diffs = np.linalg.norm(hand[1:] - hand[:-1], axis=1)
    return diffs


def hybrid_segment(seq, vel_thresh=0.015, hand_thresh=0.02, min_gap=5):
    """
    Combine velocity + handshape change to detect sign boundaries.
    """
    vel = frame_velocity(seq)
    hand = handshape_change(seq, hand_indices=range(0, 21 * 3))  # left hand only

    boundaries = [0]

    for i in range(1, len(seq) - 1):
        if vel[i] < vel_thresh and hand[i] < hand_thresh:
            if i - boundaries[-1] > min_gap:
                boundaries.append(i)

    boundaries.append(len(seq))
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

def load_trained_model():
    """
    Load your trained seq2seq model for inference.
    """
    model = Seq2Seq(
        input_dim=num_features,
        vocab_size=vocab_size,
        hidden_dim=256
    ).to(device)

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------
# FULL PIPELINE: VIDEO → JSON → SEQUENCE → SEGMENTS → HNS
# ---------------------------------------------------------

def annotate_video(video_path):
    """
    Full annotation pipeline:
    1. Extract landmarks to JSON
    2. Load JSON
    3. Convert frames → landmark sequence
    4. Segment sequence
    5. Run model on each segment
    6. Save results into Data/Processed/<video_name>/
    """

    # -----------------------------------------------------
    # Create output folder: Data/Processed/<video_name>/
    # -----------------------------------------------------
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join("Data", "Processed", video_name)
    os.makedirs(out_dir, exist_ok=True)

    landmarks_path = os.path.join(out_dir, "landmarks.json")
    annotations_path = os.path.join(out_dir, "annotations.json")

    # -----------------------------------------------------
    # 1. Extract landmarks → save to landmarks.json
    # -----------------------------------------------------
    extract_video(video_path, landmarks_path, language="unknown")

    # -----------------------------------------------------
    # 2. Load landmark JSON
    # -----------------------------------------------------
    with open(landmarks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]

    # -----------------------------------------------------
    # 3. Convert frames → sequence
    # -----------------------------------------------------
    seq = convert_frames_to_sequence(frames)

    # -----------------------------------------------------
    # 4. Segment
    # -----------------------------------------------------
    segments = hybrid_segment(seq)

    # -----------------------------------------------------
    # 5. Load model
    # -----------------------------------------------------
    model = load_trained_model()

    annotations = []

    # -----------------------------------------------------
    # 6. Annotate each segment
    # -----------------------------------------------------
    for start, end in segments:
        seg = seq[start:end]
        seg_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)

        pred_ids = beam_search(model, seg_tensor, sos_id, eos_id)

        if eos_id in pred_ids:
            pred_ids = pred_ids[:pred_ids.index(eos_id)]

        hamnosys = "".join(id_to_token[i] for i in pred_ids)

        annotations.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "hamnosys": hamnosys
        })

    # -----------------------------------------------------
    # 7. Save annotations.json
    # -----------------------------------------------------
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    return annotations_path


# ---------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    import sys

    # If user only gives filename, assume Data/Raw
    input_arg = sys.argv[1]
    if not os.path.isfile(input_arg):
        video_path = os.path.join("Data", "Raw", input_arg)
    else:
        video_path = input_arg

    out_file = annotate_video(video_path)
    print(f"Saved annotations to: {out_file}")
