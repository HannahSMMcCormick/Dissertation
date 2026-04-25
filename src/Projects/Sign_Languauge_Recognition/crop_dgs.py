import os
from typing import Optional
import cv2

def crop_dgs_video(input_path: str, output_dir: str) -> Optional[str]:
    
    if not os.path.isfile(input_path):
        print(f"[crop_dgs] Missing file: {input_path}")
        return None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[crop_dgs] Cannot open: {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   
    crop_w = int(width * 0.60)     
    x_start = int(width * 0.20)    
    x_end = x_start + crop_w

 
    x_start = max(0, x_start)
    x_end = min(width, x_end)
    crop_w = x_end - x_start

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_cropped.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (crop_w, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        crop = frame[:, x_start:x_end]
        writer.write(crop)

    cap.release()
    writer.release()

    print(f"[crop_dgs] Saved cropped DGS video: {out_path}")
    return out_path
