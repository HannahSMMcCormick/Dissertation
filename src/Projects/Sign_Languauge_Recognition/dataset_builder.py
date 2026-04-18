import os
import json
import time
from typing import Dict, List

import requests

from config import (
    BSL_EXTERNAL_PATH,
    GSL_EXTERNAL_PATH,
    DGS_EXTERNAL_PATH,
    LSF_EXTERNAL_PATH,
    BSL_INTERIM_PATH,
    GSL_INTERIM_PATH,
    DGS_INTERIM_PATH,
    LSF_INTERIM_PATH,
)
from crop_dgs import crop_dgs_video
from extract_video import extract_video
from concept_maps import (
    load_all_concept_maps,
    extract_hamnosys_for_language,
    norm_sign,
)


LANGUAGE_PATHS: Dict[str, Dict[str, str]] = {
    "BSL": {"external": BSL_EXTERNAL_PATH, "interim": BSL_INTERIM_PATH, "ham_lang": "BSL"},
    "GSL": {"external": GSL_EXTERNAL_PATH, "interim": GSL_INTERIM_PATH, "ham_lang": "GSL"},
    "DGS": {"external": DGS_EXTERNAL_PATH, "interim": DGS_INTERIM_PATH, "ham_lang": "DGS"},
    "LSF": {"external": LSF_EXTERNAL_PATH, "interim": LSF_INTERIM_PATH, "ham_lang": "LSF"},
}


def process_language(
    lang: str,
    paths: Dict[str, str],
    concept_map: Dict[str, str],
    session: requests.Session,
) -> List[dict]:
    external = paths["external"]
    interim = paths["interim"]
    ham_lang = paths["ham_lang"]

    os.makedirs(interim, exist_ok=True)
    annotated_dir = os.path.join(interim, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    videos: List[dict] = []

    video_files = sorted([f for f in os.listdir(external) if f.lower().endswith(".mp4")])

    for idx, filename in enumerate(video_files):
        input_path = os.path.join(external, filename)
        sign_raw = filename[:-4]
        sign_key = norm_sign(sign_raw)

        concept_url = concept_map.get(sign_key)

        if not concept_url:
            stripped = norm_sign(sign_raw.lower())
            concept_url = concept_map.get(stripped)

        hamnosys = ""
        if concept_url:
            try:
                ham = extract_hamnosys_for_language(concept_url, session, ham_lang)
                if ham:
                    hamnosys = ham
            except Exception:
                pass
            time.sleep(0.2)

        # DGS cropping
        video_for_extraction = input_path
        if lang == "DGS":
            cropped = crop_dgs_video(input_path, interim)
            if cropped is None:
                print(f"[dataset_builder] Skipping {input_path} (crop failed)")
                continue
            video_for_extraction = cropped

        json_path = os.path.join(interim, f"{sign_raw}.json")
        annotated_path = os.path.join(annotated_dir, f"{sign_raw}_annotated.mp4")

        ok = extract_video(
            input_video_path=video_for_extraction,
            json_output_path=json_path,
            language=lang,
            annotated_output_path=annotated_path,
        )
        if not ok:
            print(f"[dataset_builder] Skipping {input_path} (extraction failed)")
            continue

        videos.append(
            {
                "id": f"{lang}_{idx+1}",
                "language": lang,
                "Sign": sign_raw,
                "filepath": input_path,
                "HamNoSys": hamnosys,
                "concept_url": concept_url or "",
                "HandLandmark": json_path,
                "annotated_video": annotated_path,
            }
        )

    out_file = f"videos_{lang.lower()}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"videos": videos}, f, ensure_ascii=False, indent=2)

    print(f"[dataset_builder] Saved {out_file} ({len(videos)} videos)")
    return videos


def build_full_dataset():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (dataset builder; research use)"})

    concept_maps = load_all_concept_maps(session)

    all_videos: List[dict] = []

    for lang, paths in LANGUAGE_PATHS.items():
        print(f"\n[dataset_builder] Processing language: {lang}")
        cmap = concept_maps.get(lang, {})
        vids = process_language(lang, paths, cmap, session)
        all_videos.extend(vids)

    with open("videos_all_languages.json", "w", encoding="utf-8") as f:
        json.dump({"videos": all_videos}, f, ensure_ascii=False, indent=2)

    print("\n[dataset_builder] Saved videos_all_languages.json")
    print(f"[dataset_builder] Total videos: {len(all_videos)}")

print("DEBUG PATHS:")
print("BSL_EXTERNAL_PATH =", BSL_EXTERNAL_PATH)
print("DGS_EXTERNAL_PATH =", DGS_EXTERNAL_PATH)
print("LSF_EXTERNAL_PATH =", LSF_EXTERNAL_PATH)
print("GSL_EXTERNAL_PATH =", GSL_EXTERNAL_PATH)
