from scratch import DetectController
from pathlib import Path
from datetime import datetime
import re
import numpy as np
from PIL import Image


def load_frames_and_timestamps(folder: str):
    pattern = re.compile(r"_(\d{8})_(\d{6})_cellScratch\.jpg$", re.IGNORECASE)
    files = []
    for image_file in Path(folder).iterdir():
        match = pattern.search(image_file.name)
        if match is None:
            continue
        captured_at = datetime.strptime(
            f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S"
        )
        files.append((captured_at, image_file))

    files.sort(key=lambda item: item[0])
    if not files:
        raise ValueError(f"No matching scratch images found in {folder}")

    start_time = files[0][0]
    images = [np.array(Image.open(image_file).convert('RGB')) for _, image_file in files]
    timestamps = np.array(
        [(captured_at - start_time).total_seconds() for captured_at, _ in files],
        dtype=float,
    )
    return images, timestamps

def test_analyse_scratch():
    image_rgb = np.array(Image.open("data/input/s1/A1_11_12_20260228_131443_cellScratch.jpg").convert('RGB'))

    quality, result, invasion = DetectController.analyse_scratch(image_rgb)

    print(quality)
    print(result)

    assert result.area.pixel != 0

def test_analyse_scratch_kinetic():
    exp_image_list, timestamp_list = load_frames_and_timestamps("data/input/s1")

    error_code, result_list, invasion_list = DetectController.analyse_scratch_kinetic(exp_image_list = exp_image_list, timestamp_list = timestamp_list)

    print(result_list[0])

    
    

    