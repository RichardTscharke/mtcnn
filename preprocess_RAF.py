from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from mtcnn import MTCNN
from mtcnn.preprocessing import preprocess
from mtcnn.preprocessing.fallback import fallback

import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices("GPU"))


INPUT_DIR = Path("/home/t/tscharke/work/data/RAF/original")
OUTPUT_DIR = Path("/home/t/tscharke/work/data/RAF/aligned")
LOG_FILE = OUTPUT_DIR / "preprocess.log"

#INPUT_DIR = Path("/Users/richardachtnull/Desktop/RAF/original")
#OUTPUT_DIR = Path("/Users/richardachtnull/Desktop/RAF/self_aligned_v2")
#LOG_FILE = OUTPUT_DIR / "preprocess.log"

detector = MTCNN()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []

for img_path in sorted(INPUT_DIR.rglob("*")):

    if img_path.suffix.lower() not in [".jpg", ".png"]:
        continue

    out_path = OUTPUT_DIR / img_path.name


    image_array = np.array(Image.open(img_path))

    sample = preprocess(image_array, detector)

    if sample is not None:

        img = sample["image"]  # RGB, 64x64

        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        continue

    crop_mtcnn, crop64 = fallback(image_array)

    sample2 = preprocess(crop_mtcnn, detector)

    if sample2 is not None:

        img = sample2["image"]  # RGB, 64x64

        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        log_lines.append(f"{img_path.name}: fallback + second pass aligned")
        continue 

    cv2.imwrite(
    str(out_path),
    cv2.cvtColor(crop64, cv2.COLOR_RGB2BGR)
    )
    log_lines.append(f"{img_path.name}: fallback ROI only")

# Log schreiben
with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
