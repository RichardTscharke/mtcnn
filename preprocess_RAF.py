from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from mtcnn import MTCNN
from mtcnn.preprocessing import preprocess
import tensorflow as tf

INPUT_DIR = Path("/Users/richardachtnull/Desktop/RAF/original")
OUTPUT_DIR = Path("/Users/richardachtnull/Desktop/data/RAF_raw/Image/aligned")
LOG_FILE = OUTPUT_DIR / "preprocess.log"

detector = MTCNN()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []

with open(LOG_FILE, "w") as log_file:
    for img_path in sorted(INPUT_DIR.rglob("*")):
        if img_path.suffix.lower() not in [".jpg", ".png"]:
            continue

        out_path = OUTPUT_DIR / img_path.name

        try:
            image_array = np.array(Image.open(img_path))

            sample = preprocess(image_array, detector)

            img = sample["image"]  # RGB, 64x64

            # speichern
            cv2.imwrite(
                str(out_path),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

        except Exception as e:
            # Fallback: Original kopieren
            orig = cv2.imread(str(img_path))
            cv2.imwrite(str(out_path), orig)
            log_lines.append(f"{img_path.name}: {e}")

        break    

# Log schreiben
with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
