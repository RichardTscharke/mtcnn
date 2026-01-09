from pathlib import Path
import cv2
from mtcnn import MTCNN
from mtcnn.preprocessing import preprocess
import tensorflow as tf

INPUT_DIR = Path("/Users/richardachtnull/Desktop/RAF/original")
OUTPUT_DIR = Path("data/RAF_raw/Image/aligned")
LOG_FILE = OUTPUT_DIR / "preprocess.log"


print(tf.config.list_physical_devices('GPU'))
