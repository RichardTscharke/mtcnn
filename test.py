from mtcnn import MTCNN
from mtcnn.preprocessing import preprocess
from mtcnn.utils.images import load_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


detector = MTCNN(device="CPU:0")

#image = Image.open("tests/images/ivan.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/Bild.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/richard.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/richard3.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/mama.jpg")
image = Image.open("/Users/richardachtnull/Desktop/Bildschirmfoto 2026-01-09 um 17.22.32.png")

image_array = np.array(image)

sample = preprocess(image_array, detector, debug = True)
