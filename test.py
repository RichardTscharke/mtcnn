from mtcnn import MTCNN
from mtcnn.preprocessing import preprocess
from mtcnn.preprocessing.fallback import fallback
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
image = Image.open("/Users/richardachtnull/Desktop/RAF/self_aligned/test_0010.jpg")

image_array = np.array(image)

sample = preprocess(image_array, detector, debug = True)
#fallback(image_array, detector = detector, debug = False)
