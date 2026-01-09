import numpy as np
from .padding import apply_padding
from .crop import crop_face
from .align import align_face
from .resize import resize_face
from copy import deepcopy
from .visualize import visualize


def preprocess(image, detector, do_padding = True, do_cropping = True, do_aligning = True, do_resizing = True, debug = False):

    stages = []

    faces = detector.detect_faces(image)

    if not faces:
        raise RuntimeError("No face detected")

    face = max(faces, key=lambda f: float(f["confidence"]))

    x, y, w, h = face["box"]

    keypoints = {}

    for key, (x1, y1) in face["keypoints"].items():

        keypoints[key] = int(x1), int(y1)

    sample = {

        "image": np.array(image),
        "box": (x, y, w, h),
        "keypoints": keypoints
    }

    if debug:
        stages.append(("original", deepcopy(sample)))

    if do_padding:
        sample = apply_padding(sample)
        if debug:
            stages.append(("padded", deepcopy(sample)))

    if do_cropping:
        sample = crop_face(sample)
        if debug:
            stages.append(("cropped", deepcopy(sample)))

    if do_aligning:
        sample = align_face(sample)
        if debug:
            stages.append(("aligned", deepcopy(sample)))

    #if do_resizing:
    #    sample = resize_face(sample)
    #    if debug:
    #        stages.append(("resized", deepcopy(sample)))

    if debug:
        visualize(stages)        

    return sample
