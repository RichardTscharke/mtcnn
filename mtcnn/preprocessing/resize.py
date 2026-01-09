import cv2

def resize_face(sample, output_size=64):
    image = sample["image"]
    keypoints = sample["keypoints"]

    h, w = image.shape[:2]

    # Zentrum: Augenmittelpunkt
    xL, yL = keypoints["left_eye"]
    xR, yR = keypoints["right_eye"]

    cx = int((xL + xR) / 2)
    cy = int((yL + yR) / 2)

    half = output_size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # Padding falls nötig
    pad_x1 = max(0, -x1)
    pad_y1 = max(0, -y1)
    pad_x2 = max(0, x2 - w)
    pad_y2 = max(0, y2 - h)

    image = cv2.copyMakeBorder(
        image,
        pad_y1, pad_y2, pad_x1, pad_x2,
        borderType=cv2.BORDER_REFLECT
    )

    x1 += pad_x1
    y1 += pad_y1
    x2 += pad_x1
    y2 += pad_y1

    cropped = image[y1:y2, x1:x2]

    scale_x = output_size / cropped.shape[1]
    scale_y = output_size / cropped.shape[0]

    new_keypoints = {}
    for k, (x, y) in keypoints.items():
        new_keypoints[k] =  int((x - x1) * scale_x), int((y - y1) * scale_y)

    sample["image"] = cv2.resize(cropped, (output_size, output_size))
    sample["keypoints"] = new_keypoints

    return sample
