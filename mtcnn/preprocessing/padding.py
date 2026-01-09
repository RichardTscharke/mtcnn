def apply_padding(sample, pad_ratio = 0.4):

    image = sample["image"]

    x, y, w, h = sample["box"]

    image_h, image_w = image.shape[:2]

    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)

    x2 = min(image_w, x + w + pad_w)
    y2 = min(image_h, y + h + pad_h)

    sample["box"] = (x1, y1, x2 - x1, y2 - y1)

    return sample

