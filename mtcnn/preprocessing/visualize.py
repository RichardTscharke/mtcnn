import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize(stages, show_box = True, show_landmarks = True, fallback = False):

    n = len(stages)

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    # falls n == 1, axes ist kein Array
    if n == 1:
        axes = [axes]

    for ax, (stage, sample) in zip (axes, stages):

        if fallback:
            image = sample

        else:
            image = sample["image"]
            bx, by, bw, bh = sample["box"]

        ax.imshow(image)
        ax.set_title(stage)
        #ax.axis("off")

        if show_box:
            ax.add_patch(patches.Rectangle((bx, by), bw, bh, fill=False, color="red"))

        if show_landmarks:
            for (lx, ly) in sample["keypoints"].values():
                ax.plot(lx, ly, "ro", markersize=3)

    plt.tight_layout()
    plt.show()

