import numpy as np


def non_max_suppression(boxes, threshold):

    if len(boxes) == 0:
        print("No boxes")
        return []

    pick = []

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for box in boxes:
        x1 .append(box["xmin"])
        y1.append(box["ymin"])
        x2.append(box["xmax"])
        y2.append(box["ymax"])

    x1 = np.array(x1, dtype=np.float32)
    x2 = np.array(x2, dtype=np.float32)
    y1 = np.array(y1, dtype=np.float32)
    y2 = np.array(y2, dtype=np.float32)

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while 0 < len(idxs):
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return [boxes[i] for i in pick]
