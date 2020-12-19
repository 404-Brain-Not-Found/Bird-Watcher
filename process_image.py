try:
    import tensorflow.lite as tflite
except ModuleNotFoundError:
    import tflite_runtime.interpreter as tflite

import cv2
import numpy as np
import itertools

interpreter = tflite.Interpreter("lite-bird-classifier.tflite")
interpreter.allocate_tensors()

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

with open("labels.txt", "r") as f:
    labels = f.read().split("\n")


def rcnn_detection(image, min_conf=0.9, overlap_threshold=0.3):
    assert min_conf < 1.0

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ss_results = ss.process()

    found = []
    for e, result in enumerate(ss_results):
        if e < 2000:
            x, y, w, h = result
            timage = image[y:y + h, x:x + w]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA) / 255
            resized = np.float32(resized)
            img = np.expand_dims(resized, axis=0)

            interpreter.set_tensor(input_details[0]["index"], img)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            output = np.squeeze(output_data)
            max_index = int(np.argmax(output))

            if max_index < len(labels) and min_conf <= output[max_index]:
                found.append({
                    "label": labels[max_index],
                    "confidence": float(output[max_index]),
                    "xmin": int(x),
                    "ymin": int(y),
                    "xmax": int(x + w),
                    "ymax": int(y + h)
                })
        else:
            break

    if len(found) == 0:
        return []

    filter_finds = []

    for _, boxes in itertools.groupby(found, lambda x: x["label"]):
        filter_finds.extend(non_max_suppression(list(boxes), overlap_threshold))

    return filter_finds


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


def draw_bounding_boxes(image, boxes):
    for box in boxes:
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]

        box_color = (255, 0, 255)
        label_color = (0, 0, 0)

        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, 2)

        text = f"{box['label']}: {box['confidence'] * 100: .2f}%"

        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        image = cv2.rectangle(image,
                              (xmin, ymin),
                              (xmin + label_size[0][0], ymin - label_size[0][1]),
                              box_color,
                              cv2.FILLED)

        image = cv2.putText(
            image,
            text,
            (xmin, ymin),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            label_color,
            2
        )

    return image


if __name__ == "__main__":
    image = cv2.imread("/home/thomasquirk/PycharmProjects/Bird Classifier/data/valid/AFRICAN FIREFINCH/1.jpg")
    boxes = rcnn_detection(image, min_conf=.97)
    image = draw_bounding_boxes(image, boxes)

    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
