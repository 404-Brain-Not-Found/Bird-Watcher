try:
    import tensorflow.lite as tflite
except ModuleNotFoundError:
    import tflite_runtime.interpreter as tflite

import cv2
import numpy as np
import itertools
from yolo_utils import decode_yolo_output


classifier_inter = None
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

yolo_inter = None

with open("labels.txt", "r") as f:
    labels = f.read().split("\n")


def rcnn_detection(image, min_conf=0.9, overlap_threshold=0.3):
    assert min_conf < 1.0
    global classifier_inter

    if classifier_inter is None:
        classifier_inter = tflite.Interpreter("lite-bird-classifier.tflite")
        classifier_inter.allocate_tensors()

    input_details = classifier_inter.get_input_details()
    output_details = classifier_inter.get_output_details()

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

            classifier_inter.set_tensor(input_details[0]["index"], img)
            classifier_inter.invoke()

            output_data = classifier_inter.get_tensor(output_details[0]['index'])
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


def yolo_detection(image, min_conf=0.9, overlap_threshold=0.3):
    assert min_conf < 1.0

    global yolo_inter

    if yolo_inter is None:
        yolo_inter = tflite.Interpreter("lite-bird-classifier.tflite")
        yolo_inter.allocate_tensors()

    input_details = yolo_inter.get_input_details()
    output_details = yolo_inter.get_output_details()

    image = np.float32(cv2.resize(image, (448, 448))) / 255

    image = np.expand_dims(image, axis=0)

    yolo_inter.set_tensor(input_details[0]["index"], image)
    yolo_inter.invoke()

    output_data = yolo_inter.get_tensor(output_details[0]['index'])
    output = np.squeeze(output_data)

    return decode_yolo_output(image, output, labels)


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
