import cv2
from process_image import rcnn_detection, draw_bounding_boxes
import time
import threading
from queue import Queue
import os
import json
import argparse


class CapturedImage:
    def __init__(self, image):
        self.image = image
        self.timestamp = time.localtime()
        self.birds = []
        self.formatted_timestamp = time.strftime('%Y-%m-%dT%H:%M:%S', self.timestamp)


def find_birds_thread(input_queue: Queue, drawing_queue: Queue, annotation_queue: Queue, csv_queue: Queue):
    try:
        while True:
            if input_queue.empty() or drawing_queue.full():
                time.sleep(30)
                continue
            captured: CapturedImage = input_queue.get()

            captured.birds = rcnn_detection(captured.image)

            drawing_queue.put(captured)
            annotation_queue.put(captured)
            csv_queue.put(captured)

    except KeyboardInterrupt:
        pass


def video_capture_thread(video: cv2.VideoCapture, detection_queue: Queue, org_queue: Queue, delay_time: int):
    try:
        while True:
            if not detection_queue.full():
                _, frame = video.read()

                captured = CapturedImage(frame)

                detection_queue.put(captured)
                org_queue.put(captured)

            time.sleep(delay_time)
    except KeyboardInterrupt:
        video.release()
        pass


def draw_bounding_boxes_thread(input_queue: Queue, base_dir: str):
    if not os.path.exists(os.path.join(base_dir, "Boxed")):
        os.makedirs(os.path.join(base_dir, "Boxed"))
    try:
        while True:
            if input_queue.empty():
                time.sleep(30)
                continue
            captured: CapturedImage = input_queue.get()

            image = draw_bounding_boxes(captured.image, captured.birds)

            cv2.imwrite(os.path.join(base_dir, "Boxed", f"{captured.formatted_timestamp}.png"), image)
    except KeyboardInterrupt:
        pass


def save_original_image_thread(input_queue: Queue, base_dir: str):
    if not os.path.exists(os.path.join(base_dir, "Original")):
        os.makedirs(os.path.join(base_dir, "Original"))
    try:
        while True:
            if input_queue.empty():
                time.sleep(30)
                continue
            captured: CapturedImage = input_queue.get()

            cv2.imwrite(os.path.join(base_dir, "Original", f"{captured.formatted_timestamp}.png"), captured.image)
    except KeyboardInterrupt:
        pass


def save_annotations_thread(input_queue: Queue, base_dir: str):
    if not os.path.exists(os.path.join(base_dir, "Annotations")):
        os.makedirs(os.path.join(base_dir, "Annotations"))
    try:
        while True:
            if input_queue.empty():
                time.sleep(30)
                continue
            captured: CapturedImage = input_queue.get()

            with open(os.path.join(base_dir, "Annotations", f"{captured.formatted_timestamp}.json"), "w") as f:
                json.dump(str(captured.birds), f)
    except KeyboardInterrupt:
        pass


def csv_writing_thread(input_queue: Queue, base_dir:str):
    with open("labels.txt", "r") as f:
        labels = f.read().split("\n")

    if not os.path.exists(os.path.join(base_dir, "data.csv")):
        with open(os.path.join(base_dir, "data.csv"), "w") as f:
            f.write("timestamp,")
            for label in labels:
                f.write(f"{label},")
            f.write("\n")

    try:
        while True:
            if input_queue.empty():
                time.sleep(30)
                continue
            captured: CapturedImage = input_queue.get()

            with open(os.path.join(base_dir, "data.csv"), "a") as f:
                f.write(f"{captured.formatted_timestamp},")
                counts = [0] * len(labels)

                for bird in captured.birds:
                    counts[labels.index(bird["label"])] += 1

                for count in counts:
                    f.write(f"{count},")

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":

    paser = argparse.ArgumentParser()

    paser.add_argument(
        "video_capture",
        metavar="capture",
    )

    paser.add_argument(
        "target_dir",
        metavar="target"
    )

    args = paser.parse_args()

    vid = cv2.VideoCapture(args.video_capture)

    detection_queue = Queue()
    drawing_queue = Queue()
    anno_queue = Queue()
    org_queue = Queue()
    csv_queue = Queue()

    capture_thread = threading.Thread(target=video_capture_thread, args=(vid, detection_queue, org_queue, 60))
    detection_thread = threading.Thread(target=find_birds_thread, args=(detection_queue, drawing_queue, anno_queue, csv_queue))
    drawing_thread = threading.Thread(target=draw_bounding_boxes_thread, args=(drawing_queue, args.target_dir))
    org_thread = threading.Thread(target=save_original_image_thread, args=(org_queue, args.target_dir))
    anno_thread = threading.Thread(target=save_annotations_thread, args=(anno_queue, args.target_dir))
    csv_thread = threading.Thread(target=csv_writing_thread, args=(csv_queue, args.target_dir))

    capture_thread.start()
    detection_thread.start()
    drawing_thread.start()
    org_thread.start()
    anno_thread.start()
    csv_thread.start()

    capture_thread.join()
    detection_thread.join()
    drawing_thread.join()
    org_thread.join()
    anno_thread.join()
    csv_thread.join()