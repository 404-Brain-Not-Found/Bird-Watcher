import argparse
from process_image import rcnn_detection
from tqdm import tqdm
import os
import cv2
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from queue import Queue
from multiprocessing import pool


def process_folder(args):
    folders, src_dir, target_dir = args
    generator = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        brightness_range=(0.2, 1.8)
    )

    for folder in folders:
        images = []
        labels = []
        for filename in os.listdir(os.path.join(src_dir, folder)):
            image = cv2.imread(os.path.join(src_dir, folder, filename))
            images.append(image)
            labels.append(os.path.join(src_dir, folder, filename))

        count = 0
        for x, y in generator.flow(np.array(images), np.array(labels)):
            for (index, (image, label)) in enumerate(zip(x, y)):
                filename = label.split("/")[-1]
                label = label.split("/")[-2]

                boxes = rcnn_detection(image)

                if len(boxes) == 1 and boxes[0]["label"] == label.title():
                    cv2.imwrite(os.path.join(target_dir, "images", f"{folder}_{count}.jpg"), image)
                    with open(os.path.join(target_dir, "annotations", f"{folder}-{count}"), 'w') as f:
                        json.dump(boxes, f)
                    count += 1

            if 100 <= count:
                break


if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument("src")
    parse.add_argument("target")

    args = parse.parse_args()

    src_dir = args.src
    target_dir = args.target

    if not os.path.exists(os.path.join(target_dir, "images")):
        os.makedirs(os.path.join(target_dir, "images"))
    if not os.path.exists(os.path.join(target_dir, "annotations")):
        os.makedirs(os.path.join(target_dir, "annotations"))

    folders = os.listdir(src_dir)
    n = len(folders) // 20
    process_input = [(folders[i: n + i], src_dir, target_dir) for i in range(0, len(folders), n)]

    with pool.Pool(20) as p:
        p.map(process_folder, process_input)


