from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from yolo_utils import build_yolo_loss
import os
import random
import numpy as np
import cv2
import json
import tensorflow as tf


def build_model(n_classes, nb_boxes=2, grid_w=7, grid_h=7, cell_w=64, cell_h=64):

    base_model = MobileNetV2(input_shape=(grid_h * cell_h, grid_w * cell_w, 3), weights="imagenet", include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-1].output
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = Dense(grid_w * grid_h * (nb_boxes * 5 + n_classes), activation='sigmoid')(x)
    x = Reshape(target_shape=(grid_h, grid_w, n_classes + (nb_boxes * 5)))(x)

    model = Model(base_model.input, x)

    model.summary()

    model.compile(Adam(), loss=build_yolo_loss(n_classes, nb_boxes, grid_w, grid_h))

    return model


class ImageSequence(Sequence):

    def __init__(self,
                 image_dir,
                 annotation_dir,
                 labels,
                 batch_size=32,
                 image_size=(448, 448),
                 grid_w=7,
                 grid_h=7,
                 n_boxes=2):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.filenames = [filename.split(".")[0] for filename in os.listdir(image_dir) if int(filename.split(".")[0].split("_")[-1]) < 100]
        self.batch_size = batch_size
        self.image_size = image_size

        self.n_classes = len(labels)
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.n_boxes = n_boxes
        self.labels = labels

        random.shuffle(self.filenames)

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def __getitem__(self, item):
        files_batch = self.filenames[item * self.batch_size: (item + 1) * self.batch_size]

        images = []
        labels = []

        for file in files_batch:
            image, label = self.load_image(file)
            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)

    def load_image(self, filename):
        image_file = os.path.join(self.image_dir, filename + ".jpg")
        annotation_file = os.path.join(self.annotation_dir, filename.replace("_", "-"))

        image = cv2.imread(image_file) / 255
        image_h, image_w = image.shape[:2]

        with open(annotation_file, "r") as f:
            boxes = json.load(f)

        label = np.zeros([self.grid_h, self.grid_w, (self.n_classes + (self.n_boxes * 5))])

        for box in boxes:
            x1 = float(box["xmin"])
            x2 = float(box["xmax"])
            y1 = float(box["ymin"])
            y2 = float(box["ymax"])

            x = (x1 + x2) / 2 / image_w
            y = (y1 + y2) / 2 / image_h
            w = (x2 - x1) / image_w
            h = (y2 - y1) / image_h

            loc = [self.grid_w * x, self.grid_h * y]
            loc_i = int(loc[1])
            loc_j = int(loc[0])

            y = loc[1] - loc_i
            x = loc[0] - loc_j

            cls_index = self.labels.index(box["label"])

            label[loc_i][loc_j][cls_index] = 1
            label[loc_i][loc_j][self.n_classes:self.n_classes + 4] = [x, y, w, h]
            label[loc_i][loc_j][self.n_classes + 4] = 1

        return cv2.resize(image, self.image_size), label


if __name__ == "__main__":
    nb_boxes = 2
    grid_w = 7
    grid_h = 7
    cell_w = 64
    cell_h = 64
    batch = 32

    image_shape = (grid_h * cell_h, grid_w * cell_w, 3)

    with open("labels.txt", "r") as f:
        labels = f.read().split("\n")

    model = build_model(len(labels), nb_boxes, grid_w, grid_h, cell_w, cell_h)

    train_data = ImageSequence(
        "data/train/images",
        "data/train/annotations",
        labels,
        batch,
        image_shape,
        grid_w,
        grid_h,
        nb_boxes
    )

    test_data = ImageSequence(
        "data/test/images",
        "data/test/annotations",
        labels,
        batch,
        image_shape,
        grid_w,
        grid_h,
        nb_boxes
    )

    model.fit(
        train_data,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=len(test_data),
        epochs=150,
        callbacks=[ReduceLROnPlateau()]
    )

    model.save("yolo_model", include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(load_model("yolo_model"))

    converter.allow_custom_ops = True
    lite_model = converter.convert()

    with open("lite-bird-classifier.tflite", 'wb') as f:
        f.write(lite_model)
