from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import random
import numpy as np


class DataGenerator(Sequence):

    def __init__(self, dir):
        self.labels = []
        self.min_files = 9999999
        self.image_dir = dir
        self.generator = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1/255,
            rotation_range=180,
            brightness_range=(0.2, 1.8)
        )

        for folder in os.listdir(dir):
            if os.path.isdir(os.path.join(dir, folder)):
                self.labels.append(folder)
                files = [os.path.join(folder, ) for file in os.listdir(os.path.join(dir, folder)) if os.path.isfile(os.path.join(dir, folder, file))]
                self.min_files = min(self.min_files, len(files))

    def __len__(self):
        return self.min_files

    def __getitem__(self, item):
        x = []
        y = []

        for index, label in enumerate(self.labels):
            image_file = os.listdir(os.path.join(self.image_dir, label))[item]

            image = cv2.imread(os.path.join(self.image_dir, label, image_file), cv2.COLOR_BGR2RGB)
            random_x = random.randint(0, int(image.shape[1] / 2))
            random_y = random.randint(0, int(image.shape[0] / 2))
            random_crop = image[random_y:random.randint(random_y + 1, image.shape[0]),
                                random_x:random.randint(random_x + 1, image.shape[1])]
            random_crop = cv2.resize(random_crop, (224, 224))
            image = cv2.resize(image, (224, 224))

            image_label = [0] * (len(self.labels) + 1)
            image_label[index] = 1

            crop_label = [0] * (len(self.labels) + 1)
            crop_label[-1] = 1

            x.append(image)
            y.append(image_label)

            x.append(random_crop)
            y.append(crop_label)

        out_x = []
        out_y = []
        x = np.array(x)
        y = np.array(y)

        for x_batch, y_batch in self.generator.flow(x, y):
            out_x.extend(x_batch)
            out_y.extend(y_batch)

            if len(self.labels) * 2 < len(out_x):
                break

        return np.array(out_x), np.array(out_y)
