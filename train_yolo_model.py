from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from DataGenerator import DataGenerator
from yolo_utils import YoloReshape


def build_model(n_classes, nb_boxes=1, grid_w=7, grid_h=7, cell_w=64, cell_h=64):

    base_model = MobileNetV2(input_shape=(grid_h * cell_h, grid_w * cell_w, 3), weights="imagenet", include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-1].output
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = Dense(grid_w * grid_h * (nb_boxes * 5 + n_classes), activation='sigmoid')(x)
    x = YoloReshape(target_shape=(grid_h, grid_w, (nb_boxes * 5 + n_classes)), n_classes=n_classes, n_boxes=nb_boxes)(x)

    return Model(base_model.input, x)
