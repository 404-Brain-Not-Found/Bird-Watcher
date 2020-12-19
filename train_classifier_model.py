from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from DataGenerator import DataGenerator


def build_model(input_shape=(224, 224, 3), number_classes=255) -> Model:
    mobile = MobileNet(input_shape=input_shape, include_top=False)

    for layer in mobile.layers:
        layer.trainable = False

    x = mobile.layers[-1].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(number_classes, activation="sigmoid")(x)

    return Model(mobile.input, x)


if __name__ == "__main__":
    # generator = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     rescale=1/255,
    #     rotation_range=180,
    #     brightness_range=(0.2, 1.8)
    # )


    batch_size = 255
    image_size = (224, 224)

    train_data = DataGenerator("/home/thomasquirk/PycharmProjects/Bird Classifier/data/train")
    test_data = DataGenerator("/home/thomasquirk/PycharmProjects/Bird Classifier/data/test")

    # print(test_data.class_indices)

    model = build_model(image_size + (3,), 226)

    model.compile(Adam(learning_rate=0.1),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )

    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(patience=5)

    model.fit(
        train_data,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=len(test_data),
        epochs=150,
        callbacks=[early, reduce_lr]
    )

    model.save("bird_classifier")
    lite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()

    with open("lite-bird-classifier.tflite", 'wb') as f:
        f.write(lite_model)
