import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU,
    Add, GlobalAveragePooling2D, Dense,
    Conv2DTranspose, ReLU
)
from tensorflow.keras.models import Model


def resnet(img_height, img_width, img_channels):
    input_shape = (img_height, img_width, img_channels)
    inputs = Input(shape=input_shape)

    # DOWNSAMPLING
    # Initial conv
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual Block 1
    shortcut = x
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)

    # Residual Block 2
    shortcut = Conv2D(128, (1, 1), strides=2, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = ReLU()(x)

    # Residual Block 3
    shortcut = Conv2D(256, (1, 1), strides=2, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = ReLU()(x)

    # UPSAMPLING

    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Classification head
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs)
