import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Add, Conv2DTranspose,
    TimeDistributed, LSTM, Dense, ConvLSTM2D
)
from tensorflow.keras.models import Model


def lstm_resnet(img_height, img_width, img_channels, time_steps):
    """
    LSTM-ResNet architecture for temporal image segmentation.

    Identical to the baseline ResNet in every way, with two changes:
      1. All layers are wrapped in TimeDistributed to handle (time_steps, H, W, C) input.
      2. The second Conv2D in Residual Block 3 (bottleneck) is replaced with
         ConvLSTM2D to introduce temporal processing — this is the ONLY
         architectural difference from the baseline.
    """
    inputs = Input(shape=(time_steps, img_height, img_width, img_channels))

    # -------------------------------------------------------------------------
    # ENCODER
    # -------------------------------------------------------------------------

    # Initial conv
    x = TimeDistributed(Conv2D(64, (7, 7), strides=2, padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    # Residual Block 1
    shortcut = x
    x = TimeDistributed(Conv2D(64, (3, 3), strides=1, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=1, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Add()([shortcut, x])
    x = TimeDistributed(ReLU())(x)

    # Residual Block 2
    shortcut = TimeDistributed(Conv2D(128, (1, 1), strides=2, padding='same'))(x)
    shortcut = TimeDistributed(BatchNormalization())(shortcut)
    x = TimeDistributed(Conv2D(128, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)
    x = TimeDistributed(Conv2D(128, (3, 3), strides=1, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Add()([shortcut, x])
    x = TimeDistributed(ReLU())(x)

    # Residual Block 3
    shortcut = TimeDistributed(Conv2D(256, (1, 1), strides=2, padding='same'))(x)
    shortcut = TimeDistributed(BatchNormalization())(shortcut)
    x = TimeDistributed(Conv2D(256, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)
    x = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same',
                   return_sequences=True, activation='tanh')(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Add()([shortcut, x])
    x = TimeDistributed(ReLU())(x)

    # -------------------------------------------------------------------------
    # DECODER
    # -------------------------------------------------------------------------

    x = TimeDistributed(Conv2DTranspose(128, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    x = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    x = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    # Per-frame segmentation output
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid'))(x)

    return Model(inputs=inputs, outputs=outputs)
