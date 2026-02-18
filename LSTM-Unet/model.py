from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, Dropout, TimeDistributed, MaxPooling2D, Concatenate, Conv2DTranspose, Conv2D, BatchNormalization, ReLU

def lstm_unet(input_shape=(15, 256, 256, 3)):
    """
    LSTM-UNet architecture for temporal image segmentation.

    Identical to the baseline UNet in every way, with two changes:
      1. All layers are wrapped in TimeDistributed to handle (time_steps, H, W, C) input.
      2. The second Conv2D in the bottleneck (c5) is replaced with ConvLSTM2D
         to introduce temporal processing — this is the ONLY architectural
         difference from the baseline.
    """
    inputs = Input(input_shape)

    # -------------------------------------------------------------------------
    # ENCODER (Contracting Path)
    # -------------------------------------------------------------------------

    # Block 1
    c1 = TimeDistributed(Conv2D(16, (3, 3), padding='same'))(inputs)
    c1 = TimeDistributed(BatchNormalization())(c1)
    c1 = TimeDistributed(ReLU())(c1)
    c1 = TimeDistributed(Dropout(0.1))(c1)
    c1 = TimeDistributed(Conv2D(16, (3, 3), padding='same'))(c1)
    c1 = TimeDistributed(BatchNormalization())(c1)
    c1 = TimeDistributed(ReLU())(c1)
    p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)

    # Block 2
    c2 = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(p1)
    c2 = TimeDistributed(BatchNormalization())(c2)
    c2 = TimeDistributed(ReLU())(c2)
    c2 = TimeDistributed(Dropout(0.1))(c2)
    c2 = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(c2)
    c2 = TimeDistributed(BatchNormalization())(c2)
    c2 = TimeDistributed(ReLU())(c2)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)

    # Block 3
    c3 = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(p2)
    c3 = TimeDistributed(BatchNormalization())(c3)
    c3 = TimeDistributed(ReLU())(c3)
    c3 = TimeDistributed(Dropout(0.2))(c3)
    c3 = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(c3)
    c3 = TimeDistributed(BatchNormalization())(c3)
    c3 = TimeDistributed(ReLU())(c3)
    p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

    # Block 4
    c4 = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(p3)
    c4 = TimeDistributed(BatchNormalization())(c4)
    c4 = TimeDistributed(ReLU())(c4)
    c4 = TimeDistributed(Dropout(0.2))(c4)
    c4 = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(c4)
    c4 = TimeDistributed(BatchNormalization())(c4)
    c4 = TimeDistributed(ReLU())(c4)
    p4 = TimeDistributed(MaxPooling2D((2, 2)))(c4)

    # -------------------------------------------------------------------------
    # BOTTLENECK — 256 filters
    # -------------------------------------------------------------------------
    c5 = TimeDistributed(Conv2D(256, (3, 3), padding='same'))(p4)
    c5 = TimeDistributed(BatchNormalization())(c5)
    c5 = TimeDistributed(ReLU())(c5)
    c5 = TimeDistributed(Dropout(0.3))(c5)
    c5 = ConvLSTM2D(256, (3, 3), padding='same',
                    return_sequences=True, activation='tanh')(c5)
    c5 = TimeDistributed(BatchNormalization())(c5)

    # -------------------------------------------------------------------------
    # DECODER (Expansive Path)
    # -------------------------------------------------------------------------

    # Block 6
    u6 = TimeDistributed(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))(c5)
    u6 = TimeDistributed(BatchNormalization())(u6)
    u6 = TimeDistributed(ReLU())(u6)
    u6 = Concatenate(axis=-1)([u6, c4])
    c6 = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(u6)
    c6 = TimeDistributed(BatchNormalization())(c6)
    c6 = TimeDistributed(ReLU())(c6)
    c6 = TimeDistributed(Dropout(0.2))(c6)
    c6 = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(c6)
    c6 = TimeDistributed(BatchNormalization())(c6)
    c6 = TimeDistributed(ReLU())(c6)

    # Block 7
    u7 = TimeDistributed(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'))(c6)
    u7 = TimeDistributed(BatchNormalization())(u7)
    u7 = TimeDistributed(ReLU())(u7)
    u7 = Concatenate(axis=-1)([u7, c3])
    c7 = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(u7)
    c7 = TimeDistributed(BatchNormalization())(c7)
    c7 = TimeDistributed(ReLU())(c7)
    c7 = TimeDistributed(Dropout(0.2))(c7)
    c7 = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(c7)
    c7 = TimeDistributed(BatchNormalization())(c7)
    c7 = TimeDistributed(ReLU())(c7)

    # Block 8
    u8 = TimeDistributed(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'))(c7)
    u8 = TimeDistributed(BatchNormalization())(u8)
    u8 = TimeDistributed(ReLU())(u8)
    u8 = Concatenate(axis=-1)([u8, c2])
    c8 = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(u8)
    c8 = TimeDistributed(BatchNormalization())(c8)
    c8 = TimeDistributed(ReLU())(c8)
    c8 = TimeDistributed(Dropout(0.1))(c8)
    c8 = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(c8)
    c8 = TimeDistributed(BatchNormalization())(c8)
    c8 = TimeDistributed(ReLU())(c8)

    # Block 9
    u9 = TimeDistributed(Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same'))(c8)
    u9 = TimeDistributed(BatchNormalization())(u9)
    u9 = TimeDistributed(ReLU())(u9)
    u9 = Concatenate(axis=-1)([u9, c1])
    c9 = TimeDistributed(Conv2D(16, (3, 3), padding='same'))(u9)
    c9 = TimeDistributed(BatchNormalization())(c9)
    c9 = TimeDistributed(ReLU())(c9)
    c9 = TimeDistributed(Dropout(0.1))(c9)
    c9 = TimeDistributed(Conv2D(16, (3, 3), padding='same'))(c9)
    c9 = TimeDistributed(BatchNormalization())(c9)
    c9 = TimeDistributed(ReLU())(c9)

    # Per-frame segmentation output
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid'))(c9)

    return Model(inputs, outputs)