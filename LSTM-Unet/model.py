from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling2D, UpSampling2D, Concatenate, ConvLSTM2D, BatchNormalization, TimeDistributed

def lstm_unet(input_shape=(15, 256, 256, 3)):
    inputs = Input(input_shape)

    # Encoder
    c1 = ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True)(inputs)
    c1 = BatchNormalization()(c1)
    p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)

    c2 = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(p1)
    c2 = BatchNormalization()(c2)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)

    c3 = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(p2)
    c3 = BatchNormalization()(c3)
    p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

    c4 = ConvLSTM2D(128, (3, 3), padding='same', return_sequences=True)(p3)
    c4 = BatchNormalization()(c4)
    p4 = TimeDistributed(MaxPooling2D((2, 2)))(c4)

    # Bottleneck
    c5 = ConvLSTM2D(256, (3, 3), padding='same', return_sequences=True)(p4)
    c5 = BatchNormalization()(c5)

    # Decoder
    u6 = TimeDistributed(UpSampling2D((2, 2)))(c5)
    u6 = Concatenate(axis=-1)([u6, c4])
    c6 = ConvLSTM2D(128, (3, 3), padding='same', return_sequences=True)(u6)
    c6 = BatchNormalization()(c6)

    u7 = TimeDistributed(UpSampling2D((2, 2)))(c6)
    u7 = Concatenate(axis=-1)([u7, c3])
    c7 = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(u7)
    c7 = BatchNormalization()(c7)

    u8 = TimeDistributed(UpSampling2D((2, 2)))(c7)
    u8 = Concatenate(axis=-1)([u8, c2])
    c8 = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(u8)
    c8 = BatchNormalization()(c8)

    u9 = TimeDistributed(UpSampling2D((2, 2)))(c8)
    u9 = Concatenate(axis=-1)([u9, c1])
    c9 = ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True)(u9)
    c9 = BatchNormalization()(c9)

    # Output layer
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid'))(c9)

    return Model(inputs, outputs)