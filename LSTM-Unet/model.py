from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling2D, UpSampling2D, Concatenate, ConvLSTM2D, BatchNormalization, TimeDistributed

def lstm_unet(input_shape=(15, 256, 256, 3)):
    inputs = Input(input_shape)
    
    # Encoder (Contracting Path)
    c1 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(c1)
    p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
    
    c2 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(p1)
    c2 = Dropout(0.1)(c2)
    c2 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(c2)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
    
    c3 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(p2)
    c3 = Dropout(0.2)(c3)
    c3 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(c3)
    p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)
    
    c4 = ConvLSTM2D(128, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(p3)
    c4 = Dropout(0.2)(c4)
    c4 = ConvLSTM2D(128, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(c4)
    p4 = TimeDistributed(MaxPooling2D((2, 2)))(c4)
    
    # Bottleneck
    c5 = ConvLSTM2D(256, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(p4)
    c5 = Dropout(0.3)(c5)
    c5 = ConvLSTM2D(256, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(c5)
    
    # Decoder (Expansive Path)
    u6 = TimeDistributed(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))(c5)
    u6 = Concatenate(axis=-1)([u6, c4])
    c6 = ConvLSTM2D(128, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(u6)
    c6 = Dropout(0.2)(c6)
    c6 = ConvLSTM2D(128, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(c6)
    
    u7 = TimeDistributed(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'))(c6)
    u7 = Concatenate(axis=-1)([u7, c3])
    c7 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(u7)
    c7 = Dropout(0.2)(c7)
    c7 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(c7)
    
    u8 = TimeDistributed(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'))(c7)
    u8 = Concatenate(axis=-1)([u8, c2])
    c8 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(u8)
    c8 = Dropout(0.1)(c8)
    c8 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(c8)
    
    u9 = TimeDistributed(Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same'))(c8)
    u9 = Concatenate(axis=-1)([u9, c1])
    c9 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(u9)
    c9 = Dropout(0.1)(c9)
    c9 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', 
                    return_sequences=True)(c9)
    
    # Output layer
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid'))(c9)
    
    return Model(inputs, outputs)