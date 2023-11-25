from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,BatchNormalization, MaxPooling2D
from tensorflow.keras.applications import VGG16
import config

class FakeImageDetector:
    def __init__(self) -> None:
        
        self.vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(config.IMG_SIZE,config.IMG_SIZE, 3))
        
        for layer in self.vgg16.layers:
            layer.trainable = False
        
        block3_output = self.vgg16.get_layer('block3_conv3').output
        #trace()
        x = Conv2D(64, (3,3), activation='relu')(block3_output)
        x = BatchNormalization()(x)
        #x = Conv2D(512, (3,3), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3,3), activation='relu')(x)
        print(x.shape)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)   
        x = Dense(512, activation='relu')(x)    
        x = Dense(256, activation='relu')(x)
        output_layer = Dense(config.OUTPUT_CLASSES, activation='softmax')(x)
        
        self.model = Model(inputs=self.vgg16.input, outputs=output_layer)
        
        
        



class FakeImageDetectorBuilder:
    def __init__(self):
        input_shape = (config.IMG_SIZE, config.IMG_SIZE, 3)
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(config.IMG_WIDTH,config.IMG_HEIGHT, 3))
        
        for layer in self.vgg16.layers:
            layer.trainable = False
        
        block3_output = self.vgg16.get_layer('block3_conv3').output
        

        # Convolutional layers
        conv1 = Conv2D(32, (3, 3), activation='relu')(block3_output)
        maxpool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu')(maxpool1)
        maxpool2 = MaxPooling2D((2, 2))(conv2)
        conv3 = Conv2D(128, (3, 3), activation='relu')(maxpool2)
        maxpool3 = MaxPooling2D((2, 2))(conv3)

        # Flatten layer
        flatten = Flatten()(maxpool3)

        # Dense layers
        dense1 = Dense(128, activation='relu')(flatten)
        dropout = Dropout(0.5)(dense1)
        output = Dense(1, activation='sigmoid')(dropout)

        model = Model(inputs=vgg16.input, outputs=output)
        return model


