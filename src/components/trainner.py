import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import config
import pdb
#from tensorflow.keras.models import save_model
import os

class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, class_weights):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = config.EPOCHS
        self.class_weights = class_weights

    def train(self, patience=3):
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        print(self.class_weights)
        # Train the model with early stopping
        history = self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            validation_data=self.val_dataset,
            callbacks=[early_stopping],
            class_weight = self.class_weights
        )
        trainde_model = config.MODEL_PATH
        os.makedirs(trainde_model, exist_ok=True)
        pdb.set_trace()
        self.model.save(trainde_model)
        
        self.model.evaluate(self.val_dataset)
        #save_model(os.path.join(trainde_model+'fake_image_detector.h5'), self.model)
        return history