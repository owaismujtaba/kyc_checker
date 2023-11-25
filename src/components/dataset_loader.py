import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
import os
import json
import pdb
class DatasetLoader:
    def __init__(self):
        self.img_size = (config.IMG_SIZE, config.IMG_SIZE)
        self.batch_size = config.BATCH_SIZE
        self.shuffle = True
        self.data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    def create_dataset(self, data_dir):
        
        class_counts = {}
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = len(os.listdir(class_path))
                
        total_samples = sum(class_counts.values())
       
        train_dataset = self.data_generator.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=self.shuffle,
        )
        class_indicies = train_dataset.class_indices
        class_weights = {class_indicies[class_name]: count / total_samples for class_name, count in class_counts.items()}
        print(class_weights)
        validation_dataset = self.data_generator.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=self.shuffle
        )

        mapping_path = os.path.join(config.CUR_DIR, 'artifacts', 'mappings.json')
        with open(mapping_path, 'w') as json_file:
            json.dump(train_dataset.class_indices, json_file)
        
        
        return train_dataset, validation_dataset, class_weights


