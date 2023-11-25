import os
from src.components.models import FakeImageDetector  # Import your model definition
from src.components.trainner import ModelTrainer 
from src.components.dataset_loader import DatasetLoader
from src.components.predictions import run_prediction_video
import config
import pdb

def main():
    path = os.path.join(config.CUR_DIR, 'Data','test_videos','aassnaulhq.mp4')
    pred = run_prediction_video(path)
    print(pred)
    '''
    data_dir = os.path.join(config.CUR_DIR, 'artifacts', 'data',)
    
    dataset_loader = DatasetLoader()
    train_dataset, validation_dataset , class_weights = dataset_loader.create_dataset(data_dir)
   
    model = FakeImageDetector()

    # Train the model
    trainer = ModelTrainer(model.model, train_dataset, validation_dataset, class_weights)
    history = trainer.train(class_weights)

    # Optionally, save the trained model
    trainde_model = os.path.join(config.CUR_DIR, 'artifacts', 'models')
    '''

if __name__ == "__main__":
    main()