import os
from src.components.utils import extract_frames_for_test
from tensorflow.keras.models import load_model
from src.components.models import FakeImageDetector
import config
import numpy as np
import pdb
def run_prediction_video(video_path):
    #video_path = os.path.joi(config.CUR_DIR , 'Data','test_videos','aassnaulhq.mp4')
    frames = extract_frames_for_test(video_path)
    pdb.set_trace()
    model_path = config.MODEL_PATH
    #model = FakeImageDetector()
    #model.load_weights(model_path)
    model = load_model(model_path)
    predictions = model.predict(frames)
    
    threshold = config.THRESHOLD
    positives_count = np.sum(predictions > threshold)/frames.shape[0]
    negatives_count = np.sum(predictions <= threshold)/frames.shape[1]
    
    
    if positives_count > 0.7:
        return {video_path, 'FAKE'}
    else:
        return {video_path, 'REAL'}
    
    
def run_prediction_directory(dir_path):
    video_paths = list(os.listdir(dir_path))
    video_paths = [file for file in video_paths if file.endswitj('.mp4')]
    
    predictions = []
    
    for file in video_paths:
        print('processing ', file)
        pred = run_prediction_video(os.path.join(dir_path, file))
        predictions.append({file:pred})
        
    return predictions