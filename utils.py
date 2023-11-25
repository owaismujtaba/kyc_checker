import os
import config
import pandas as pd
import numpy as np
import pdb
import json
import cv2
from multiprocessing import Manager, Pool
VIDEO_INDEX = 1

def get_train_list():
    
    train_list = list(os.listdir(config.TRAIN_SAMPLE_FOLDER))
    train_files = []
    for file in train_list:
        if file.endswith('.mp4'):
            train_files.append(file)
        else:
            train_meta_data = file
    train_meta_data =pd.read_json(os.path.join(config.TRAIN_SAMPLE_FOLDER, train_meta_data))
    train_meta_data = train_meta_data.T
    return train_files, train_meta_data
    
def get_test_list():
    
    test_list = list(os.listdir(config.TEST_FOLDER))
    test_files = []
    for file in test_list:
       if file.endswith('.mp4'):
           test_files.append(file)
       else:
           test_meta_data = file    
    test_meta_data = pd.read_json(os.path.join(config.TEST_FOLDER, test_meta_data))
    test_meta_data = test_meta_data.T     
    return test_files, test_meta_data

def extract_faces_from_video(path):
    
    cap = cv2.VideoCapture(path)
    frames = []
    
    face_detector = MTCNN()

    try:
        index = 0
        while True:
            ret, frame = cap.read()
            if index%10 != 0:
                continue
            index += 1
            if not ret:
                break
            try:
                features = face_detector.detect_faces(frame)
                if len(features)>0:
                    features = features[0]
                    x, y, w, h = features['box'][0], features['box'][1],features['box'][2],features['box'][3],
                    frame_roi = frame[y:y+h, x:x+w]
                    frame_roi = cv2.resize(frame_roi, (config.IMG_SIZE, config.IMG_SIZE))
                    frames.append(frame_roi)
            except:
                continue
            if len(frames) == config.MAX_FRAMES:
                break
    finally:
        cap.release()
    return np.array(frames)

def process_train_video(video, dir_path, labels):
    print("Processing {} Video:{}".format(VIDEO_INDEX, video))
    VIDEO_INDEX +=1
    file_path = os.path.join(dir_path, video)
    pdb.set_trace()
    frames = extract_faces_from_video(file_path)
    if len(frames) > 0:
        label = 1 if labels[video] == 'FAKE' else 0
        return frames, [label] * len(frames)
    else:
        return [], []

def load_train_videos_parallel():
    train_list, meta_data = get_train_list()
    dir_path = config.TRAIN_SAMPLE_FOLDER
    video_files = meta_data.index
    labels = meta_data.label
    
    # Create shared lists
    with Manager() as manager:
        X_train_shared = manager.list()
        y_train_shared = manager.list()

        print('Extracting frames from Videos (parallel processing)')
        with Pool() as pool:
            results = pool.map(process_train_video, video_files, labels)

        for result in results:
            X_train_shared.extend(result[0])
            y_train_shared.extend(result[1])

        # Convert shared lists to arrays
        X_train = np.concatenate(list(X_train_shared), axis=0)
        y_train = np.array(list(y_train_shared))

        train_data = os.path.join(config.CUR_DIR, 'src', 'artifacts')
        if not os.path.exists(train_data):
            os.makedirs(train_data, exist_ok=True)
        np.save(os.path.join(train_data, 'X_train.npy'), X_train)
        np.save(os.path.join(train_data, 'y_train.npy'), y_train)

        pdb.set_trace()
        
def create_images_from_numpy():
    
    dir_path = os.path.join(config.CUR_DIR, 'src' 'artifacts')
    real_dir = os.path.join(dir_path, 'Real')
    fake_dir = os.path.join(dir_path, 'Fake')
    os.makedirs(os.path.join(dir_path, real_dir), exist_ok=True)
    os.makedirs(os.path.join(dir_path, fake_dir), exist_ok=True)
    
    files = list(os.listdir(dir_path))
    numpy_files = [file for file in files if file.endswith('.npy')]
    print(numpy_files)
    
def convert_video_to_images():
    
    train_dir = config.TRAIN_SAMPLE_FOLDER
    video_files = list(os.listdir(train_dir))
    train_meta_data = [file for file in video_files if file.endswith('.json')][0]
    train_meta_data = pd.read_json(os.path.join(train_dir, train_meta_data))
    train_meta_data = train_meta_data.T.label
    video_files = [file for file in video_files if file.endswith('.mp4')]
    
    dataset_dir = os.path.join(config.CUR_DIR, 'artifacts', 'data')
    real_dir = os.path.join(dataset_dir, 'Real')
    fake_dir = os.path.join(dataset_dir, 'Fake')
    os.makedirs(os.path.join(dataset_dir, real_dir), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, fake_dir), exist_ok=True)
    
    for file in video_files:
        print("processing Video: ", file)
        label = train_meta_data[file]
        print('label: ', label)

        cap = cv2.VideoCapture(os.path.join(config.TRAIN_SAMPLE_FOLDER, file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * 500 / 1000)
        
        frame_count = 0
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
            ret, frame = cap.read()
            if not ret or frame_count * frame_interval >= total_frames:
                break
            frame_filename = file + '_'+str(frame_count) + '.jpeg'
            if label == "FAKE":
                frame_path = os.path.join(fake_dir, frame_filename)
            else:
                frame_path = os.path.join(real_dir, frame_filename)
                
            cv2.imwrite(frame_path, frame)

            frame_count += 1
        cap.release()
    
    
    #print(video_files)
    
    
convert_video_to_images()