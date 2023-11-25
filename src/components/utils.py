import os
import cv2
import config
import numpy as np
def extract_frames_for_test(video_path):
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * 500 / 1000)
        
    frames = []
    frame_count = 0
    while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
            ret, frame = cap.read()
            if not ret or frame_count * frame_interval >= total_frames:
                break
            frame = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE))
            frames.append(frame)
            
            frame_count += 1
    
    cap.release()
    return np.array(frames)