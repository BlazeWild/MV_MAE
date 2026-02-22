import os
import cv2
from multiprocessing import Pool
from tqdm import tqdm

base_dir = '/home/blaze/MV_MAE/datasets/UAVHuman_240p_mp4'
video_dir = os.path.join(base_dir, 'Action_Videos')
train_split = os.path.join(base_dir, 'train_split.txt')

def check_video(line):
    video_name = line.strip().split()[0]
    video_path = os.path.join(video_dir, video_name)
    if not os.path.exists(video_path):
        return video_name
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_name
    
    # Try reading a frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return video_name
    return None

if __name__ == '__main__':
    with open(train_split, 'r') as f:
        lines = f.readlines()
        
    total = len(lines)
    corrupted = []
    
    with Pool(16) as p:
        results = list(tqdm(p.imap(check_video, lines), total=total, desc="Checking videos"))
        
    corrupted = [r for r in results if r is not None]
    
    print(f"Total training video files: {total}")
    print(f"Corrupted files: {len(corrupted)}")
    
    with open('corrupted_videos_info.txt', 'w') as out_f:
        out_f.write(f"Total training video files: {total}\n")
        out_f.write(f"Corrupted files: {len(corrupted)}\n")
        for c in corrupted:
            out_f.write(f"  - {c}\n")
