import os
import shutil
from sklearn.model_selection import train_test_split

def split_videos(source_dir, dest_dir):
    videos = os.listdir(source_dir)
    train, test = train_test_split(videos, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)
    
    for split, name in zip([train, val, test], ['train', 'val', 'test']):
        os.makedirs(f"{dest_dir}/{name}", exist_ok=True)
        for video in split:
            shutil.move(os.path.join(source_dir, video), f"{dest_dir}/{name}/{video}")

if __name__ == "__main__":
    split_videos("datasets/videos", "datasets/videos")
