import os
import cv2
from tqdm import tqdm

# Paths to video folders
VIDEO_DIRS = {
    'train': "datasets/videos/train_videos",
    'val': "datasets/videos/val_videos",
    'test': "datasets/videos/test_videos"
}

# Target folder for extracted frames
FRAME_DIR = "datasets/frames"


# Frame extraction settings
FPS = 1  # Extract 1 frame per second

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_frames(video_path, output_folder, fps=1):
    """
    Extract frames from a video at 1 FPS and save them in the output folder.
    """
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    frame_count = 0
    sec = 0  # Start at 0 seconds

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get original FPS of video
    interval = int(frame_rate) if frame_rate > 0 else 1  # Capture every second

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # Move to the next second
        ret, frame = cap.read()

        if not ret:
            break  # Stop when the video ends

        frame_filename = f"{video_name}_frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        sec += 1  # Move to next second

    cap.release()
    print(f"Extracted {frame_count} frames from {video_name}")

def process_videos():
    """
    Extract frames from videos in train, val, and test folders
    and store them in corresponding frame directories.
    """
    for split, video_folder in VIDEO_DIRS.items():
        output_dir = os.path.join(FRAME_DIR, split)
        ensure_dir(output_dir)

        videos = [v for v in os.listdir(video_folder) if v.endswith(('.mp4', '.avi', '.mov'))]

        for video in tqdm(videos, desc=f"Processing {split} videos"):
            video_path = os.path.join(video_folder, video)
            extract_frames(video_path, output_dir, FPS)

if __name__ == "__main__":
    print("🚀 Starting frame extraction from videos...")
    process_videos()
    print("🎉 Frame extraction complete!")
