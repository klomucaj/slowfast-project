import os
import shutil
import random

# === CONFIG ===
VIDEO_DIR = "datasets/videos/videosGroup"
SRT_DIR = "datasets/annotations/srt_merged"
OUTPUT_VIDEO_DIR = "datasets/videos"
OUTPUT_SRT_DIR = "datasets/annotations"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# === LOGIC ===
def split_dataset():
    # Support .avi and .mp4 files
    all_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".avi", ".mp4"))]
    all_videos.sort()
    random.shuffle(all_videos)

    total = len(all_videos)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_videos = all_videos[:train_count]
    val_videos = all_videos[train_count:train_count+val_count]
    test_videos = all_videos[train_count+val_count:]

    splits = {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos
    }

    for split, files in splits.items():
        # Create output folders
        split_video_dir = os.path.join(OUTPUT_VIDEO_DIR, split)
        split_srt_dir = os.path.join(OUTPUT_SRT_DIR, f"srt_{split}")
        os.makedirs(split_video_dir, exist_ok=True)
        os.makedirs(split_srt_dir, exist_ok=True)

        for filename in files:
            basename = os.path.splitext(filename)[0]

            # Copy video
            video_src = os.path.join(VIDEO_DIR, filename)
            video_dst = os.path.join(split_video_dir, filename)
            shutil.copy(video_src, video_dst)

            # Copy SRT (if exists)
            srt_file = f"{basename}.srt"
            srt_src = os.path.join(SRT_DIR, srt_file)
            srt_dst = os.path.join(split_srt_dir, srt_file)
            if os.path.exists(srt_src):
                shutil.copy(srt_src, srt_dst)
            else:
                print(f"  Warning: No matching .srt found for {filename}")

        print(f" Split '{split}': {len(files)} videos → {split_video_dir}, {split_srt_dir}")

if __name__ == "__main__":
    split_dataset()
