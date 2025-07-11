import os
import cv2

VIDEO_SPLITS = ["train", "val", "test"]
VIDEO_BASE_DIR = "datasets/videos"
FRAME_OUTPUT_DIR = "datasets/frames"
FRAME_RATE = 1  # Extract 1 frame per second

def extract_frames(video_path, output_dir, frame_rate):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    #  Skip if already done
    if os.listdir(video_output_dir):
        print(f" Skipping {video_name} — frames already extracted.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"  Skipping (can't read FPS): {video_path}")
        return

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % round(fps / frame_rate)) == 0:
            frame_filename = os.path.join(video_output_dir, f"{saved:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1
        count += 1
    cap.release()
    print(f" Extracted {saved} frames from {video_path} into {video_output_dir}")

def extract_all_splits():
    for split in VIDEO_SPLITS:
        video_dir = os.path.join(VIDEO_BASE_DIR, split)
        output_dir = os.path.join(FRAME_OUTPUT_DIR, f"{split}_videos")
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(video_dir):
            if filename.lower().endswith((".avi", ".mp4")):
                video_path = os.path.join(video_dir, filename)
                extract_frames(video_path, output_dir, FRAME_RATE)

if __name__ == "__main__":
    extract_all_splits()
