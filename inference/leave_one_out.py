import os
import shutil
import argparse

def move_leaveout_video(video_id):
    # Paths
    FRAME_SRC = "datasets/frames/train_videos"
    ANNOT_SRC = "datasets/annotations/json_train"

    FRAME_DST = "datasets/frames/leaveout_video"
    ANNOT_DST = "datasets/annotations/leaveout_json"

    os.makedirs(FRAME_DST, exist_ok=True)
    os.makedirs(ANNOT_DST, exist_ok=True)

    annot_filename = f"{video_id}.json"
    src_annot_path = os.path.join(ANNOT_SRC, annot_filename)
    dst_annot_path = os.path.join(ANNOT_DST, annot_filename)

    if not os.path.exists(src_annot_path):
        raise FileNotFoundError(f"Annotation not found: {src_annot_path}")

    src_frame_dir = os.path.join(FRAME_SRC, video_id)
    dst_frame_dir = os.path.join(FRAME_DST, video_id)

    if not os.path.exists(src_frame_dir):
        raise FileNotFoundError(f"Frames not found: {src_frame_dir}")

    # Move both
    shutil.move(src_annot_path, dst_annot_path)
    shutil.move(src_frame_dir, dst_frame_dir)

    print(f" Left out video: {video_id}")
    print(f" Moved annotation to: {dst_annot_path}")
    print(f" Moved frames to: {dst_frame_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave one video out for validation.")
    parser.add_argument("--video", required=True, help="Video ID to leave out, e.g., S4-Drill_side")
    args = parser.parse_args()
    move_leaveout_video(args.video)
