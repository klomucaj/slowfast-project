import os

# Paths to frames and annotations
frames_dir = "datasets/frames/train"  # Adjust if needed
json_dir = "datasets/annotations/json_train"

# Get all frame folder names (videos)
frame_videos = set(os.listdir(frames_dir))

# Get all annotation JSON filenames (removing .json extension)
json_videos = set(f.replace(".json", "") for f in os.listdir(json_dir))

# Check mismatches
missing_annotations = frame_videos - json_videos
missing_frames = json_videos - frame_videos

# Print results
print(f"Videos with no JSON annotations: {missing_annotations}")
print(f"JSON files without frames: {missing_frames}")
