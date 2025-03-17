import os
import cv2
import json

class ExtractFrames:
    def __init__(self, video_base, annotation_base, output_base, frame_rate=1):
        self.video_dirs = {
            "train": os.path.join(video_base, "train_videos"),
            "val": os.path.join(video_base, "val_videos"),
            "test": os.path.join(video_base, "test_videos"),
        }
        
        self.annotation_dirs = {
            "train": os.path.join(annotation_base, "srt_train"),
            "val": os.path.join(annotation_base, "srt_val"),
            "test": os.path.join(annotation_base, "srt_test"),
        }
        
        self.output_dirs = {
            "train": os.path.join(output_base, "train_frames"),
            "val": os.path.join(output_base, "val_frames"),
            "test": os.path.join(output_base, "test_frames"),
        }
        
        self.frame_rate = frame_rate  # Extract 1 frame per second
        
        # Ensure output directories exist
        for split in self.output_dirs.values():
            os.makedirs(split, exist_ok=True)

    def get_annotation_file(self, video_name, split):
        """Finds the matching annotation file based on S1, S2, S3, S4 in the filename."""
        annotation_folder = self.annotation_dirs[split]
        possible_annotations = os.listdir(annotation_folder)
        
        for annotation in possible_annotations:
            if any(marker in video_name for marker in ["S1", "S2", "S3", "S4"]):
                if marker in annotation: # type: ignore
                    return os.path.join(annotation_folder, annotation)
        
        return None  # No matching annotation found

    def parse_annotations(self, annotation_path):
        """Parses an SRT-like annotation file into a structured list."""
        annotations = []
        with open(annotation_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i in range(0, len(lines), 4):  # Assuming each block has 4 lines
            if i + 2 < len(lines):
                timestamp = lines[i + 1].strip()  # e.g., "00:04:21,231 --> 00:04:22,531"
                label = lines[i + 2].strip()  # e.g., "316 - Door1"
                annotations.append({"timestamp": timestamp, "label": label})

        return annotations

    def extract_frames(self):
        """Extract frames from videos and match them with annotations."""
        for split, video_folder in self.video_dirs.items():
            output_folder = self.output_dirs[split]
            
            if not os.path.exists(video_folder):
                print(f" Skipping {video_folder}, folder does not exist.")
                continue

            for video_file in sorted(os.listdir(video_folder)):
                if not video_file.endswith((".mp4", ".avi", ".mov")):
                    continue
                
                video_path = os.path.join(video_folder, video_file)
                annotation_path = self.get_annotation_file(video_file, split)
                
                if annotation_path:
                    annotations = self.parse_annotations(annotation_path)
                    print(f" Processing: {video_file} |  Annotation: {os.path.basename(annotation_path)}")
                else:
                    print(f" No annotation found for {video_file}, skipping.")
                    continue

                # Extract frames and match with annotations
                self.process_video(video_path, output_folder, video_file, annotations)

    def process_video(self, video_path, output_folder, video_name, annotations):
        """Extracts frames from a video and assigns them corresponding annotation texts."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, fps // self.frame_rate)
        frame_count = 0
        saved_frames = 0

        video_output_dir = os.path.join(output_folder, video_name.rsplit(".", 1)[0])
        os.makedirs(video_output_dir, exist_ok=True)

        annotation_idx = 0  # Keep track of annotation sequence

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if the video ends

            if frame_count % frame_interval == 0 and annotation_idx < len(annotations):
                # Assign annotation text to this frame
                annotation_text = annotations[annotation_idx]["label"]
                
                # Save frame
                frame_filename = os.path.join(video_output_dir, f"frame_{saved_frames:05d}.jpg")
                cv2.imwrite(frame_filename, frame)

                # Save metadata in JSON
                json_filename = frame_filename.replace(".jpg", ".json")
                metadata = {
                    "video": video_name,
                    "frame_number": saved_frames,
                    "annotation": annotation_text
                }

                with open(json_filename, "w", encoding="utf-8") as json_file:
                    json.dump(metadata, json_file, indent=4)

                saved_frames += 1
                annotation_idx += 1  # Move to the next annotation

            frame_count += 1

        cap.release()
        print(f" Extracted {saved_frames} frames from {video_name}")

# Main execution
if __name__ == "__main__":
    video_base = "datasets/videos"
    annotation_base = "datasets/annotations"
    output_base = "datasets/frames"

    frame_extractor = ExtractFrames(video_base, annotation_base, output_base, frame_rate=1)
    frame_extractor.extract_frames()
